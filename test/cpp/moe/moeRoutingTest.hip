#include <hip/hip_runtime.h>

#include "../../../src/hip/hipHostCommon.h"
#include "../../../src/hip/taskSystem/taskManager.h"
#include "../../../src/hip/tasks/internal/gemmTileBlock.h"
#include "../../../src/hip/tasks/internal/softmaxBlock.h"
#include "../../../src/hip/tasks/internal/topkBlock.h"
#include "../../../src/hip/utils/hipDeviceUtils.h"
#include "../reference/refMoeGate.h"
#include "../utils/utils.h"

constexpr int TOKENS_NUM = 48;
constexpr int EXPERTS_NUM = 128;
constexpr int HIDDEN_SIZE = 2048;
constexpr int EXPERT_INTERMEDIATE_SIZE = 768;
constexpr int TOPK = 8;
constexpr int THREADS = 128;
constexpr int BLOCKS = 304;

// Gate => tokens[TOKENS_NUM,HIDDEN_SIZE] * gate[HIDDEN_SIZE,EXPERTS_NUM] =>
// routing[TOKENS_NUM,EXPERTS_NUM]
using TGateGemmTileParams =
    GemmTileParams<EXPERTS_NUM, HIDDEN_SIZE, 1, EXPERTS_NUM, 32, THREADS>;

constexpr int EXPERT_TILE_SIZE = 256;
// Expert => token[1,HIDDEN_SIZE] *
// expertW[HIDDEN_SIZE,EXPERT_INTERMEDIATE_SIZE] => [1,EXPERT_INTERMEDIATE_SIZE]
using TExpertGemmTileParams =
    GemmTileParams<EXPERT_INTERMEDIATE_SIZE, HIDDEN_SIZE, 1, EXPERT_TILE_SIZE,
                   32, THREADS>;

struct __align__(16) ExpertTileTask {
  const float* expertWeights;
  const float* tokens;
  float* result;
  float expertWeight;
  int tokenIdx;
  int expertIdx;
  int blockTileColStartIdx;
};

__device__ void ExecuteTask(const ExpertTileTask& task, void* sharedMemPool) {
  if (threadIdx.x == 65) {
    HIP_DEVICE_LOG(
        "Worker %i: Executes tile no: %i, tokenIdx: %i, expert: %i, weight: "
        "%f\n",
        blockIdx.x, task.blockTileColStartIdx, task.tokenIdx, task.expertIdx,
        task.expertWeight);
  }
}

using TaskManagerType = TaskManager<ExpertTileTask, 8192>;
__constant__ TaskManagerType globalTaskManager;

template <typename TGEMM_TILE>
__global__ void refRoutingKernel(const float* __restrict__ tokens,
                                 const float* __restrict__ gateWeights,
                                 float* __restrict__ topkVals,
                                 int* __restrict__ topIdx, int TOKENS_NUM) {
  __shared__ char sharedMemPool[TGEMM_TILE::SHARED_MEM_NEEDES_BYTES];

  for (int tokenIdx = blockIdx.x; tokenIdx < TOKENS_NUM;
       tokenIdx += gridDim.x) {
    typename TGEMM_TILE::TOutputType out_regs[TGEMM_TILE::THREAD_OUTPUT_SIZE];

    GemmTile_block<TGEMM_TILE>(tokens, gateWeights, out_regs, tokenIdx, 0,
                               sharedMemPool);
    out_regs[0] =
        Softmax_block<TGEMM_TILE::THREADS, float>(out_regs[0], sharedMemPool);

    char* sharedMemPoolBytes = reinterpret_cast<char*>(sharedMemPool);
    float* s_topkVals = reinterpret_cast<float*>(sharedMemPoolBytes);
    int* s_topkIdx =
        reinterpret_cast<int*>(sharedMemPoolBytes + sizeof(float) * TOPK);
    int* nextPool = s_topkIdx + TOPK;

    Topk8_block<TGEMM_TILE::THREADS, float>(out_regs[0], s_topkVals, s_topkIdx,
                                            nextPool);

    // At this point each participating block has topkVals and TopkIdx for
    // selected token in shared mem.

    const int tid = threadIdx.x;
    if (threadIdx.x < TOPK) {
      topkVals[tokenIdx * 8 + tid] = s_topkVals[tid];
      topIdx[tokenIdx * 8 + tid] = s_topkIdx[tid];
    }

    // Warp 0 pushes all the tasks...
    for (int i = 0; i < TOPK; ++i) {
      ExpertTileTask rTask;
      rTask.expertWeights = nullptr;
      rTask.tokens = nullptr;
      rTask.result = nullptr;
      rTask.tokenIdx = tokenIdx;
      rTask.expertIdx = s_topkIdx[i];
      rTask.expertWeight = s_topkVals[i];
      rTask.blockTileColStartIdx = 0;

      globalTaskManager.PushTask_warp(&rTask);

      rTask.blockTileColStartIdx = 1;

      globalTaskManager.PushTask_warp(&rTask);

      rTask.blockTileColStartIdx = 2;

      globalTaskManager.PushTask_warp(&rTask);
    }
  }

  // Tokens dispatched to workers.
  __syncthreads();

  while (true) {
    bool* shouldExitNow_s = reinterpret_cast<bool*>(sharedMemPool);
    ExpertTileTask* task_s =
        reinterpret_cast<ExpertTileTask*>(sharedMemPool + sizeof(bool));
    // Get new task:
    const bool noMoreTasks = globalTaskManager.WaitAndPopTask_warp(task_s);

    if (threadIdx.x == 112) {
      HIP_DEVICE_LOG("Worker: Got new task\n");
    }

    if (threadIdx.x == 0) {
      *shouldExitNow_s = noMoreTasks;
    }

    __syncthreads();
    if (*shouldExitNow_s) {
      break;
    }

    // Block cooperatively execute task:
    ExpertTileTask rTask;
    rTask.expertWeights = task_s->expertWeights;
    rTask.tokens = task_s->tokens;
    rTask.result = task_s->result;
    rTask.tokenIdx = task_s->tokenIdx;
    rTask.expertIdx = task_s->expertIdx;
    rTask.expertWeight = task_s->expertWeight;
    rTask.blockTileColStartIdx = task_s->blockTileColStartIdx;
    __syncthreads();
    ExecuteTask(rTask, sharedMemPool);
  }
}

hipError_t refHipRouting(const float* tokens, const float* gateWeights,
                         float* topkVals, int* topkIdx, int tokensNum,
                         int expertsNum, int hiddenSize) {
  assert(hiddenSize == HIDDEN_SIZE);
  assert(tokensNum == TOKENS_NUM);
  assert(expertsNum == EXPERTS_NUM);

  dim3 grid(BLOCKS);
  dim3 block(TGateGemmTileParams::THREADS);

  hipStream_t stream = 0;
  TaskManagerType managerHost;
  const int expectedNumTasks =
      TOKENS_NUM * TOPK * (EXPERT_INTERMEDIATE_SIZE / EXPERT_TILE_SIZE);
  HIP_ERROR_CHECK(managerHost.Init(stream, expectedNumTasks));

  HIP_ERROR_CHECK(hipMemcpyToSymbolAsync(HIP_SYMBOL(globalTaskManager),
                                         &managerHost, sizeof(TaskManagerType),
                                         0, hipMemcpyHostToDevice, stream));

  refRoutingKernel<TGateGemmTileParams><<<grid, block, 0, 0>>>(
      tokens, gateWeights, topkVals, topkIdx, tokensNum);

  HIP_ERROR_CHECK(managerHost.Deinit(stream));

  return hipSuccess;
}

TEST(MoeRouting, basicTest) {
  constexpr int TOKENS_SIZE = TOKENS_NUM * HIDDEN_SIZE;
  constexpr int GATE_WEIGHTS_SIZE = HIDDEN_SIZE * EXPERTS_NUM;
  constexpr int ROUTING_SIZE = TOKENS_NUM * TOPK;

  std::vector<float> tokens_host =
      RandomVector<float>(TOKENS_SIZE, -0.1f, 0.1f);
  std::vector<float> gateWeights_host =
      RandomVector<float>(GATE_WEIGHTS_SIZE, -0.1f, 0.1f);
  std::vector<float> host_topkVals(ROUTING_SIZE);
  std::vector<int> host_topkIdx(ROUTING_SIZE);

  float* tokens_device = nullptr;
  float* gateWeights_device = nullptr;
  float* d_topkVals = nullptr;
  int* d_topkIdx = nullptr;

  HIP_ERROR_ASSERT(hipMalloc(&tokens_device, TOKENS_SIZE * sizeof(float)));
  HIP_ERROR_ASSERT(
      hipMalloc(&gateWeights_device, GATE_WEIGHTS_SIZE * sizeof(float)));
  HIP_ERROR_ASSERT(hipMalloc(&d_topkVals, ROUTING_SIZE * sizeof(float)));
  HIP_ERROR_ASSERT(hipMalloc(&d_topkIdx, ROUTING_SIZE * sizeof(int)));

  HIP_ERROR_ASSERT(hipMemcpy(tokens_device, tokens_host.data(),
                             TOKENS_SIZE * sizeof(float),
                             hipMemcpyHostToDevice));
  HIP_ERROR_ASSERT(hipMemcpy(gateWeights_device, gateWeights_host.data(),
                             GATE_WEIGHTS_SIZE * sizeof(float),
                             hipMemcpyHostToDevice));
  HIP_ERROR_ASSERT(refHipRouting(tokens_device, gateWeights_device, d_topkVals,
                                 d_topkIdx, TOKENS_NUM, EXPERTS_NUM,
                                 HIDDEN_SIZE));

  HIP_ERROR_ASSERT(hipDeviceSynchronize())
  HIP_ERROR_ASSERT(hipGetLastError());

  HIP_ERROR_ASSERT(hipMemcpy(host_topkVals.data(), d_topkVals,
                             ROUTING_SIZE * sizeof(float),
                             hipMemcpyDeviceToHost));
  HIP_ERROR_ASSERT(hipMemcpy(host_topkIdx.data(), d_topkIdx,
                             ROUTING_SIZE * sizeof(int),
                             hipMemcpyDeviceToHost));

  // Free device memory.
  HIP_ERROR_ASSERT(hipFree(tokens_device));
  HIP_ERROR_ASSERT(hipFree(gateWeights_device));
  HIP_ERROR_ASSERT(hipFree(d_topkVals));
  HIP_ERROR_ASSERT(hipFree(d_topkIdx));

  auto refOut = test::refMoeGate(tokens_host.data(), gateWeights_host.data(),
                                 TOKENS_NUM, EXPERTS_NUM, HIDDEN_SIZE, TOPK);

  CheckAgainstRefBuffer(host_topkVals.data(), refOut.topkVals.data(),
                        refOut.topkVals.size());
  CheckAgainstRefBuffer(host_topkIdx.data(), refOut.topkIdx.data(),
                        refOut.topkIdx.size());
}