#include <hip/hip_runtime.h>

#include "../../../src/hip/hipHostCommon.h"
#include "../../../src/hip/taskSystem/taskManager.h"
#include "../../../src/hip/taskSystem/workerLoop.h"
#include "../../../src/hip/tasks/internal/expertFFN1.h"
#include "../../../src/hip/tasks/internal/gemmTileBlock.h"
#include "../../../src/hip/tasks/internal/reduceBlock.h"
#include "../../../src/hip/tasks/internal/softmaxBlock.h"
#include "../../../src/hip/tasks/internal/topkBlock.h"
#include "../../../src/hip/utils/hipDeviceUtils.h"
#include "../reference/refFullMoe.h"
#include "../utils/utils.h"

namespace {
constexpr int TOKENS_NUM = 48;
constexpr int EXPERTS_NUM = 128;
constexpr int HIDDEN_SIZE = 2048;
constexpr int EXPERT_INTERMEDIATE_SIZE = 768;
constexpr int TOPK = 8;
constexpr int THREADS = 128;
constexpr int BLOCKS = 304;
constexpr int REDUCTION_TILE_SIZE = 512;
constexpr int REDUCTION_CHUNKS_PER_TOKEN = (HIDDEN_SIZE / REDUCTION_TILE_SIZE);

// Gate => tokens[TOKENS_NUM,HIDDEN_SIZE] * gate[HIDDEN_SIZE,EXPERTS_NUM] =>
// routing[TOKENS_NUM,EXPERTS_NUM]
using TGateGemmTileParams =
    GemmTileParams<EXPERTS_NUM, HIDDEN_SIZE, 1, EXPERTS_NUM, 32, THREADS>;

constexpr int EXPERT_TILE_SIZE = 256;
// Expert => token[1,HIDDEN_SIZE] *
// expertW[HIDDEN_SIZE,EXPERT_INTERMEDIATE_SIZE] => [1,EXPERT_INTERMEDIATE_SIZE]
using TExpertFFN1TileParams =
    GemmTileParams<EXPERT_INTERMEDIATE_SIZE, HIDDEN_SIZE, 1, EXPERT_TILE_SIZE,
                   32, THREADS>;

// Expert => token[1,EXPERT_INTERMEDIATE_SIZE] *
// expertW[EXPERT_INTERMEDIATE_SIZE,HIDDEN_SIZE] => [1,HIDDEN_SIZE]
using TExpertFFN2TileParams =
    GemmTileParams<HIDDEN_SIZE, EXPERT_INTERMEDIATE_SIZE, 1, EXPERT_TILE_SIZE,
                   32, THREADS>;

struct __align__(16) ExpertTileTask {
  const float* __restrict__ ffn1ExpertWeights;
  const float* __restrict__ ffn2ExpertWeights;
  const float* __restrict__ tokens;
  float* __restrict__ ffn1Output;
  float* __restrict__ ffn2Output;
  float* __restrict__ finalOutput;
  float expertWeight;
  int tokenIdx;
  int expertIdx;
  int blockTileColStartIdx;
  int topkSlotIdx;
  int taskType;
};

using TaskManagerType = TaskManager<ExpertTileTask, 8192>;
static __constant__ TaskManagerType globalTaskManager;
static __constant__ int* globalFFN1SyncArray;
static __constant__ int* globalChunkReduceSyncArray;

template <typename T_FFN1_TILE>
__device__ void ExecuteExpertFFN1Task(const ExpertTileTask& task,
                                      void* sharedMemPool) {
  if (threadIdx.x == 65) {
    HIP_DEVICE_LOG(
        "Worker %i: Executes tile no: %i, tokenIdx: %i, expert: %i, weight: "
        "%f, col: %i\n",
        blockIdx.x, task.blockTileColStartIdx, task.tokenIdx, task.expertIdx,
        task.expertWeight, task.blockTileColStartIdx);
  }

  const float* thisBlockTokens = task.tokens + task.tokenIdx * T_FFN1_TILE::K;
  const float* thisffn1ExpertWeights =
      task.ffn1ExpertWeights + task.expertIdx * T_FFN1_TILE::K * T_FFN1_TILE::N;

  typename T_FFN1_TILE::TOutputType outRegs[T_FFN1_TILE::THREAD_OUTPUT_SIZE];

  tasks::expertFFN1Task<T_FFN1_TILE>(thisBlockTokens, thisffn1ExpertWeights,
                                     outRegs, 0, task.blockTileColStartIdx,
                                     sharedMemPool);

  float* thisBlockOutput = task.ffn1Output +
                           task.tokenIdx * TOPK * T_FFN1_TILE::N +
                           task.topkSlotIdx * T_FFN1_TILE::N;

  WriteGemmTileToGlobalMem_block<T_FFN1_TILE>(outRegs, thisBlockOutput, 0,
                                              task.blockTileColStartIdx);

  __threadfence();
  // TODO: Is mem barrier needed?
  constexpr int N_CHUNKS = T_FFN1_TILE::N / T_FFN1_TILE::TILE_N;
  bool* lastBlock = reinterpret_cast<bool*>(sharedMemPool);
  if (threadIdx.x == 0) {
    int processed = atomicAdd(
        &globalFFN1SyncArray[task.tokenIdx * TOPK + task.topkSlotIdx], 1);
    *lastBlock = ((processed + 1) == N_CHUNKS);
  }

  __syncthreads();
  if (*lastBlock) {
    // Add tasks for ffn2:
    ExpertTileTask ffn2Task = task;
    ffn2Task.taskType = 1;

    constexpr int FFN2_CHUNKS = T_FFN1_TILE::K / T_FFN1_TILE::TILE_N;
    for (int colIdx = 0; colIdx < FFN2_CHUNKS; ++colIdx) {
      ffn2Task.blockTileColStartIdx = colIdx;
      globalTaskManager.PushTask_warp(&ffn2Task);
    }
  }
}

template <typename T_FFN2_TILE>
__device__ void ExecuteExpertFFN2Task(const ExpertTileTask& task,
                                      void* sharedMemPool) {
  if (threadIdx.x == 65) {
    HIP_DEVICE_LOG(
        "Worker %i: Executes tile no: %i, tokenIdx: %i, expert: %i, weight: "
        "%f\n",
        blockIdx.x, task.blockTileColStartIdx, task.tokenIdx, task.expertIdx,
        task.expertWeight);
  }

  const float* thisBlockInput = task.ffn1Output +
                                task.tokenIdx * TOPK * T_FFN2_TILE::K +
                                task.topkSlotIdx * T_FFN2_TILE::K;
  const float* thisffn2ExpertWeights =
      task.ffn2ExpertWeights + task.expertIdx * T_FFN2_TILE::K * T_FFN2_TILE::N;

  typename T_FFN2_TILE::TOutputType outRegs[T_FFN2_TILE::THREAD_OUTPUT_SIZE];

  GemmTile_block<T_FFN2_TILE>(thisBlockInput, thisffn2ExpertWeights, outRegs, 0,
                              task.blockTileColStartIdx, sharedMemPool);

  float* thisBlockOutput = task.ffn2Output +
                           task.tokenIdx * TOPK * T_FFN2_TILE::N +
                           task.topkSlotIdx * T_FFN2_TILE::N;

#pragma unroll
  for (int i = 0; i < T_FFN2_TILE::THREAD_OUTPUT_SIZE; ++i) {
    outRegs[i] *= task.expertWeight;
  }

  WriteGemmTileToGlobalMem_block<T_FFN2_TILE>(outRegs, thisBlockOutput, 0,
                                              task.blockTileColStartIdx);

  __threadfence();
  // TODO: Is mem barrier needed?
  constexpr int FFN2_DEP_SIZE = REDUCTION_TILE_SIZE / T_FFN2_TILE::TILE_N;
  constexpr int CHUNKS_NEED = FFN2_DEP_SIZE * TOPK;
  bool* lastBlock = reinterpret_cast<bool*>(sharedMemPool);
  if (threadIdx.x == 0) {
    int processed = atomicAdd(
        &globalChunkReduceSyncArray[task.tokenIdx * REDUCTION_CHUNKS_PER_TOKEN +
                                    task.blockTileColStartIdx / FFN2_DEP_SIZE],
        1);
    *lastBlock = ((processed + 1) == CHUNKS_NEED);
  }

  __syncthreads();
  if (*lastBlock) {
    // Add tasks for reduction:
    ExpertTileTask reductionTask = task;
    reductionTask.taskType = 2;
    reductionTask.blockTileColStartIdx =
        task.blockTileColStartIdx / FFN2_DEP_SIZE;
    globalTaskManager.PushTask_warp(&reductionTask);
  }
}

__device__ void ExecuteFinalReductionTask(const ExpertTileTask& task,
                                          void* sharedMemPool) {
  const float* thisBlockInput =
      task.ffn2Output + task.tokenIdx * TOPK * HIDDEN_SIZE;

  float* thisBlockOutput = task.finalOutput + task.tokenIdx * HIDDEN_SIZE;

  tasks::Reduce_block<TOPK, REDUCTION_TILE_SIZE, THREADS, HIDDEN_SIZE>(
      thisBlockInput, thisBlockOutput, task.blockTileColStartIdx);
}

template <typename T_FFN1_TILE, typename T_FFN2_TILE>
__device__ void ExecuteGeneralTask(const ExpertTileTask& task,
                                   void* sharedMemPool) {
  switch (task.taskType) {
    case 0: {
      ExecuteExpertFFN1Task<T_FFN1_TILE>(task, sharedMemPool);
      break;
    }
    case 1: {
      ExecuteExpertFFN2Task<T_FFN2_TILE>(task, sharedMemPool);
      break;
    }
    case 2: {
      ExecuteFinalReductionTask(task, sharedMemPool);
      break;
    }
    default: {
      HIP_DEVICE_ASSERT(false);
    }
  }
}

template <typename T>
constexpr __device__ T max(T a, T b) {
  return a > b ? a : b;
}

template <typename T_GATE_TILE, typename T_FFN1_TILE, typename T_FFN2_TILE>
__global__ void refHipFullMoeKernel(const float* __restrict__ tokens,
                                    const float* __restrict__ gateWeights,
                                    const float* __restrict__ ffn1ExpertWeights,
                                    const float* __restrict__ ffn2ExpertWeights,
                                    float* __restrict__ ffn1Output,
                                    float* __restrict__ ffn2Output,
                                    float* __restrict__ finalOutput,
                                    int TOKENS_NUM) {
  constexpr int SHARED_MEM_SIZE_BYTES =
      max(T_GATE_TILE::SHARED_MEM_NEEDES_BYTES,
          max(T_FFN1_TILE::SHARED_MEM_NEEDES_BYTES,
              T_FFN2_TILE::SHARED_MEM_NEEDES_BYTES));

  __shared__ char sharedMemPool[SHARED_MEM_SIZE_BYTES];

  for (int tokenIdx = blockIdx.x; tokenIdx < TOKENS_NUM;
       tokenIdx += gridDim.x) {
    typename T_GATE_TILE::TOutputType out_regs[T_GATE_TILE::THREAD_OUTPUT_SIZE];

    GemmTile_block<T_GATE_TILE>(tokens, gateWeights, out_regs, tokenIdx, 0,
                                sharedMemPool);
    out_regs[0] =
        Softmax_block<T_GATE_TILE::THREADS, float>(out_regs[0], sharedMemPool);

    char* sharedMemPoolBytes = reinterpret_cast<char*>(sharedMemPool);
    float* s_topkVals = reinterpret_cast<float*>(sharedMemPoolBytes);
    int* s_topkIdx =
        reinterpret_cast<int*>(sharedMemPoolBytes + sizeof(float) * TOPK);
    int* nextPool = s_topkIdx + TOPK;

    Topk8_block<T_GATE_TILE::THREADS, float>(out_regs[0], s_topkVals, s_topkIdx,
                                             nextPool);

    // Warp 0 pushes all the tasks...
    for (int i = 0; i < TOPK; ++i) {
      ExpertTileTask rTask;
      rTask.ffn1ExpertWeights = ffn1ExpertWeights;
      rTask.ffn2ExpertWeights = ffn2ExpertWeights;
      rTask.tokens = tokens;
      rTask.ffn1Output = ffn1Output;
      rTask.ffn2Output = ffn2Output;
      rTask.finalOutput = finalOutput;
      rTask.tokenIdx = tokenIdx;
      rTask.expertIdx = s_topkIdx[i];
      rTask.expertWeight = s_topkVals[i];
      rTask.blockTileColStartIdx = 0;
      rTask.topkSlotIdx = i;
      rTask.taskType = 0;

      globalTaskManager.PushTask_warp(&rTask);

      rTask.blockTileColStartIdx = 1;

      globalTaskManager.PushTask_warp(&rTask);

      rTask.blockTileColStartIdx = 2;

      globalTaskManager.PushTask_warp(&rTask);
    }
  }

  workerTaskSystemLoop_block(globalTaskManager,
                             ExecuteGeneralTask<T_FFN1_TILE, T_FFN2_TILE>,
                             sharedMemPool);
}

static hipError_t refHipFullMoe(const float* tokens, const float* gateWeights,
                                const float* ffn1ExpertWeights,
                                const float* ffn2ExpertWeights,
                                float* ffn1Output, float* ffn2Output,
                                float* finalOutput, int tokensNum,
                                int expertsNum, int hiddenSize) {
  assert(hiddenSize == HIDDEN_SIZE);
  assert(tokensNum == TOKENS_NUM);
  assert(expertsNum == EXPERTS_NUM);

  dim3 grid(BLOCKS);
  dim3 block(TGateGemmTileParams::THREADS);

  hipStream_t stream = 0;
  TaskManagerType managerHost;
  constexpr int ffn1TasksNum =
      TOKENS_NUM * TOPK * (EXPERT_INTERMEDIATE_SIZE / EXPERT_TILE_SIZE);
  constexpr int ffn2TasksNum =
      TOKENS_NUM * TOPK * (HIDDEN_SIZE / EXPERT_TILE_SIZE);
  constexpr int reductionTasksNum = TOKENS_NUM * REDUCTION_CHUNKS_PER_TOKEN;
  constexpr int expectedNumTasks =
      ffn1TasksNum + ffn2TasksNum + reductionTasksNum;
  printf("expectedNumTasks: %i\n", expectedNumTasks);
  HIP_ERROR_CHECK(managerHost.Init(stream, expectedNumTasks));

  int* ffn1SyncArray;
  HIP_ERROR_CHECK(
      hipMallocAsync(&ffn1SyncArray, sizeof(int) * TOKENS_NUM * TOPK, stream));
  HIP_ERROR_CHECK(hipMemsetAsync(ffn1SyncArray, 0,
                                 sizeof(int) * TOKENS_NUM * TOPK, stream));
  HIP_ERROR_CHECK(hipMemcpyToSymbolAsync(HIP_SYMBOL(globalFFN1SyncArray),
                                         &ffn1SyncArray, sizeof(int*), 0,
                                         hipMemcpyHostToDevice, stream));

  int* chunkReduceSyncArray;
  HIP_ERROR_CHECK(hipMallocAsync(
      &chunkReduceSyncArray,
      sizeof(int) * TOKENS_NUM * REDUCTION_CHUNKS_PER_TOKEN, stream));
  HIP_ERROR_CHECK(hipMemsetAsync(
      chunkReduceSyncArray, 0,
      sizeof(int) * TOKENS_NUM * REDUCTION_CHUNKS_PER_TOKEN, stream));
  HIP_ERROR_CHECK(hipMemcpyToSymbolAsync(HIP_SYMBOL(globalChunkReduceSyncArray),
                                         &chunkReduceSyncArray, sizeof(int*), 0,
                                         hipMemcpyHostToDevice, stream));

  HIP_ERROR_CHECK(hipMemcpyToSymbolAsync(HIP_SYMBOL(globalTaskManager),
                                         &managerHost, sizeof(TaskManagerType),
                                         0, hipMemcpyHostToDevice, stream));

  refHipFullMoeKernel<TGateGemmTileParams, TExpertFFN1TileParams,
                      TExpertFFN2TileParams><<<grid, block, 0, 0>>>(
      tokens, gateWeights, ffn1ExpertWeights, ffn2ExpertWeights, ffn1Output,
      ffn2Output, finalOutput, tokensNum);

  HIP_ERROR_CHECK(managerHost.Deinit(stream));
  HIP_ERROR_CHECK(hipFreeAsync(ffn1SyncArray, stream));
  HIP_ERROR_CHECK(hipFreeAsync(chunkReduceSyncArray, stream));

  return hipSuccess;
}

static void printResult(const float* result) {
  for (int tokenIdx = 0; tokenIdx < TOKENS_NUM; ++tokenIdx) {
    std::cout << "TokenIdx: " << tokenIdx << std::endl;
    for (int j = 0; j < 10; ++j) {
      std::cout << "|" << result[tokenIdx * HIDDEN_SIZE + j];
    }

    std::cout << std::endl;
  }
}

TEST(MoeFull, basicTest) {
  constexpr int TOKENS_SIZE = TOKENS_NUM * HIDDEN_SIZE;
  constexpr int GATE_WEIGHTS_SIZE = HIDDEN_SIZE * EXPERTS_NUM;
  constexpr int ROUTING_SIZE = TOKENS_NUM * TOPK;
  constexpr int EXPERTS_FNN1_WEIGHTS_SIZE =
      EXPERTS_NUM * HIDDEN_SIZE * EXPERT_INTERMEDIATE_SIZE * 2;
  constexpr int FFN1_OUTPUT_SIZE = TOKENS_NUM * TOPK * EXPERT_INTERMEDIATE_SIZE;
  constexpr int EXPERTS_FNN2_WEIGHTS_SIZE =
      EXPERTS_NUM * HIDDEN_SIZE * EXPERT_INTERMEDIATE_SIZE;
  constexpr int FFN2_OUTPUT_SIZE = TOKENS_NUM * TOPK * HIDDEN_SIZE;
  constexpr int FINAL_OUTPUT_SIZE = TOKENS_NUM * HIDDEN_SIZE;

  std::vector<float> tokens_host =
      RandomVector<float>(TOKENS_SIZE, -1.0f, 1.0f);
  std::vector<float> gateWeights_host =
      RandomVector<float>(GATE_WEIGHTS_SIZE, -0.1f, 0.1f);
  std::vector<float> host_topkVals(ROUTING_SIZE);
  std::vector<int> host_topkIdx(ROUTING_SIZE);
  std::vector<float> expertFFN1Weights_host =
      RandomVector<float>(EXPERTS_FNN1_WEIGHTS_SIZE, -0.1f, 0.1f);
  std::vector<float> expertFFN2Weights_host =
      RandomVector<float>(EXPERTS_FNN2_WEIGHTS_SIZE, -0.1f, 0.1f);

  std::vector<float> output_host(FINAL_OUTPUT_SIZE, 0.0f);

  float* tokens_device = nullptr;
  float* gateWeights_device = nullptr;
  float* expertFFN1Weights_device = nullptr;
  float* expertFFN2Weights_device = nullptr;
  float* ffn1Output_device = nullptr;
  float* ffn2Output_device = nullptr;
  float* finalOutput_device = nullptr;

  HIP_ERROR_ASSERT(hipMalloc(&tokens_device, TOKENS_SIZE * sizeof(float)));
  HIP_ERROR_ASSERT(
      hipMalloc(&gateWeights_device, GATE_WEIGHTS_SIZE * sizeof(float)));
  HIP_ERROR_ASSERT(hipMalloc(&expertFFN1Weights_device,
                             EXPERTS_FNN1_WEIGHTS_SIZE * sizeof(float)));
  HIP_ERROR_ASSERT(hipMalloc(&expertFFN2Weights_device,
                             EXPERTS_FNN2_WEIGHTS_SIZE * sizeof(float)));
  HIP_ERROR_ASSERT(
      hipMalloc(&ffn1Output_device, FFN1_OUTPUT_SIZE * sizeof(float)));
  HIP_ERROR_ASSERT(
      hipMalloc(&ffn2Output_device, FFN2_OUTPUT_SIZE * sizeof(float)));
  HIP_ERROR_ASSERT(
      hipMalloc(&finalOutput_device, FINAL_OUTPUT_SIZE * sizeof(float)));

  HIP_ERROR_ASSERT(hipMemcpy(tokens_device, tokens_host.data(),
                             TOKENS_SIZE * sizeof(float),
                             hipMemcpyHostToDevice));
  HIP_ERROR_ASSERT(hipMemcpy(gateWeights_device, gateWeights_host.data(),
                             GATE_WEIGHTS_SIZE * sizeof(float),
                             hipMemcpyHostToDevice));
  HIP_ERROR_ASSERT(hipMemcpy(
      expertFFN1Weights_device, expertFFN1Weights_host.data(),
      EXPERTS_FNN1_WEIGHTS_SIZE * sizeof(float), hipMemcpyHostToDevice));
  HIP_ERROR_ASSERT(hipMemcpy(
      expertFFN2Weights_device, expertFFN2Weights_host.data(),
      EXPERTS_FNN2_WEIGHTS_SIZE * sizeof(float), hipMemcpyHostToDevice));

  HIP_ERROR_ASSERT(refHipFullMoe(
      tokens_device, gateWeights_device, expertFFN1Weights_device,
      expertFFN2Weights_device, ffn1Output_device, ffn2Output_device,
      finalOutput_device, TOKENS_NUM, EXPERTS_NUM, HIDDEN_SIZE));

  HIP_ERROR_ASSERT(hipDeviceSynchronize())
  HIP_ERROR_ASSERT(hipGetLastError());

  HIP_ERROR_ASSERT(hipMemcpy(output_host.data(), finalOutput_device,
                             FINAL_OUTPUT_SIZE * sizeof(float),
                             hipMemcpyDeviceToHost));

  // Free device memory.
  HIP_ERROR_ASSERT(hipFree(tokens_device));
  HIP_ERROR_ASSERT(hipFree(gateWeights_device));
  HIP_ERROR_ASSERT(hipFree(expertFFN1Weights_device));
  HIP_ERROR_ASSERT(hipFree(expertFFN2Weights_device));
  HIP_ERROR_ASSERT(hipFree(ffn1Output_device));
  HIP_ERROR_ASSERT(hipFree(ffn2Output_device));
  HIP_ERROR_ASSERT(hipFree(finalOutput_device));

  auto refOut = test::refFullMoe(
      tokens_host.data(), gateWeights_host.data(),
      expertFFN1Weights_host.data(), expertFFN2Weights_host.data(), TOKENS_NUM,
      EXPERTS_NUM, HIDDEN_SIZE, TOPK, EXPERT_INTERMEDIATE_SIZE);

  // printResult(refOut.data());

  ASSERT_EQ(output_host.size(), refOut.size());
  CheckAgainstRefBuffer(output_host.data(), refOut.data(), refOut.size(),
                        1e-5f);
}
}  // namespace