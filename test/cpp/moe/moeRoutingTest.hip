#include <hip/hip_runtime.h>

#include "../../../src/hip/hipHostCommon.h"
#include "../../../src/hip/taskSystem/taskManager.h"
#include "../../../src/hip/taskSystem/workerLoop.h"
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
namespace test {
struct __align__(16) ExpertTileTask {
  const float* expertWeights;
  const float* tokens;
  float* output;
  float expertWeight;
  int tokenIdx;
  int expertIdx;
  int blockTileColStartIdx;
  int topkSlotIdx;
};

using TaskManagerType = TaskManager<ExpertTileTask, 8192>;
__constant__ TaskManagerType globalTaskManager;

template <typename T_EXPERT_TILE>
__device__ void ExecuteTask(const ExpertTileTask& task, void* sharedMemPool) {
  if (threadIdx.x == 65) {
    HIP_DEVICE_LOG(
        "Worker %i: Executes tile no: %i, tokenIdx: %i, expert: %i, weight: "
        "%f\n",
        blockIdx.x, task.blockTileColStartIdx, task.tokenIdx, task.expertIdx,
        task.expertWeight);
  }

  const float* thisBlockTokens = task.tokens + task.tokenIdx * T_EXPERT_TILE::K;
  const float* thisExpertWeights =
      task.expertWeights + task.expertIdx * T_EXPERT_TILE::K * T_EXPERT_TILE::N;

  typename T_EXPERT_TILE::TOutputType
      out_regs[T_EXPERT_TILE::THREAD_OUTPUT_SIZE];
  GemmTile_block<T_EXPERT_TILE>(thisBlockTokens, thisExpertWeights, out_regs, 0,
                                task.blockTileColStartIdx, sharedMemPool);

  float* thisBlockOutput = task.output +
                           task.tokenIdx * TOPK * (T_EXPERT_TILE::N + 2) +
                           task.topkSlotIdx * (T_EXPERT_TILE::N + 2);

  if (task.blockTileColStartIdx == 0) {
    if (threadIdx.x == 0) {
      thisBlockOutput[0] = (float)task.expertIdx;
      thisBlockOutput[1] = (float)task.expertWeight;
    }
  }

  thisBlockOutput += 2;

  WriteGemmTileToGlobalMem_block<T_EXPERT_TILE>(out_regs, thisBlockOutput, 0,
                                                task.blockTileColStartIdx);
}

template <typename T_GATE_TILE, typename T_EXPERT_TILE>
__global__ void refRoutingKernel(const float* __restrict__ tokens,
                                 const float* __restrict__ gateWeights,
                                 const float* __restrict__ expertWeights,
                                 float* __restrict__ output, int TOKENS_NUM) {
  constexpr int SHARED_MEM_SIZE_BYTES =
      T_GATE_TILE::SHARED_MEM_NEEDES_BYTES >
              T_EXPERT_TILE::SHARED_MEM_NEEDES_BYTES
          ? T_GATE_TILE::SHARED_MEM_NEEDES_BYTES
          : T_EXPERT_TILE::SHARED_MEM_NEEDES_BYTES;
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
      rTask.expertWeights = expertWeights;
      rTask.tokens = tokens;
      rTask.output = output;
      rTask.tokenIdx = tokenIdx;
      rTask.expertIdx = s_topkIdx[i];
      rTask.expertWeight = s_topkVals[i];
      rTask.blockTileColStartIdx = 0;
      rTask.topkSlotIdx = i;

      globalTaskManager.PushTask_warp(&rTask);

      rTask.blockTileColStartIdx = 1;

      globalTaskManager.PushTask_warp(&rTask);

      rTask.blockTileColStartIdx = 2;

      globalTaskManager.PushTask_warp(&rTask);
    }
  }

  workerTaskSystemLoop_block(globalTaskManager, ExecuteTask<T_EXPERT_TILE>,
                             sharedMemPool);
}

static hipError_t refHipRouting(const float* tokens, const float* gateWeights,
                                const float* expertWeights, float* output,
                                int tokensNum, int expertsNum, int hiddenSize) {
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

  refRoutingKernel<TGateGemmTileParams, TExpertGemmTileParams>
      <<<grid, block, 0, 0>>>(tokens, gateWeights, expertWeights, output,
                              tokensNum);

  HIP_ERROR_CHECK(managerHost.Deinit(stream));

  return hipSuccess;
}

// GPT5-mini generated function with minor changes.
// Replace simple stub with CPU reference that runs gate (refMoeGate) and then
// projects each selected (token, expert) pair through the expert weights.
// Output layout:
//   rows = tokensNum * topk
//   cols = expertIntermediateSize + 2
// For each output row r corresponding to (token t, topk-pos k):
//   out[r,0] = float(expertId)          (or -1.0f if invalid)
//   out[r,1] = gateScore                (weight)
//   out[r, 2 .. 2+expertIntermediateSize-1] = tokenProjection (1 x
//   expertIntermediateSize)
// No final reduction is performed: each (token,expert) produces its own row.
static std::vector<float> refRouting(const float* tokens,
                                     const float* gateWeights,
                                     const float* expertWeights, int tokensNum,
                                     int expertsNum, int hiddenSize, int topk,
                                     int expertIntermediateSize) {
  assert(tokens != nullptr);
  assert(gateWeights != nullptr);
  // expertWeights may be nullptr -> then projection fields will be zeros.

  // Run gating to obtain topk info
  auto gateOut = test::refMoeGate(tokens, gateWeights, tokensNum, expertsNum,
                                  hiddenSize, topk);
  // gateOut.topkVals and gateOut.topkIdx are expected to be vectors of size
  // tokensNum*topk
  const std::vector<float>& topkVals = gateOut.topkVals;
  const std::vector<int>& topkIdx = gateOut.topkIdx;
  const int rows = tokensNum * topk;
  const int cols = expertIntermediateSize + 2;

  std::vector<float> out;
  out.assign(size_t(rows) * size_t(cols), 0.0f);

  for (int t = 0; t < tokensNum; ++t) {
    const float* tokenPtr = tokens + size_t(t) * size_t(hiddenSize);

    for (int k = 0; k < topk; ++k) {
      const int row = t * topk + k;
      const int expert = topkIdx[row];
      const float gateScore = topkVals[row];
      const size_t rowBase = size_t(row) * size_t(cols);

      // expert id as float in column 0
      out[rowBase + 0] = (expert >= 0) ? static_cast<float>(expert) : -1.0f;
      // gate weight in column 1
      out[rowBase + 1] = gateScore;

      // projection columns start at offset 2
      if (expert < 0 || expert >= expertsNum || expertWeights == nullptr) {
        // leave projection zeros
        continue;
      }

      // expertWeights layout assumed: [expert][hidden][expertIntermediate]
      const float* expertW = expertWeights + size_t(expert) *
                                                 size_t(hiddenSize) *
                                                 size_t(expertIntermediateSize);

      // Use reference GEMM for 1 x hidden * hidden x expertIntermediate => 1 x
      // expertIntermediate refGemm(A,B,M,N,K) where A is 1 x K, B is K x N ->
      // returns vector length M*N (here N=expertIntermediateSize)
      std::vector<float> proj =
          test::refGemm<float>(tokenPtr, expertW, /*M=*/1,
                               /*N=*/expertIntermediateSize, /*K=*/hiddenSize);
      // Copy projection into output row
      for (int d = 0; d < expertIntermediateSize; ++d) {
        out[rowBase + 2 + d] =
            proj[size_t(d)];  // proj is 1 x N stored row-major
      }
    }
  }

  return out;
}

// GPU generates unsorted data within given topk bucket, sort it against expert
// weight. GPT5-mini generated code with my minor changes.
static void sortResultData(float* data) {
  // data layout: TOKENS_NUM x TOPK x (EXPERT_INTERMEDIATE_SIZE + 2)
  const int tokens = TOKENS_NUM;
  const int rows_per_token = TOPK;
  const int cols = EXPERT_INTERMEDIATE_SIZE + 2;

  // temporary buffer for swapping rows
  std::vector<float> tmp(cols);

  for (int t = 0; t < tokens; ++t) {
    float* token_base = data + size_t(t) * rows_per_token * cols;

    // Insertion sort descending by weight (weight is at column offset 1)
    for (int i = 1; i < rows_per_token; ++i) {
      int j = i;
      while (j > 0) {
        float w_prev = token_base[(j - 1) * cols + 1];
        float w_cur = token_base[j * cols + 1];
        if (w_prev >= w_cur) break;  // already in order (descending)
        // swap rows (j-1) and j
        float* rprev = token_base + (j - 1) * cols;
        float* rcur = token_base + j * cols;
        std::memcpy(tmp.data(), rprev, cols * sizeof(float));
        std::memcpy(rprev, rcur, cols * sizeof(float));
        std::memcpy(rcur, tmp.data(), cols * sizeof(float));
        --j;
      }
    }
  }
}

static void printResult(const float* result) {
  for (int tokenIdx = 0; tokenIdx < TOKENS_NUM; ++tokenIdx) {
    std::cout << "TokenIdx: " << tokenIdx << std::endl;
    for (int i = 0; i < TOPK; ++i) {
      const float expertId =
          result[tokenIdx * TOPK * (EXPERT_INTERMEDIATE_SIZE + 2) +
                 i * (EXPERT_INTERMEDIATE_SIZE + 2)];
      const float weight =
          result[tokenIdx * TOPK * (EXPERT_INTERMEDIATE_SIZE + 2) +
                 i * (EXPERT_INTERMEDIATE_SIZE + 2) + 1];
      std::cout << "Expert id: " << expertId << ", weight: " << weight;

      for (int i = 2; i < 10; ++i) {
        std::cout << "|"
                  << result[tokenIdx * TOPK * (EXPERT_INTERMEDIATE_SIZE + 2) +
                            i * (EXPERT_INTERMEDIATE_SIZE + 2) + i];
      }

      std::cout << std::endl;
    }
  }
}
}  // namespace test

TEST(MoeRouting, basicTest) {
  constexpr int TOKENS_SIZE = TOKENS_NUM * HIDDEN_SIZE;
  constexpr int GATE_WEIGHTS_SIZE = HIDDEN_SIZE * EXPERTS_NUM;
  constexpr int ROUTING_SIZE = TOKENS_NUM * TOPK;
  constexpr int EXPERTS_WEIGHTS_SIZE =
      EXPERTS_NUM * HIDDEN_SIZE * EXPERT_INTERMEDIATE_SIZE;
  constexpr int OUTPUT_SIZE =
      TOKENS_NUM * TOPK * (EXPERT_INTERMEDIATE_SIZE + 2);

  std::vector<float> tokens_host =
      RandomVector<float>(TOKENS_SIZE, -0.1f, 0.1f);
  std::vector<float> gateWeights_host =
      RandomVector<float>(GATE_WEIGHTS_SIZE, -0.1f, 0.1f);
  std::vector<float> host_topkVals(ROUTING_SIZE);
  std::vector<int> host_topkIdx(ROUTING_SIZE);
  std::vector<float> expertWeights_host =
      RandomVector<float>(EXPERTS_WEIGHTS_SIZE, -0.1f, 0.1f);

  std::vector<float> output_host(OUTPUT_SIZE, 0.0f);

  float* tokens_device = nullptr;
  float* gateWeights_device = nullptr;
  float* expertWeights_device = nullptr;
  float* output_device = nullptr;

  HIP_ERROR_ASSERT(hipMalloc(&tokens_device, TOKENS_SIZE * sizeof(float)));
  HIP_ERROR_ASSERT(
      hipMalloc(&gateWeights_device, GATE_WEIGHTS_SIZE * sizeof(float)));
  HIP_ERROR_ASSERT(
      hipMalloc(&expertWeights_device, EXPERTS_WEIGHTS_SIZE * sizeof(float)));
  HIP_ERROR_ASSERT(hipMalloc(&output_device, OUTPUT_SIZE * sizeof(float)));

  HIP_ERROR_ASSERT(hipMemcpy(tokens_device, tokens_host.data(),
                             TOKENS_SIZE * sizeof(float),
                             hipMemcpyHostToDevice));
  HIP_ERROR_ASSERT(hipMemcpy(gateWeights_device, gateWeights_host.data(),
                             GATE_WEIGHTS_SIZE * sizeof(float),
                             hipMemcpyHostToDevice));
  HIP_ERROR_ASSERT(hipMemcpy(expertWeights_device, expertWeights_host.data(),
                             EXPERTS_WEIGHTS_SIZE * sizeof(float),
                             hipMemcpyHostToDevice));
  HIP_ERROR_ASSERT(test::refHipRouting(tokens_device, gateWeights_device,
                                       expertWeights_device, output_device,
                                       TOKENS_NUM, EXPERTS_NUM, HIDDEN_SIZE));

  HIP_ERROR_ASSERT(hipDeviceSynchronize())
  HIP_ERROR_ASSERT(hipGetLastError());

  HIP_ERROR_ASSERT(hipMemcpy(output_host.data(), output_device,
                             OUTPUT_SIZE * sizeof(float),
                             hipMemcpyDeviceToHost));

  // Free device memory.
  HIP_ERROR_ASSERT(hipFree(tokens_device));
  HIP_ERROR_ASSERT(hipFree(gateWeights_device));
  HIP_ERROR_ASSERT(hipFree(expertWeights_device));
  HIP_ERROR_ASSERT(hipFree(output_device));

  auto refOut = test::refRouting(
      tokens_host.data(), gateWeights_host.data(), expertWeights_host.data(),
      TOKENS_NUM, EXPERTS_NUM, HIDDEN_SIZE, TOPK, EXPERT_INTERMEDIATE_SIZE);

  test::sortResultData(output_host.data());

  // test::printResult(output_host.data());

  CheckAgainstRefBuffer(output_host.data(), refOut.data(), refOut.size());
}