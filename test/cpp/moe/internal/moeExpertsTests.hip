#include <hip/hip_runtime.h>

#include "src/hip/hipHostCommon.h"
#include "src/hip/kernel/tasks/internal/expertFFN1Block.h"
#include "src/hip/kernel/tasks/internal/gemmTileBlock.h"
#include "src/hip/kernel/tasks/internal/softmaxBlock.h"
#include "src/hip/kernel/tasks/internal/topkBlock.h"
#include "src/hip/taskSystem/taskManager.h"
#include "src/hip/taskSystem/workerLoop.h"
#include "src/hip/utils/hipDeviceUtils.h"
#include "test/cpp/moe/reference/refExpertFFN1.h"
#include "test/cpp/moe/reference/refMoeGate.h"
#include "test/cpp/utils/utils.h"

namespace {
constexpr float ERROR_ABS = 1e-5f;
constexpr int TOKENS_NUM = 48;
constexpr int EXPERTS_NUM = 128;
constexpr int HIDDEN_SIZE = 2048;
constexpr int EXPERT_INTERMEDIATE_SIZE = 768;
constexpr int TOPK = 8;
constexpr int THREADS = 128;
constexpr int BLOCKS = 304;

// Gate => tokens[TOKENS_NUM,HIDDEN_SIZE] * gate[HIDDEN_SIZE,EXPERTS_NUM] =>
// routing[TOKENS_NUM,EXPERTS_NUM]
using TGateGemmTileParams = GemmTileParams<EXPERTS_NUM, HIDDEN_SIZE, 1,
                                           EXPERTS_NUM, 32, THREADS, float>;

constexpr int EXPERT_TILE_SIZE = 256;
// Expert => token[1,HIDDEN_SIZE] *
// expertW[HIDDEN_SIZE,EXPERT_INTERMEDIATE_SIZE] => [1,EXPERT_INTERMEDIATE_SIZE]
using TExpertFFN1TileParams =
    GemmTileParams<EXPERT_INTERMEDIATE_SIZE, HIDDEN_SIZE, 1, EXPERT_TILE_SIZE,
                   32, THREADS, float>;

// Expert => token[1,EXPERT_INTERMEDIATE_SIZE] *
// expertW[EXPERT_INTERMEDIATE_SIZE,HIDDEN_SIZE] => [1,HIDDEN_SIZE]
using TExpertFFN2TileParams =
    GemmTileParams<HIDDEN_SIZE, EXPERT_INTERMEDIATE_SIZE, 1, EXPERT_TILE_SIZE,
                   32, THREADS, float>;

struct __align__(16) ExpertTileTask {
  const float* ffn1ExpertWeights;
  const float* ffn2ExpertWeights;
  const float* tokens;
  float* ffn1Output;
  float* ffn2Output;
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

  typename T_FFN1_TILE::TType outRegs[T_FFN1_TILE::THREAD_OUTPUT_SIZE];

  moe::tasks::internal::expertFFN1_block<T_FFN1_TILE>(
      thisBlockTokens, thisffn1ExpertWeights, outRegs, 0,
      task.blockTileColStartIdx, sharedMemPool);

  float* thisBlockOutput = task.ffn1Output +
                           task.tokenIdx * TOPK * T_FFN1_TILE::N +
                           task.topkSlotIdx * T_FFN1_TILE::N;

  moe::tasks::internal::WriteGemmTileToGlobalMem_block<T_FFN1_TILE>(
      outRegs, thisBlockOutput, 0, task.blockTileColStartIdx);

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

  typename T_FFN2_TILE::TType outRegs[T_FFN2_TILE::THREAD_OUTPUT_SIZE];

  moe::tasks::internal::GemmTile_block<T_FFN2_TILE>(
      thisBlockInput, thisffn2ExpertWeights, outRegs, 0,
      task.blockTileColStartIdx, sharedMemPool);

  float* thisBlockOutput = task.ffn2Output +
                           task.tokenIdx * TOPK * (T_FFN2_TILE::N + 2) +
                           task.topkSlotIdx * (T_FFN2_TILE::N + 2);

  if (task.blockTileColStartIdx == 0) {
    if (threadIdx.x == 0) {
      thisBlockOutput[0] = (float)task.expertIdx;
      thisBlockOutput[1] = (float)task.expertWeight;
    }
  }

  thisBlockOutput += 2;

  moe::tasks::internal::WriteGemmTileToGlobalMem_block<T_FFN2_TILE>(
      outRegs, thisBlockOutput, 0, task.blockTileColStartIdx);
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
__global__ void refHipGatefRoutingExpertsKernel(
    const float* __restrict__ tokens, const float* __restrict__ gateWeights,
    const float* __restrict__ ffn1ExpertWeights,
    const float* __restrict__ ffn2ExpertWeights, float* __restrict__ ffn1Output,
    float* __restrict__ ffn2Output, int TOKENS_NUM) {
  constexpr int SHARED_MEM_SIZE_BYTES =
      max(T_GATE_TILE::SHARED_MEM_NEEDES_BYTES,
          max(T_FFN1_TILE::SHARED_MEM_NEEDES_BYTES,
              T_FFN2_TILE::SHARED_MEM_NEEDES_BYTES));

  __shared__ char sharedMemPool[SHARED_MEM_SIZE_BYTES];

  for (int tokenIdx = blockIdx.x; tokenIdx < TOKENS_NUM;
       tokenIdx += gridDim.x) {
    typename T_GATE_TILE::TType out_regs[T_GATE_TILE::THREAD_OUTPUT_SIZE];

    moe::tasks::internal::GemmTile_block<T_GATE_TILE>(
        tokens, gateWeights, out_regs, tokenIdx, 0, sharedMemPool);
    out_regs[0] =
        moe::tasks::internal::Softmax_block<T_GATE_TILE::THREADS, float>(
            out_regs[0], sharedMemPool);

    char* sharedMemPoolBytes = reinterpret_cast<char*>(sharedMemPool);
    float* s_topkVals = reinterpret_cast<float*>(sharedMemPoolBytes);
    int* s_topkIdx =
        reinterpret_cast<int*>(sharedMemPoolBytes + sizeof(float) * TOPK);
    int* nextPool = s_topkIdx + TOPK;

    moe::tasks::internal::Topk8_block<T_GATE_TILE::THREADS, float>(
        out_regs[0], s_topkVals, s_topkIdx, nextPool);

    // Warp 0 pushes all the tasks...
    for (int i = 0; i < TOPK; ++i) {
      ExpertTileTask rTask;
      rTask.ffn1ExpertWeights = ffn1ExpertWeights;
      rTask.ffn2ExpertWeights = ffn2ExpertWeights;
      rTask.tokens = tokens;
      rTask.ffn1Output = ffn1Output;
      rTask.ffn2Output = ffn2Output;
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

static hipError_t refHipGatefRoutingExperts(const float* tokens,
                                            const float* gateWeights,
                                            const float* ffn1ExpertWeights,
                                            const float* ffn2ExpertWeights,
                                            float* ffn1Output,
                                            float* ffn2Output, int tokensNum,
                                            int expertsNum, int hiddenSize) {
  assert(hiddenSize == HIDDEN_SIZE);
  assert(tokensNum == TOKENS_NUM);
  assert(expertsNum == EXPERTS_NUM);

  dim3 grid(BLOCKS);
  dim3 block(TGateGemmTileParams::THREADS);

  hipStream_t stream = 0;
  TaskManagerType managerHost;
  const int ffn1TasksNum =
      TOKENS_NUM * TOPK * (EXPERT_INTERMEDIATE_SIZE / EXPERT_TILE_SIZE);
  const int ffn2TasksNum = TOKENS_NUM * TOPK * (HIDDEN_SIZE / EXPERT_TILE_SIZE);
  const int expectedNumTasks = ffn1TasksNum + ffn2TasksNum;
  HIP_ERROR_CHECK(managerHost.Init(stream));
  HIP_ERROR_CHECK(managerHost.PrepareForNextLaunch(stream, expectedNumTasks));

  int* ffn1SyncArray;
  HIP_ERROR_CHECK(
      hipMallocAsync(&ffn1SyncArray, sizeof(int) * TOKENS_NUM * TOPK, stream));
  HIP_ERROR_CHECK(hipMemsetAsync(ffn1SyncArray, 0,
                                 sizeof(int) * TOKENS_NUM * TOPK, stream));

  HIP_ERROR_CHECK(hipMemcpyToSymbolAsync(HIP_SYMBOL(globalFFN1SyncArray),
                                         &ffn1SyncArray, sizeof(int*), 0,
                                         hipMemcpyHostToDevice, stream));

  HIP_ERROR_CHECK(hipMemcpyToSymbolAsync(HIP_SYMBOL(globalTaskManager),
                                         &managerHost, sizeof(TaskManagerType),
                                         0, hipMemcpyHostToDevice, stream));

  refHipGatefRoutingExpertsKernel<TGateGemmTileParams, TExpertFFN1TileParams,
                                  TExpertFFN2TileParams><<<grid, block, 0, 0>>>(
      tokens, gateWeights, ffn1ExpertWeights, ffn2ExpertWeights, ffn1Output,
      ffn2Output, tokensNum);

  HIP_ERROR_CHECK(managerHost.Deinit(stream));
  HIP_ERROR_CHECK(hipFreeAsync(ffn1SyncArray, stream));

  return hipSuccess;
}

// Ref CPU implementation.
static std::vector<float> refGatefRoutingExperts(const float* tokens,
                                                 const float* gateWeights,
                                                 const float* ffn1ExpertWeights,
                                                 const float* ffn2ExpertWeights,
                                                 int tokensNum, int expertsNum,
                                                 int hiddenSize, int topk,
                                                 int expertIntermediateSize) {
  assert(tokens != nullptr);
  assert(gateWeights != nullptr);
  // ffn1ExpertWeights may be nullptr -> then projection fields will be zeros.

  // Run gating to obtain topk info
  auto gateOut = test::refMoeGate(tokens, gateWeights, tokensNum, expertsNum,
                                  hiddenSize, topk);
  // gateOut.topkVals and gateOut.topkIdx are expected to be vectors of size
  // tokensNum*topk
  const std::vector<float>& topkVals = gateOut.topkVals;
  const std::vector<int>& topkIdx = gateOut.topkIdx;
  const int rows = tokensNum * topk;
  const int cols = hiddenSize + 2;

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
      if (expert < 0 || expert >= expertsNum || ffn1ExpertWeights == nullptr) {
        // leave projection zeros
        continue;
      }

      // ffn1ExpertWeights layout assumed: [expert][hidden][expertIntermediate]
      const float* expertFFN1W =
          ffn1ExpertWeights +
          size_t(expert) * size_t(hiddenSize) * size_t(expertIntermediateSize);

      auto refFFN1Out = test::refExpertFFN1(tokenPtr, expertFFN1W, 1,
                                            hiddenSize, expertIntermediateSize);

      // ffn1ExpertWeights layout assumed: [expert][expertIntermediate][hidden]
      const float* expertFFN2W =
          ffn2ExpertWeights +
          size_t(expert) * size_t(hiddenSize) * size_t(expertIntermediateSize);
      auto refFFN2Out = test::refGemm(refFFN1Out.data(), expertFFN2W, 1,
                                      hiddenSize, expertIntermediateSize);

      // Copy projection into output row
      for (int d = 0; d < hiddenSize; ++d) {
        out[rowBase + 2 + d] =
            refFFN2Out[size_t(d)];  // proj is 1 x N stored row-major
      }
    }
  }

  return out;
}

// GPU generates unsorted data within given topk bucket, sort it against expert
// weight. GPT5-mini generated code with my minor changes.
static void sortResultData(float* data) {
  // data layout: TOKENS_NUM x TOPK x (HIDDEN_SIZE + 2)
  const int tokens = TOKENS_NUM;
  const int rows_per_token = TOPK;
  const int cols = HIDDEN_SIZE + 2;

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
          result[tokenIdx * TOPK * (HIDDEN_SIZE + 2) + i * (HIDDEN_SIZE + 2)];
      const float weight = result[tokenIdx * TOPK * (HIDDEN_SIZE + 2) +
                                  i * (HIDDEN_SIZE + 2) + 1];
      std::cout << "Expert id: " << expertId << ", weight: " << weight;

      for (int j = 2; j < 10; ++j) {
        std::cout << "|"
                  << result[tokenIdx * TOPK * (HIDDEN_SIZE + 2) +
                            i * (HIDDEN_SIZE + 2) + j];
      }

      std::cout << std::endl;
    }
  }
}

TEST(MoeTasksInternalTests, MoeExperts_basicTest) {
  constexpr int TOKENS_SIZE = TOKENS_NUM * HIDDEN_SIZE;
  constexpr int GATE_WEIGHTS_SIZE = HIDDEN_SIZE * EXPERTS_NUM;
  constexpr int ROUTING_SIZE = TOKENS_NUM * TOPK;
  constexpr int EXPERTS_FNN1_WEIGHTS_SIZE =
      EXPERTS_NUM * HIDDEN_SIZE * EXPERT_INTERMEDIATE_SIZE * 2;
  constexpr int FFN1_OUTPUT_SIZE = TOKENS_NUM * TOPK * EXPERT_INTERMEDIATE_SIZE;
  constexpr int EXPERTS_FNN2_WEIGHTS_SIZE =
      EXPERTS_NUM * HIDDEN_SIZE * EXPERT_INTERMEDIATE_SIZE;
  constexpr int FFN2_OUTPUT_SIZE = TOKENS_NUM * TOPK * (HIDDEN_SIZE + 2);

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

  std::vector<float> output_host(FFN2_OUTPUT_SIZE, 0.0f);

  float* tokens_device = nullptr;
  float* gateWeights_device = nullptr;
  float* expertFFN1Weights_device = nullptr;
  float* expertFFN2Weights_device = nullptr;
  float* ffn1Output_device = nullptr;
  float* ffn2Output_device = nullptr;

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

  HIP_ERROR_ASSERT(refHipGatefRoutingExperts(
      tokens_device, gateWeights_device, expertFFN1Weights_device,
      expertFFN2Weights_device, ffn1Output_device, ffn2Output_device,
      TOKENS_NUM, EXPERTS_NUM, HIDDEN_SIZE));

  HIP_ERROR_ASSERT(hipDeviceSynchronize())
  HIP_ERROR_ASSERT(hipGetLastError());

  HIP_ERROR_ASSERT(hipMemcpy(output_host.data(), ffn2Output_device,
                             FFN2_OUTPUT_SIZE * sizeof(float),
                             hipMemcpyDeviceToHost));

  // Free device memory.
  HIP_ERROR_ASSERT(hipFree(tokens_device));
  HIP_ERROR_ASSERT(hipFree(gateWeights_device));
  HIP_ERROR_ASSERT(hipFree(expertFFN1Weights_device));
  HIP_ERROR_ASSERT(hipFree(expertFFN2Weights_device));
  HIP_ERROR_ASSERT(hipFree(ffn1Output_device));
  HIP_ERROR_ASSERT(hipFree(ffn2Output_device));

  auto refOut = refGatefRoutingExperts(
      tokens_host.data(), gateWeights_host.data(),
      expertFFN1Weights_host.data(), expertFFN2Weights_host.data(), TOKENS_NUM,
      EXPERTS_NUM, HIDDEN_SIZE, TOPK, EXPERT_INTERMEDIATE_SIZE);

  sortResultData(output_host.data());

  // printResult(output_host.data());

  CheckAgainstRefBuffer(output_host, refOut, ERROR_ABS);
}
}  // namespace