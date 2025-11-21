#include "moeKernelLauncher.h"
#include "src/hip/common/metadata.h"
#include "src/hip/hipHostCommon.h"
#include "src/hip/taskSystem/taskManager.h"
#include "src/hip/taskSystem/workerLoop.h"
#include "src/hip/tasks/internal/expertFFN1Block.h"
#include "src/hip/tasks/internal/gemmTileBlock.h"
#include "src/hip/tasks/internal/reduceBlock.h"
#include "src/hip/tasks/internal/softmaxBlock.h"
#include "src/hip/tasks/internal/topkBlock.h"
#include "src/hip/utils/hipDeviceUtils.h"

namespace moe {
struct __align__(16) MoeTaskDesc {
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

using TaskManagerType = TaskManager<MoeTaskDesc, 8192>;

static __constant__ TaskManagerType globalTaskManager;
static __constant__ int* globalFFN1SyncArray;
static __constant__ int* globalChunkReduceSyncArray;

template <typename MOE_METADATA>
__device__ void ExecuteExpertFFN1Task(const MoeTaskDesc& task,
                                      void* sharedMemPool) {
  using T_FFN1_TILE = typename MOE_METADATA::TILES_CONFIG::FFN1_TILE_METADATA;
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

  tasks::internal::expertFFN1_block<T_FFN1_TILE>(
      thisBlockTokens, thisffn1ExpertWeights, outRegs, 0,
      task.blockTileColStartIdx, sharedMemPool);

  float* thisBlockOutput =
      task.ffn1Output +
      task.tokenIdx * MOE_METADATA::MOE_PROBLEM_CONFIG::TOPK * T_FFN1_TILE::N +
      task.topkSlotIdx * T_FFN1_TILE::N;

  WriteGemmTileToGlobalMem_block<T_FFN1_TILE>(outRegs, thisBlockOutput, 0,
                                              task.blockTileColStartIdx);

  __threadfence();
  constexpr int N_CHUNKS = T_FFN1_TILE::N / T_FFN1_TILE::TILE_N;
  bool* lastBlock = reinterpret_cast<bool*>(sharedMemPool);
  if (threadIdx.x == 0) {
    int processed = atomicAdd(
        &globalFFN1SyncArray[task.tokenIdx *
                                 MOE_METADATA::MOE_PROBLEM_CONFIG::TOPK +
                             task.topkSlotIdx],
        1);
    *lastBlock = ((processed + 1) == N_CHUNKS);
  }

  __syncthreads();
  if (*lastBlock) {
    // Add tasks for ffn2:
    MoeTaskDesc ffn2Task = task;
    ffn2Task.taskType = 1;

    constexpr int FFN2_CHUNKS = T_FFN1_TILE::K / T_FFN1_TILE::TILE_N;
    for (int colIdx = 0; colIdx < FFN2_CHUNKS; ++colIdx) {
      ffn2Task.blockTileColStartIdx = colIdx;
      globalTaskManager.PushTask_warp(&ffn2Task);
    }
  }
}

template <typename MOE_METADATA>
__device__ void ExecuteExpertFFN2Task(const MoeTaskDesc& task,
                                      void* sharedMemPool) {
  using T_FFN2_TILE = typename MOE_METADATA::TILES_CONFIG::FFN2_TILE_METADATA;

  if (threadIdx.x == 65) {
    HIP_DEVICE_LOG(
        "Worker %i: Executes tile no: %i, tokenIdx: %i, expert: %i, weight: "
        "%f\n",
        blockIdx.x, task.blockTileColStartIdx, task.tokenIdx, task.expertIdx,
        task.expertWeight);
  }

  const float* thisBlockInput =
      task.ffn1Output +
      task.tokenIdx * MOE_METADATA::MOE_PROBLEM_CONFIG::TOPK * T_FFN2_TILE::K +
      task.topkSlotIdx * T_FFN2_TILE::K;
  const float* thisffn2ExpertWeights =
      task.ffn2ExpertWeights + task.expertIdx * T_FFN2_TILE::K * T_FFN2_TILE::N;

  typename T_FFN2_TILE::TOutputType outRegs[T_FFN2_TILE::THREAD_OUTPUT_SIZE];

  GemmTile_block<T_FFN2_TILE>(thisBlockInput, thisffn2ExpertWeights, outRegs, 0,
                              task.blockTileColStartIdx, sharedMemPool);

  float* thisBlockOutput =
      task.ffn2Output +
      task.tokenIdx * MOE_METADATA::MOE_PROBLEM_CONFIG::TOPK * T_FFN2_TILE::N +
      task.topkSlotIdx * T_FFN2_TILE::N;

#pragma unroll
  for (int i = 0; i < T_FFN2_TILE::THREAD_OUTPUT_SIZE; ++i) {
    outRegs[i] *= task.expertWeight;
  }

  WriteGemmTileToGlobalMem_block<T_FFN2_TILE>(outRegs, thisBlockOutput, 0,
                                              task.blockTileColStartIdx);

  __threadfence();
  constexpr int FFN2_DEP_SIZE =
      MOE_METADATA::TILES_CONFIG::REDUCTION_TILE_SIZE / T_FFN2_TILE::TILE_N;
  constexpr int CHUNKS_NEED =
      FFN2_DEP_SIZE * MOE_METADATA::MOE_PROBLEM_CONFIG::TOPK;
  bool* lastBlock = reinterpret_cast<bool*>(sharedMemPool);
  if (threadIdx.x == 0) {
    int processed = atomicAdd(
        &globalChunkReduceSyncArray
            [task.tokenIdx *
                 MOE_METADATA::TILES_CONFIG::REDUCTION_CHUNKS_PER_TOKEN +
             task.blockTileColStartIdx / FFN2_DEP_SIZE],
        1);
    *lastBlock = ((processed + 1) == CHUNKS_NEED);
  }

  __syncthreads();
  if (*lastBlock) {
    // Add tasks for reduction:
    MoeTaskDesc reductionTask = task;
    reductionTask.taskType = 2;
    reductionTask.blockTileColStartIdx =
        task.blockTileColStartIdx / FFN2_DEP_SIZE;
    globalTaskManager.PushTask_warp(&reductionTask);
  }
}

template <typename MOE_METADATA>
__device__ void ExecuteFinalReductionTask(const MoeTaskDesc& task,
                                          void* sharedMemPool) {
  const float* thisBlockInput =
      task.ffn2Output + task.tokenIdx * MOE_METADATA::MOE_PROBLEM_CONFIG::TOPK *
                            MOE_METADATA::MOE_PROBLEM_CONFIG::HIDDEN_SIZE;

  float* thisBlockOutput =
      task.finalOutput +
      task.tokenIdx * MOE_METADATA::MOE_PROBLEM_CONFIG::HIDDEN_SIZE;

  tasks::internal::Reduce_block<
      MOE_METADATA::MOE_PROBLEM_CONFIG::TOPK,
      MOE_METADATA::TILES_CONFIG::REDUCTION_TILE_SIZE,
      MOE_METADATA::TILES_CONFIG::GATE_TILE_METADATA::THREADS,
      MOE_METADATA::MOE_PROBLEM_CONFIG::HIDDEN_SIZE>(
      thisBlockInput, thisBlockOutput, task.blockTileColStartIdx);
}

template <typename MOE_METADATA>
__device__ void ExecuteGeneralTask(const MoeTaskDesc& task,
                                   void* sharedMemPool) {
  switch (task.taskType) {
    case 0: {
      ExecuteExpertFFN1Task<MOE_METADATA>(task, sharedMemPool);
      break;
    }
    case 1: {
      ExecuteExpertFFN2Task<MOE_METADATA>(task, sharedMemPool);
      break;
    }
    case 2: {
      ExecuteFinalReductionTask<MOE_METADATA>(task, sharedMemPool);
      break;
    }
    default: {
      HIP_DEVICE_ASSERT(false);
    }
  }
}

template <typename T>
constexpr T max(T a, T b) {
  return a > b ? a : b;
}

template <typename MOE_METADATA>
__global__ void moeKernel(const typename MOE_METADATA::MOE_PROBLEM_CONFIG::
                              TDataType* __restrict__ tokens,
                          const typename MOE_METADATA::MOE_PROBLEM_CONFIG::
                              TDataType* __restrict__ gateWeights,
                          const typename MOE_METADATA::MOE_PROBLEM_CONFIG::
                              TDataType* __restrict__ ffn1ExpertWeights,
                          const typename MOE_METADATA::MOE_PROBLEM_CONFIG::
                              TDataType* __restrict__ ffn2ExpertWeights,
                          typename MOE_METADATA::MOE_PROBLEM_CONFIG::
                              TDataType* __restrict__ ffn1Output,
                          typename MOE_METADATA::MOE_PROBLEM_CONFIG::
                              TDataType* __restrict__ ffn2Output,
                          typename MOE_METADATA::MOE_PROBLEM_CONFIG::
                              TDataType* __restrict__ finalOutput,
                          int tokensNum) {
  constexpr int SHARED_MEM_SIZE_BYTES = max(
      MOE_METADATA::TILES_CONFIG::GATE_TILE_METADATA::SHARED_MEM_NEEDES_BYTES,
      max(MOE_METADATA::TILES_CONFIG::FFN1_TILE_METADATA::
              SHARED_MEM_NEEDES_BYTES,
          MOE_METADATA::TILES_CONFIG::FFN2_TILE_METADATA::
              SHARED_MEM_NEEDES_BYTES));

  __shared__ char sharedMemPool[SHARED_MEM_SIZE_BYTES];

  for (int tokenIdx = blockIdx.x; tokenIdx < tokensNum; tokenIdx += gridDim.x) {
    typename MOE_METADATA::TILES_CONFIG::GATE_TILE_METADATA::TOutputType
        out_regs[MOE_METADATA::TILES_CONFIG::GATE_TILE_METADATA::
                     THREAD_OUTPUT_SIZE];

    GemmTile_block<typename MOE_METADATA::TILES_CONFIG::GATE_TILE_METADATA>(
        tokens, gateWeights, out_regs, tokenIdx, 0, sharedMemPool);
    out_regs[0] =
        Softmax_block<MOE_METADATA::TILES_CONFIG::GATE_TILE_METADATA::THREADS,
                      float>(out_regs[0], sharedMemPool);

    char* sharedMemPoolBytes = reinterpret_cast<char*>(sharedMemPool);
    float* s_topkVals = reinterpret_cast<float*>(sharedMemPoolBytes);
    int* s_topkIdx = reinterpret_cast<int*>(
        sharedMemPoolBytes +
        sizeof(float) * MOE_METADATA::MOE_PROBLEM_CONFIG::TOPK);
    int* nextPool = s_topkIdx + MOE_METADATA::MOE_PROBLEM_CONFIG::TOPK;

    Topk8_block<MOE_METADATA::TILES_CONFIG::GATE_TILE_METADATA::THREADS, float>(
        out_regs[0], s_topkVals, s_topkIdx, nextPool);

    // Warp 0 pushes all the tasks...
    for (int i = 0; i < MOE_METADATA::MOE_PROBLEM_CONFIG::TOPK; ++i) {
      MoeTaskDesc rTask;
      rTask.ffn1ExpertWeights = ffn1ExpertWeights;
      rTask.ffn2ExpertWeights = ffn2ExpertWeights;
      rTask.tokens = tokens;
      rTask.ffn1Output = ffn1Output;
      rTask.ffn2Output = ffn2Output;
      rTask.finalOutput = finalOutput;
      rTask.tokenIdx = tokenIdx;
      rTask.expertIdx = s_topkIdx[i];
      rTask.expertWeight = s_topkVals[i];
      rTask.topkSlotIdx = i;
      rTask.taskType = 0;

      constexpr int FFN1_CHUNKS =
          MOE_METADATA::TILES_CONFIG::FFN1_TILE_METADATA::N /
          MOE_METADATA::TILES_CONFIG::FFN1_TILE_METADATA::TILE_N;
#pragma unroll
      for (int i = 0; i < FFN1_CHUNKS; ++i) {
        rTask.blockTileColStartIdx = i;
        globalTaskManager.PushTask_warp(&rTask);
      }
    }
  }

  workerTaskSystemLoop_block(globalTaskManager,
                             ExecuteGeneralTask<MOE_METADATA>, sharedMemPool);
}

struct InternalState {
  TaskManagerType taskManagerHost;
  int* ffn1SyncArrayDevice;
  int* chunkReduceSyncArrayDevice;
  void* ffn1Output_device;
  void* ffn2Output_device;
};

template <typename MOE_METADATA>
hipError_t moeKernelLaunchFunc(InternalState* state, const void* tokens,
                               const void* gateWeights,
                               const void* ffn1ExpertWeights,
                               const void* ffn2ExpertWeights, void* output,
                               int tokensNum, hipStream_t stream) {
  using TMoeDataType = typename MOE_METADATA::MOE_PROBLEM_CONFIG::TDataType;
  dim3 grid(MOE_METADATA::HW_CONFIG::BLOCKS);
  dim3 block(MOE_METADATA::HW_CONFIG::THREADS);

  const int ffn1TasksNum =
      tokensNum * MOE_METADATA::MOE_PROBLEM_CONFIG::TOPK *
      (MOE_METADATA::MOE_PROBLEM_CONFIG::EXPERT_INTERMEDIATE_SIZE /
       MOE_METADATA::TILES_CONFIG::FFN1_TILE_METADATA::TILE_N);
  const int ffn2TasksNum =
      tokensNum * MOE_METADATA::MOE_PROBLEM_CONFIG::TOPK *
      (MOE_METADATA::MOE_PROBLEM_CONFIG::HIDDEN_SIZE /
       MOE_METADATA::TILES_CONFIG::FFN2_TILE_METADATA::TILE_N);
  const int reductionTasksNum =
      tokensNum * MOE_METADATA::TILES_CONFIG::REDUCTION_CHUNKS_PER_TOKEN;

  const int expectedNumTasks = ffn1TasksNum + ffn2TasksNum + reductionTasksNum;
  HIP_ERROR_CHECK(
      state->taskManagerHost.PrepareForNextLaunch(stream, expectedNumTasks));

  HIP_ERROR_CHECK(hipMemsetAsync(
      state->ffn1SyncArrayDevice, 0,
      sizeof(int) * tokensNum * MOE_METADATA::MOE_PROBLEM_CONFIG::TOPK,
      stream));

  HIP_ERROR_CHECK(
      hipMemsetAsync(state->chunkReduceSyncArrayDevice, 0,
                     sizeof(int) * tokensNum *
                         MOE_METADATA::TILES_CONFIG::REDUCTION_CHUNKS_PER_TOKEN,
                     stream));

  moeKernel<MOE_METADATA><<<grid, block, 0, stream>>>(
      static_cast<const TMoeDataType*>(tokens),
      static_cast<const TMoeDataType*>(gateWeights),
      static_cast<const TMoeDataType*>(ffn1ExpertWeights),
      static_cast<const TMoeDataType*>(ffn2ExpertWeights),
      static_cast<TMoeDataType*>(state->ffn1Output_device),
      static_cast<TMoeDataType*>(state->ffn2Output_device),
      static_cast<TMoeDataType*>(output), tokensNum);

  return hipSuccess;
}

////////////////////////////////////////////////////////////////
hipError_t MoeKernelLauncher::Init(hipStream_t stream, int maxTokens) {
  m_internalState = new InternalState();

  const int ffn1OutputSize =
      maxTokens * moe::MoeImplMetadata::MOE_PROBLEM_CONFIG::TOPK *
      moe::MoeImplMetadata::MOE_PROBLEM_CONFIG::EXPERT_INTERMEDIATE_SIZE;
  const int ffn2OutputSize =
      maxTokens * moe::MoeImplMetadata::MOE_PROBLEM_CONFIG::TOPK *
      moe::MoeImplMetadata::MOE_PROBLEM_CONFIG::HIDDEN_SIZE;

  HIP_ERROR_CHECK(m_internalState->taskManagerHost.Init(stream));
  HIP_ERROR_CHECK(hipMemcpyToSymbolAsync(
      HIP_SYMBOL(globalTaskManager), &m_internalState->taskManagerHost,
      sizeof(TaskManagerType), 0, hipMemcpyHostToDevice, stream));

  HIP_ERROR_CHECK(hipMallocAsync(
      &m_internalState->ffn1SyncArrayDevice,
      sizeof(int) * maxTokens * moe::MoeImplMetadata::MOE_PROBLEM_CONFIG::TOPK,
      stream));
  HIP_ERROR_CHECK(hipMemcpyToSymbolAsync(
      HIP_SYMBOL(globalFFN1SyncArray), &m_internalState->ffn1SyncArrayDevice,
      sizeof(int*), 0, hipMemcpyHostToDevice, stream));

  HIP_ERROR_CHECK(hipMallocAsync(
      &m_internalState->chunkReduceSyncArrayDevice,
      sizeof(int) * maxTokens *
          moe::MoeImplMetadata::TILES_CONFIG::REDUCTION_CHUNKS_PER_TOKEN,
      stream));
  HIP_ERROR_CHECK(
      hipMemcpyToSymbolAsync(HIP_SYMBOL(globalChunkReduceSyncArray),
                             &m_internalState->chunkReduceSyncArrayDevice,
                             sizeof(int*), 0, hipMemcpyHostToDevice, stream));

  HIP_ERROR_CHECK(hipMalloc(
      &m_internalState->ffn1Output_device,
      ffn1OutputSize *
          sizeof(moe::MoeImplMetadata::MOE_PROBLEM_CONFIG::TDataType)));
  HIP_ERROR_CHECK(hipMalloc(
      &m_internalState->ffn2Output_device,
      ffn2OutputSize *
          sizeof(moe::MoeImplMetadata::MOE_PROBLEM_CONFIG::TDataType)));

  HIP_ERROR_CHECK(hipStreamSynchronize(stream));

  return hipSuccess;
}

////////////////////////////////////////////////////////////////
hipError_t MoeKernelLauncher::Deinit(hipStream_t stream) {
  HIP_ERROR_CHECK(hipFree(m_internalState->ffn1Output_device));
  HIP_ERROR_CHECK(hipFree(m_internalState->ffn2Output_device));

  HIP_ERROR_CHECK(hipFreeAsync(m_internalState->ffn1SyncArrayDevice, stream));
  HIP_ERROR_CHECK(
      hipFreeAsync(m_internalState->chunkReduceSyncArrayDevice, stream));
  HIP_ERROR_CHECK(m_internalState->taskManagerHost.Deinit(stream));

  HIP_ERROR_CHECK(hipStreamSynchronize(stream));

  delete m_internalState;
  return hipSuccess;
}

////////////////////////////////////////////////////////////////
hipError_t MoeKernelLauncher::Launch(const void* tokens,
                                     const void* gateWeights,
                                     const void* ffn1ExpertWeights,
                                     const void* ffn2ExpertWeights,
                                     void* output, int tokensNum,
                                     hipStream_t stream) {
  HIP_ERROR_CHECK(moeKernelLaunchFunc<moe::MoeImplMetadata>(
      m_internalState, tokens, gateWeights, ffn1ExpertWeights,
      ffn2ExpertWeights, output, tokensNum, stream));
  return hipSuccess;
}

}  // namespace moe