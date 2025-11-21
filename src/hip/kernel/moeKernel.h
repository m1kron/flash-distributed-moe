#pragma once
#include "src/hip/common/metadata.h"
#include "src/hip/kernel/tasks/internal/expertFFN1Block.h"
#include "src/hip/kernel/tasks/internal/gemmTileBlock.h"
#include "src/hip/kernel/tasks/internal/reduceBlock.h"
#include "src/hip/kernel/tasks/internal/softmaxBlock.h"
#include "src/hip/kernel/tasks/internal/topkBlock.h"
#include "src/hip/taskSystem/taskManager.h"
#include "src/hip/taskSystem/workerLoop.h"
#include "src/hip/utils/hipDeviceUtils.h"

namespace moe {

enum TaskType : int { FFN1, FFN2, REDUCE };

struct __align__(16) MoeTaskDesc {
  const void* __restrict__ ffn1ExpertWeights;
  const void* __restrict__ ffn2ExpertWeights;
  const void* __restrict__ tokens;
  void* __restrict__ ffn1Output;
  void* __restrict__ ffn2Output;
  void* __restrict__ finalOutput;
  float expertWeight;
  int tokenIdx;
  int expertIdx;
  int blockTileColStartIdx;
  int topkSlotIdx;
  TaskType taskType;
};

using TaskManagerType = TaskManager<MoeTaskDesc, 8192>;

static __constant__ TaskManagerType globalTaskManager;
static __constant__ int* globalFFN1SyncArray;
static __constant__ int* globalChunkReduceSyncArray;

template <typename MOE_METADATA>
__device__ void ExecuteExpertFFN1Task(const MoeTaskDesc& task,
                                      void* sharedMemPool) {
  using T_FFN1_TILE = typename MOE_METADATA::TILES_CONFIG::FFN1_TILE_METADATA;
  using TType = typename T_FFN1_TILE::TType;
  if (threadIdx.x == 65) {
    HIP_DEVICE_LOG(
        "Worker %i: Executes tile no: %i, tokenIdx: %i, expert: %i, weight: "
        "%f, col: %i\n",
        blockIdx.x, task.blockTileColStartIdx, task.tokenIdx, task.expertIdx,
        task.expertWeight, task.blockTileColStartIdx);
  }

  const TType* thisBlockTokens =
      static_cast<const TType*>(task.tokens) + task.tokenIdx * T_FFN1_TILE::K;
  const TType* thisffn1ExpertWeights =
      static_cast<const TType*>(task.ffn1ExpertWeights) +
      task.expertIdx * T_FFN1_TILE::K * T_FFN1_TILE::N;

  TType outRegs[T_FFN1_TILE::THREAD_OUTPUT_SIZE];

  tasks::internal::expertFFN1_block<T_FFN1_TILE>(
      thisBlockTokens, thisffn1ExpertWeights, outRegs, 0,
      task.blockTileColStartIdx, sharedMemPool);

  TType* thisBlockOutput =
      static_cast<TType*>(task.ffn1Output) +
      task.tokenIdx * MOE_METADATA::MOE_PROBLEM_CONFIG::TOPK * T_FFN1_TILE::N +
      task.topkSlotIdx * T_FFN1_TILE::N;

  moe::tasks::internal::WriteGemmTileToGlobalMem_block<T_FFN1_TILE>(
      outRegs, thisBlockOutput, 0, task.blockTileColStartIdx);

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
    ffn2Task.taskType = TaskType::FFN2;

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
  using TType = typename T_FFN2_TILE::TType;
  if (threadIdx.x == 65) {
    HIP_DEVICE_LOG(
        "Worker %i: Executes tile no: %i, tokenIdx: %i, expert: %i, weight: "
        "%f\n",
        blockIdx.x, task.blockTileColStartIdx, task.tokenIdx, task.expertIdx,
        task.expertWeight);
  }

  const TType* thisBlockInput =
      static_cast<const TType*>(task.ffn1Output) +
      task.tokenIdx * MOE_METADATA::MOE_PROBLEM_CONFIG::TOPK * T_FFN2_TILE::K +
      task.topkSlotIdx * T_FFN2_TILE::K;

  const TType* thisffn2ExpertWeights =
      static_cast<const TType*>(task.ffn2ExpertWeights) +
      task.expertIdx * T_FFN2_TILE::K * T_FFN2_TILE::N;

  TType outRegs[T_FFN2_TILE::THREAD_OUTPUT_SIZE];

  moe::tasks::internal::GemmTile_block<T_FFN2_TILE>(
      thisBlockInput, thisffn2ExpertWeights, outRegs, 0,
      task.blockTileColStartIdx, sharedMemPool);

  TType* thisBlockOutput =
      static_cast<TType*>(task.ffn2Output) +
      task.tokenIdx * MOE_METADATA::MOE_PROBLEM_CONFIG::TOPK * T_FFN2_TILE::N +
      task.topkSlotIdx * T_FFN2_TILE::N;

#pragma unroll
  for (int i = 0; i < T_FFN2_TILE::THREAD_OUTPUT_SIZE; ++i) {
    outRegs[i] *= task.expertWeight;
  }

  moe::tasks::internal::WriteGemmTileToGlobalMem_block<T_FFN2_TILE>(
      outRegs, thisBlockOutput, 0, task.blockTileColStartIdx);

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
    reductionTask.taskType = TaskType::REDUCE;
    reductionTask.blockTileColStartIdx =
        task.blockTileColStartIdx / FFN2_DEP_SIZE;
    globalTaskManager.PushTask_warp(&reductionTask);
  }
}

template <typename MOE_METADATA>
__device__ void ExecuteFinalReductionTask(const MoeTaskDesc& task,
                                          void* sharedMemPool) {
  using TType = typename MOE_METADATA::MOE_PROBLEM_CONFIG::TDataType;
  const TType* thisBlockInput =
      static_cast<const TType*>(task.ffn2Output) +
      task.tokenIdx * MOE_METADATA::MOE_PROBLEM_CONFIG::TOPK *
          MOE_METADATA::MOE_PROBLEM_CONFIG::HIDDEN_SIZE;

  TType* thisBlockOutput =
      static_cast<TType*>(task.finalOutput) +
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
    case TaskType::FFN1: {
      ExecuteExpertFFN1Task<MOE_METADATA>(task, sharedMemPool);
      break;
    }
    case TaskType::FFN2: {
      ExecuteExpertFFN2Task<MOE_METADATA>(task, sharedMemPool);
      break;
    }
    case TaskType::REDUCE: {
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

  using TType = typename MOE_METADATA::TILES_CONFIG::GATE_TILE_METADATA::TType;

  for (int tokenIdx = blockIdx.x; tokenIdx < tokensNum; tokenIdx += gridDim.x) {
    TType out_regs
        [MOE_METADATA::TILES_CONFIG::GATE_TILE_METADATA::THREAD_OUTPUT_SIZE];

    moe::tasks::internal::GemmTile_block<
        typename MOE_METADATA::TILES_CONFIG::GATE_TILE_METADATA>(
        tokens, gateWeights, out_regs, tokenIdx, 0, sharedMemPool);
    out_regs[0] = moe::tasks::internal::Softmax_block<
        MOE_METADATA::TILES_CONFIG::GATE_TILE_METADATA::THREADS, TType>(
        out_regs[0], sharedMemPool);

    char* sharedMemPoolBytes = reinterpret_cast<char*>(sharedMemPool);
    TType* s_topkVals = reinterpret_cast<TType*>(sharedMemPoolBytes);
    int* s_topkIdx = reinterpret_cast<int*>(
        sharedMemPoolBytes +
        sizeof(TType) * MOE_METADATA::MOE_PROBLEM_CONFIG::TOPK);
    int* nextPool = s_topkIdx + MOE_METADATA::MOE_PROBLEM_CONFIG::TOPK;

    moe::tasks::internal::Topk8_block<
        MOE_METADATA::TILES_CONFIG::GATE_TILE_METADATA::THREADS, TType>(
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
      rTask.taskType = TaskType::FFN1;

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
}  // namespace moe