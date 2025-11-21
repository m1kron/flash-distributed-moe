#pragma once
#include "src/hip/kernel/tasks/internal/gemmTileBlock.h"
#include "src/hip/kernel/tasks/taskDesc.h"
#include "src/hip/utils/hipDeviceUtils.h"

namespace moe {

// TODO: do sth with __constant__ scattered over multiple files..
__constant__ int* globalChunkReduceSyncArray;

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
}  // namespace moe