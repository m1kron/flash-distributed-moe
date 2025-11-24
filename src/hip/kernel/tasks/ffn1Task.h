#pragma once
#include "src/hip/kernel/tasks/internal/expertFFN1Block.h"
#include "src/hip/kernel/tasks/taskDesc.h"
#include "src/hip/utils/hipDeviceUtils.h"

namespace moe {

// TODO: do sth with __constant__ scattered over multiple files..
__constant__ int* globalFFN1SyncArray;

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
}  // namespace moe