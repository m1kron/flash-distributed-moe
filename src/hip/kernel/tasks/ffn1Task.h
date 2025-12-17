#pragma once
#include "src/hip/kernel/tasks/internal/expertFFN1Block.h"
#include "src/hip/kernel/tasks/internal/writeGemmTileBlock.h"
#include "src/hip/kernel/tasks/taskDesc.h"
#include "src/hip/utils/hipDeviceUtils.h"

namespace moe {

// TODO: do sth with __constant__ scattered over multiple files..
__constant__ int* globalFFN1SyncArray;

template <typename RUNTIME_CONFIG>
__device__ void ExecuteExpertFFN1Task(const MoeTaskDesc& task,
                                      void* sharedMemPool) {
  using FFN1_TILE_IMPL =
      typename RUNTIME_CONFIG::GEMM_RUNTIME_CONFIG::FFN1_GEMM_TILE_IMPL;
  using FFN1_TILE = typename FFN1_TILE_IMPL::TILE_METADATA;
  using TType = typename FFN1_TILE::TType;
  constexpr auto TOPK = RUNTIME_CONFIG::MOE_METADATA::MOE_PROBLEM_CONFIG::TOPK;

  const TType* thisBlockTokens =
      static_cast<const TType*>(task.tokens) + task.tokenIdx * FFN1_TILE::K;
  const TType* thisffn1ExpertWeights =
      static_cast<const TType*>(task.ffn1ExpertWeights) +
      task.expertIdx * FFN1_TILE::K * 2 * FFN1_TILE::N;

  TType outRegs[FFN1_TILE::THREAD_OUTPUT_SIZE];

  tasks::internal::expertFFN1_block<FFN1_TILE_IMPL>(
      thisBlockTokens, thisffn1ExpertWeights, outRegs, 0,
      task.blockTileColStartIdx, sharedMemPool);

  TType* thisBlockOutput = static_cast<TType*>(task.ffn1Output) +
                           task.tokenIdx * TOPK * FFN1_TILE::N +
                           task.topkSlotIdx * FFN1_TILE::N;

  moe::tasks::internal::WriteGemmTileToGlobalMem_block<FFN1_TILE>(
      outRegs, thisBlockOutput, 0, task.blockTileColStartIdx);

  __threadfence();
  constexpr int N_CHUNKS = FFN1_TILE::N / FFN1_TILE::TILE_N;
  bool* lastBlock = reinterpret_cast<bool*>(sharedMemPool);
  if (threadIdx.x == 0) {
    int processed = atomicAdd(
        &globalFFN1SyncArray[task.tokenIdx * TOPK + task.topkSlotIdx], 1);
    *lastBlock = ((processed + 1) == N_CHUNKS);
  }

  __syncthreads();
  if (*lastBlock) {
    // Add tasks for ffn2:
    MoeTaskDesc ffn2Task = task;
    ffn2Task.taskType = TaskType::FFN2;

    using FFN2_TILE = typename RUNTIME_CONFIG::GEMM_RUNTIME_CONFIG::
        FFN2_GEMM_TILE_IMPL::TILE_METADATA;

    constexpr int FFN2_CHUNKS = FFN2_TILE::N / FFN2_TILE::TILE_N;
    for (int colIdx = 0; colIdx < FFN2_CHUNKS; ++colIdx) {
      ffn2Task.blockTileColStartIdx = colIdx;
      globalTaskManager.PushTask_warp(&ffn2Task);
    }
  }
}
}  // namespace moe