#pragma once
#include "src/hip/kernel/tasks/internal/reduceBlock.h"
#include "src/hip/kernel/tasks/taskDesc.h"
#include "src/hip/utils/hipDeviceUtils.h"

namespace moe {
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

}  // namespace moe