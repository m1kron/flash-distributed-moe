#pragma once
#include "src/hip/common/metadata.h"
#include "src/hip/kernel/tasks/ffn1Task.h"
#include "src/hip/kernel/tasks/ffn2Task.h"
#include "src/hip/kernel/tasks/internal/gateBlock.h"
#include "src/hip/kernel/tasks/reduceTask.h"
#include "src/hip/kernel/tasks/taskDesc.h"
#include "src/hip/taskSystem/workerLoop.h"
#include "src/hip/utils/hipDeviceUtils.h"

namespace moe {

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

// Main MOE kernel.
template <typename MOE_METADATA>
__global__ void moeKernel(
    const typename MOE_METADATA::MOE_PROBLEM_CONFIG::TDataType* tokens,
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
    typename MOE_METADATA::MOE_PROBLEM_CONFIG::TDataType* finalOutput,
    int tokensNum) {
  constexpr int SHARED_MEM_SIZE_BYTES = max(
      MOE_METADATA::TILES_CONFIG::GATE_TILE_METADATA::SHARED_MEM_NEEDES_BYTES,
      max(MOE_METADATA::TILES_CONFIG::FFN1_TILE_METADATA::
              SHARED_MEM_NEEDES_BYTES,
          MOE_METADATA::TILES_CONFIG::FFN2_TILE_METADATA::
              SHARED_MEM_NEEDES_BYTES));

  __shared__ char sharedMemPool[SHARED_MEM_SIZE_BYTES];

  using TType = typename MOE_METADATA::TILES_CONFIG::GATE_TILE_METADATA::TType;

  // Gate is scheduled statisically to avoid overhead of task system
  // scheduling.
  for (int tokenIdx = blockIdx.x; tokenIdx < tokensNum; tokenIdx += gridDim.x) {
    TType* topkVals_shared = nullptr;
    int* topkIdx_shared = nullptr;

    moe::tasks::internal::gate_block<
        typename MOE_METADATA::TILES_CONFIG::GATE_TILE_METADATA,
        MOE_METADATA::MOE_PROBLEM_CONFIG::TOPK>(tokens, gateWeights, tokenIdx,
                                                &topkVals_shared,
                                                &topkIdx_shared, sharedMemPool);

    // Warp 0 pushes all the tasks...
    // TODO: make it more paralllel.
    for (int i = 0; i < MOE_METADATA::MOE_PROBLEM_CONFIG::TOPK; ++i) {
      MoeTaskDesc rTask;
      rTask.ffn1ExpertWeights = ffn1ExpertWeights;
      rTask.ffn2ExpertWeights = ffn2ExpertWeights;
      rTask.tokens = tokens;
      rTask.ffn1Output = ffn1Output;
      rTask.ffn2Output = ffn2Output;
      rTask.finalOutput = finalOutput;
      rTask.tokenIdx = tokenIdx;
      rTask.expertIdx = topkIdx_shared[i];
      rTask.expertWeight = topkVals_shared[i];
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