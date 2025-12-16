#pragma once
#include "src/hip/common/metadata.h"
#include "src/hip/common/runtimeConfig.h"
#include "src/hip/kernel/tasks/ffn1Task.h"
#include "src/hip/kernel/tasks/ffn2Task.h"
#include "src/hip/kernel/tasks/internal/gateBlock.h"
#include "src/hip/kernel/tasks/reduceTask.h"
#include "src/hip/kernel/tasks/taskDesc.h"
#include "src/hip/taskSystem/workerLoop.h"
#include "src/hip/utils/hipDeviceUtils.h"

namespace moe {

template <typename RUNTIME_CONFIG>
__device__ void ExecuteGeneralTask(const MoeTaskDesc& task,
                                   void* sharedMemPool) {
  switch (task.taskType) {
    case TaskType::FFN1: {
      ExecuteExpertFFN1Task<RUNTIME_CONFIG>(task, sharedMemPool);
      break;
    }
    case TaskType::FFN2: {
      ExecuteExpertFFN2Task<RUNTIME_CONFIG>(task, sharedMemPool);
      break;
    }
    case TaskType::REDUCE: {
      ExecuteFinalReductionTask<typename RUNTIME_CONFIG::MOE_METADATA>(
          task, sharedMemPool);
      break;
    }
    default: {
      HIP_DEVICE_ASSERT(false);
    }
  }
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
  using RUNTIME_CONFIG = moe::MoeRuntimeConfig;
  __shared__ char sharedMemPool[RUNTIME_CONFIG::SHARED_MEM_SIZE_BYTES];

  using TType = typename MOE_METADATA::TILES_CONFIG::GATE_TILE_METADATA::TType;

  // Gate is scheduled statisically to avoid overhead of task system
  // scheduling.
  for (int tokenIdx = blockIdx.x; tokenIdx < tokensNum; tokenIdx += gridDim.x) {
    TType* topkVals_shared = nullptr;
    int* topkIdx_shared = nullptr;

    moe::tasks::internal::gate_block<
        typename RUNTIME_CONFIG::MOE_METADATA::TILES_CONFIG::GATE_TILE_METADATA,
        RUNTIME_CONFIG::MOE_METADATA::MOE_PROBLEM_CONFIG::TOPK>(
        tokens, gateWeights, tokenIdx, &topkVals_shared, &topkIdx_shared,
        sharedMemPool);

    // Warp 0 pushes all the tasks...
    // TODO: make it more paralllel.
    for (int i = 0; i < RUNTIME_CONFIG::MOE_METADATA::MOE_PROBLEM_CONFIG::TOPK;
         ++i) {
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
          RUNTIME_CONFIG::MOE_METADATA::TILES_CONFIG::FFN1_TILE_METADATA::N /
          RUNTIME_CONFIG::MOE_METADATA::TILES_CONFIG::FFN1_TILE_METADATA::
              TILE_N;
#pragma unroll
      for (int i = 0; i < FFN1_CHUNKS; ++i) {
        rTask.blockTileColStartIdx = i;
        globalTaskManager.PushTask_warp(&rTask);
      }
    }
  }

  workerTaskSystemLoop_block(globalTaskManager,
                             ExecuteGeneralTask<RUNTIME_CONFIG>, sharedMemPool);
}
}  // namespace moe