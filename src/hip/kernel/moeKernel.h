#pragma once
#include "src/hip/kernel/remote/remoteComManager.h"
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
template <typename RUNTIME_CONFIG>
__global__ void moeKernel(
    const typename RUNTIME_CONFIG::MOE_METADATA::MOE_PROBLEM_CONFIG::TDataType*
        tokens,
    const typename RUNTIME_CONFIG::MOE_METADATA::MOE_PROBLEM_CONFIG::
        TDataType* __restrict__ gateWeights,
    const typename RUNTIME_CONFIG::MOE_METADATA::MOE_PROBLEM_CONFIG::
        TDataType* __restrict__ ffn1ExpertWeights,
    const typename RUNTIME_CONFIG::MOE_METADATA::MOE_PROBLEM_CONFIG::
        TDataType* __restrict__ ffn2ExpertWeights,
    typename RUNTIME_CONFIG::MOE_METADATA::MOE_PROBLEM_CONFIG::
        TDataType* __restrict__ ffn1Output,
    typename RUNTIME_CONFIG::MOE_METADATA::MOE_PROBLEM_CONFIG::
        TDataType* __restrict__ ffn2Output,
    typename RUNTIME_CONFIG::MOE_METADATA::MOE_PROBLEM_CONFIG::TDataType*
        finalOutput,
    int tokensNum) {
  constexpr int TOPK = RUNTIME_CONFIG::MOE_METADATA::MOE_PROBLEM_CONFIG::TOPK;

  __shared__ char sharedMemPool[RUNTIME_CONFIG::SHARED_MEM_SIZE_BYTES];

  using TType = typename RUNTIME_CONFIG::MOE_METADATA::TILES_CONFIG::
      GATE_TILE_METADATA::TType;

  // Gate is scheduled statisically to avoid overhead of task system
  // scheduling.
  for (int tokenIdx = blockIdx.x; tokenIdx < tokensNum; tokenIdx += gridDim.x) {
    TType* topkVals_shared = nullptr;
    int* topkIdx_shared = nullptr;

    moe::tasks::internal::gate_block<
        typename RUNTIME_CONFIG::GEMM_RUNTIME_CONFIG::GATE_GEMM_TILE_IMPL,
        TOPK>(tokens, gateWeights, tokenIdx, &topkVals_shared, &topkIdx_shared,
              sharedMemPool);

    bool tokensWillBeSent = false;

    // Warp 0 pushes all the tasks...
    // TODO: make it more paralllel.
    for (int i = 0; i < TOPK; ++i) {
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

      constexpr int EXPERTS_NUM =
          RUNTIME_CONFIG::MOE_METADATA::MOE_PROBLEM_CONFIG::EXPERTS_NUM;
      const int expertsPerGpu =
          EXPERTS_NUM / globalRemoteComManager.GetWorldSize();
      const int expertIdxGlobal = rTask.expertIdx;
      const int dstPeOfExpert = expertIdxGlobal / expertsPerGpu;

      rTask.expertIdx = expertIdxGlobal % expertsPerGpu;

      const bool canBeProcessedLocally =
          dstPeOfExpert == globalRemoteComManager.GetRank();

      if (canBeProcessedLocally) {
        if (threadIdx.x == 0) {
          printf(
              "RANK: %i: Token %i for global expert id: %i - will be processed "
              "locally.\n",
              globalRemoteComManager.GetRank(), rTask.tokenIdx,
              expertIdxGlobal);
        }

        constexpr int FFN1_CHUNKS =
            RUNTIME_CONFIG::MOE_METADATA::TILES_CONFIG::FFN1_TILE_METADATA::N /
            RUNTIME_CONFIG::MOE_METADATA::TILES_CONFIG::FFN1_TILE_METADATA::
                TILE_N;
#pragma unroll
        for (int i = 0; i < FFN1_CHUNKS; ++i) {
          rTask.blockTileColStartIdx = i;
          globalTaskManager.PushTask_warp(&rTask);
        }
      } else {
        tokensWillBeSent = true;
        if (threadIdx.x == 0) {
          printf(
              "RANK: %i: Token %i for global expert id: %i - sending to rank: "
              "%i\n",
              globalRemoteComManager.GetRank(), rTask.tokenIdx, expertIdxGlobal,
              dstPeOfExpert);
        }

        if (threadIdx.x == 0) {
          constexpr int tasksToDecrease =
              RUNTIME_CONFIG::GetOfTasksForOneTokenProcessedLocally();
          const auto prev =
              globalTaskManager.DecreaseNumOfExpectedMaxTasks(tasksToDecrease);

          printf(
              "RANK: %i: Decreasing number of max expected local tasks by: "
              "%i, prev number of tasks: %i\n",
              globalRemoteComManager.GetRank(), tasksToDecrease, prev);
        }
      }
    }
    {
      if (threadIdx.x == 0 && tokensWillBeSent) {
        // TEMP HACK:
        constexpr int reductionTasksNum = RUNTIME_CONFIG::MOE_METADATA::
            TILES_CONFIG::REDUCTION_CHUNKS_PER_TOKEN;

        const auto prev =
            globalTaskManager.DecreaseNumOfExpectedMaxTasks(reductionTasksNum);

        printf(
            "RANK: %i: Decreasing number of max expected local tasks by: "
            "%i, prev number of tasks: %i\n",
            globalRemoteComManager.GetRank(), reductionTasksNum, prev);
      }
    }
  }

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("RANK: %i: Kernel starts worker loop.\n",
           globalRemoteComManager.GetRank());
  }

  workerTaskSystemLoop_block(globalTaskManager,
                             ExecuteGeneralTask<RUNTIME_CONFIG>, sharedMemPool);

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("RANK: %i: Kernel done.\n", globalRemoteComManager.GetRank());
  }
}
}  // namespace moe