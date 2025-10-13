#include "../../../src/hip/hipCommon.h"
#include "../../../src/hip/taskSystem/taskManager.h"
#include "schedulerPrototype.h"

// TODO: align Task struct to cache line size to avoid cache trashing.
struct __align__(16) Task {
  uint32_t taskId;
  uint32_t blockIdx;
};

using TaskManagerType = TaskManager<Task, 2048>;
__constant__ TaskManagerType globalTaskManager;

__global__ void testWorkItemQueue() {
  __shared__ Task s_task;
  __shared__ bool globalInterrupted;
  if (threadIdx.x == 0) {
    globalInterrupted = false;
  }

  // All threads create the same task.
  Task rTask;
  rTask.taskId = 0;
  rTask.blockIdx = blockIdx.x;

  // by only warp0 adds it.
  globalTaskManager.PushTask_warp(&rTask);
  __syncthreads();

  while (true) {
    // if (threadIdx.x == 65 || threadIdx.x == 1) {
    //   printf(
    //       "Consumer BlockIdx: %i, threadIdx: %i: Before pop task loop\n",
    //       blockIdx.x, threadIdx.x);
    // }
    // Get new task:
    const bool shouldInterrupt = globalTaskManager.PopTask_warp(&s_task);
    if (shouldInterrupt && threadIdx.x == 0) {
      globalInterrupted = true;
      printf("BlockIdx: %i, interrupted!\n", blockIdx.x);
    }

    // if (threadIdx.x == 65 || threadIdx.x == 1) {
    //   printf(
    //       "Consumer BlockIdx: %i, threadIdx: %i: After pop task loop, wating on barrier...\n",
    //       blockIdx.x, threadIdx.x);
    // }

    __syncthreads();

    // if (threadIdx.x == 65 || threadIdx.x == 1) {
    //   printf(
    //       "Consumer BlockIdx: %i, threadIdx: %i: __syncthreads() crossed!\n",
    //       blockIdx.x, threadIdx.x);
    // }

    if (globalInterrupted) {
      if (threadIdx.x == 1) {
        printf("BlockIdx: %i, interrupted!\n", blockIdx.x);
      }
      break;
    }

    // ---------------------
    const auto taskblockIdx = s_task.blockIdx;
    const auto taskId = s_task.taskId;

    if (threadIdx.x == 65) {
      printf(
          "Consumer BlockIdx: %i, threadIdx: %i, executing task->blockIdx: %i, task->ID: "
          "%i\n",
          blockIdx.x, threadIdx.x, taskblockIdx, taskId);
    }

    // Mock execute task...
    for (volatile int i = 0; i < 100000; ++i);
    __syncthreads();

    // All threads create the same task.
    Task rTask;
    rTask.taskId = taskId + 1;
    rTask.blockIdx = taskblockIdx;

    if (rTask.taskId < 4) {
      globalTaskManager.PushTask_warp(&rTask);
    }

    // if (threadIdx.x == 65 || threadIdx.x == 1) {
    //   printf(
    //       "Consumer BlockIdx: %i, threadIdx: %i, waiting for other threads..\n", blockIdx.x, threadIdx.x);
    // }
    __syncthreads();
  }
}

void runScheduler() {
  const int blocksNum = 304;
  TaskManagerType managerHost;
  managerHost.Init(blocksNum*4);

  HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(globalTaskManager), &managerHost,
                              sizeof(TaskManagerType), 0,
                              hipMemcpyHostToDevice));

  dim3 gridDim(blocksNum, 1, 1);
  dim3 blockDim(128, 1, 1);

  testWorkItemQueue<<<gridDim, blockDim, 0, 0>>>();

  HIP_CHECK(hipDeviceSynchronize())
  managerHost.Deinit();
}