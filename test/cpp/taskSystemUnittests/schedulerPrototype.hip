#include "../../../src/hip/hipCommon.h"
#include "../../../src/hip/taskSystem/taskManager.h"
#include "schedulerPrototype.h"

// TODO: align Task struct to cache line size to avoid cache trashing.
struct __align__(16) Task {
  uint32_t taskId;
  uint32_t blockIdx;
};

using TaskManagerType = TaskManager<Task, 1024>;
__constant__ TaskManagerType globalTaskManager;

__global__ void testWorkItemQueue() {
  __shared__ Task s_task;

  for (int i = 0; i < 2; ++i) {
    // All threads create the same task.
    Task rTask;
    rTask.taskId = i;
    rTask.blockIdx = blockIdx.x;

    globalTaskManager.PushTask_warp(&rTask);
    __syncthreads();
  }

  // Consumer block.
  for (int i = 0; i < 2; ++i) {
    // Get new task:
    globalTaskManager.PopTask_block(&s_task);

    // ---------------------

    const auto taskblockIdx = s_task.blockIdx;
    const auto taskId = s_task.taskId;

    if (threadIdx.x == 100) {
      printf(
          "Consumer BlockIdx: %i, task->blockIdx: %i, task->ID: "
          "%i\n",
          blockIdx.x, taskblockIdx, taskId);
    }

    // Mock execute task...
    for (volatile int i = 0; i < 100000; ++i);
    __syncthreads();
  }
}

void runScheduler() {
  TaskManagerType managerHost;
  managerHost.Init();

  HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(globalTaskManager), &managerHost,
                              sizeof(TaskManagerType), 0, hipMemcpyHostToDevice));

  dim3 gridDim(300, 1, 1);
  dim3 blockDim(128, 1, 1);

  testWorkItemQueue<<<gridDim, blockDim, 0, 0>>>();

  HIP_CHECK(hipDeviceSynchronize())
  managerHost.Deinit();
}