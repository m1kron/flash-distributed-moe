#pragma once
#include "../utils/hipDeviceUtils.h"

// Implements standard task system block-wide loop, where block waits for new
// task, and execute it with executeTask functor.
template <typename TaskManagerType, typename TExecuteTaskFun>
__device__ void workerTaskSystemLoop_block(TaskManagerType& globalTaskManager,
                                           const TExecuteTaskFun executeTask,
                                           void* sharedMemPool) {
  using TTaskType = typename TaskManagerType::TTaskType;
  char* sharedMemPoolBytes = static_cast<char*>(sharedMemPool);

  __syncthreads();
  while (true) {
    bool* shouldExitNow_s = reinterpret_cast<bool*>(sharedMemPoolBytes);
    TTaskType* task_s =
        reinterpret_cast<TTaskType*>(sharedMemPoolBytes + sizeof(bool));
    // Get new task.
    const bool noMoreTasks = globalTaskManager.WaitAndPopTask_warp(task_s);

    if (threadIdx.x == 21) {
      HIP_DEVICE_LOG("Worker: Got new task\n");
    }

    // It is possible that we recieved signal to finish the loop,
    // so propafate it to all threads via shared mem.
    if (threadIdx.x == 0) {
      *shouldExitNow_s = noMoreTasks;
    }

    __syncthreads();
    if (*shouldExitNow_s) {
      break;
    }

    // Block cooperatively execute task.
    TTaskType rTask = *task_s;
    __syncthreads();
    executeTask(rTask, sharedMemPool);
  }
}