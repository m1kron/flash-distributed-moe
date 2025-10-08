#include "../hipCommon.h"
#include "../workQueue/workQueue.h"
#include "schedulerPrototype.h"

// Note: For now I will try approach with single global work queue for all
// workers. This might be probelmatic due to contention, but it is implemented
// with atomic load and add only(assuminghw support here...) If the contention
// will be a problem, I will switch to backup solution, but that depends on
// workload which will be determined in the future.

// TODO: align Task struct to cache line size to avoid cash trashing.
struct __align__(16) Task {
  clock_t creationTime;
  uint32_t blockIdx;
};

struct Manager {
  uint32_t _allocatedTaskIdx;
  uint32_t _tasksSize;
  Task* _tasksAlloc;
  WorkQueue<Task, 1024> _workQueue;

  __host__ void Init() {
    _allocatedTaskIdx = 0;

    _workQueue.Init();
    _tasksSize = 1024;
    HIP_CHECK(hipMalloc(&_tasksAlloc, _tasksSize * sizeof(Task)));
    HIP_CHECK(hipMemset(_tasksAlloc, 0, _tasksSize * sizeof(Task)));
  }

  __host__ void Deinit() {
    _workQueue.Deinit();
    HIP_CHECK(hipFree(_tasksAlloc));
  }

  __device__ Task* AllocateTask() {
    const uint32_t idx = atomicAdd(&_allocatedTaskIdx, 1);
    if (idx >= _tasksSize) {
      assert(false);
      return nullptr;
    }
    return _tasksAlloc + idx;
  }

  __device__ uint32_t GetSize() const { return _tasksSize; }
};

__constant__ Manager globalTaskManager;

__global__ void testWorkItemQueue() {
  __shared__ Task s_task;
  if (blockIdx.x == 0 || blockIdx.x == 1) {
    // Producer block.
    for (int i = 0; i < gridDim.x - 2; ++i) {

      // All threads create the same task.
      Task rTask;
      rTask.creationTime = time;
      rTask.blockIdx = blockIdx.x;

      Task* task_globalMem = nullptr;

      if (threadIdx.x == 0) {
        // Thread 0 allocates task in global mem:
        task_globalMem = globalTaskManager.AllocateTask();
      }

      // Warp 0 cooperatvely fill task in global mem:
      unsigned long long allocatedTaskInt =
          reinterpret_cast<unsigned long long>(task_globalMem);

      // Broadcast pointer to all threads in the block.
      allocatedTaskInt = __shfl(allocatedTaskInt, 0);

      task_globalMem = reinterpret_cast<Task*>(allocatedTaskInt);

      char* taskBlob_g = reinterpret_cast<char*>(&task_globalMem);
      const char* taskBlob_r = reinterpret_cast<char*>(rTask);

      if (threadIdx.x / warpSize == 0) {
        for (int i = threadIdx.x; i < sizeof(Task); i += warpSize) {
          taskBlob_g[b] = taskBlob_r[bi];
        }
      }

      // Thread 0 pushes task to the queue:
      if(threadIdx.x == 0) {
        globalTaskManager._workQueue.Push(task_globalMem);
        printf("Producer BlockIdx: %i, added task creation time: %li\n",
               blockIdx.x, time);
        }
      }
      __syncthreads();
    }
  } else {
    // Consumer block.
    for (int i = 0; i < 2; ++i) {
      // Get new task:
      Task* task = nullptr;
      if (threadIdx.x == 0) {
        task = globalTaskManager._workQueue.PopOrBusyWait();
        printf("Consumer BlockIdx: %i, got new task!\n", blockIdx.x);
      }
      unsigned long long poppedTaskInt =
          reinterpret_cast<unsigned long long>(task);

      // Broadcast pointer to all threads in the block.
      poppedTaskInt = __shfl(poppedTaskInt, 0);

      task = reinterpret_cast<Task*>(poppedTaskInt);

      char* taskBlob_s = reinterpret_cast<char*>(&s_task);
      const char* taskBlob_g = reinterpret_cast<char*>(task);

      if (threadIdx.x / warpSize == 0) {
        for (int i = threadIdx.x; i < sizeof(Task); i += warpSize) {
          taskBlob_s[b] = taskBlob_g[bi];
        }
      }

      __syncthreads();

      // ---------------------

      const auto taskblockIdx = s_task.blockIdx;
      const auto creationTime = s_task.creationTime;

      if (threadIdx.x == 100) {
        printf(
            "Consumer BlockIdx: %i, task->blockIdx: %i, task->creationTime: "
            "%li\n",
            blockIdx.x, taskblockIdx, creationTime);
      }

      // Mock execute task...
      for (volatile int i = 0; i < 100000; ++i)
        ;
      __syncthreads();
    }
  }
}

void runScheduler() {
  Manager managerHost;
  managerHost.Init();

  HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(globalTaskManager), &managerHost,
                              sizeof(Manager), 0, hipMemcpyHostToDevice));

  dim3 gridDim(300, 1, 1);
  dim3 blockDim(128, 1, 1);

  testWorkItemQueue<<<gridDim, blockDim, 0, 0>>>();

  HIP_CHECK(hipDeviceSynchronize())
  managerHost.Deinit();
}