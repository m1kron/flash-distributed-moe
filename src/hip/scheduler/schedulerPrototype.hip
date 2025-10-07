#include "../hipCommon.h"
#include "schedulerPrototype.h"
#include "../workQueue/workQueue.h"

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

  if(blockIdx.x == 0 || blockIdx.x == 1) {
    // Producer block.
    if (threadIdx.x == 0) {
      for (int i = 0; i < gridDim.x-2; ++i) {
        Task* task = globalTaskManager.AllocateTask();
        const clock_t time = clock64();
        task->creationTime = time;
        task->blockIdx = blockIdx.x;

        globalTaskManager._workQueue.Push(task);

        printf("Producer BlockIdx: %i, added task creation time: %li\n", blockIdx.x, time);
      }
    }

  } else {
    // Consumer block.
    if (threadIdx.x == 0) {
      Task* task = globalTaskManager._workQueue.PopOrBusyWait();

      printf("Consumer BlockIdx: %i, got new task!\n", blockIdx.x);
      const auto taskblockIdx = task->blockIdx;
      const auto creationTime = task->creationTime;

      printf("Consumer BlockIdx: %i, task->blockIdx: %i, task->creationTime: %li\n", blockIdx.x, taskblockIdx, creationTime);
    }
  } 
}

void runScheduler() {

  Manager managerHost;
  managerHost.Init();


  HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(globalTaskManager), &managerHost,
                              sizeof(Manager), 0, hipMemcpyHostToDevice));

  dim3 gridDim(100, 1, 1);
  dim3 blockDim(128, 1, 1);

  testWorkItemQueue<<<gridDim, blockDim, 0, 0>>>();


  HIP_CHECK(hipDeviceSynchronize())
  managerHost.Deinit();

}