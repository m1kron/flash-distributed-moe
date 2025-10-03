#include "../hipCommon.h"
#include "schedulerPrototype.h"

struct Task {
  uint32_t _index;
  uint32_t _size;

  __device__ uint32_t GetIndex() const { return _index; }

  __device__ uint32_t GetSize() const { return _size; }
};

struct Manager {
  Task* _tasksAlloc;
  uint32_t _tasksSize;

  __device__ Task* GetTask(uint32_t index) const { return _tasksAlloc + index; }

  __device__ uint32_t GetSize() const { return _tasksSize; }
};

__constant__ Manager globalTaskManager;

__global__ void runSchedulerGpu(uint32_t tasksSize) {
  if (threadIdx.x == 0) {
    Task* task = globalTaskManager.GetTask(blockIdx.x);
    Task rTask;
    rTask._index = task->GetIndex();
    rTask._size = task->GetSize();

    printf("BlockIdx: %i, task index: %i, task size: %i\n", blockIdx.x,
           rTask.GetIndex(), rTask.GetSize());
  }
}

void runScheduler() {
  const uint32_t TASKS_SIZE = 124;
  Task* tasksHost = new Task[TASKS_SIZE];
  for (unsigned i = 0; i < TASKS_SIZE; i++) {
    tasksHost[i]._index = i;
    tasksHost[i]._size = i*2;
  }

  Task* tasksDevice;
  HIP_CHECK(hipMalloc(&tasksDevice, TASKS_SIZE*sizeof(Task)));
  HIP_CHECK(hipMemcpy(tasksDevice, tasksHost, TASKS_SIZE * sizeof(Task),
                      hipMemcpyHostToDevice));

  Manager managerHost;

  managerHost._tasksAlloc = tasksDevice;
  managerHost._tasksSize = TASKS_SIZE;


  HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(globalTaskManager), &managerHost, sizeof(Manager), 0, hipMemcpyHostToDevice));

  delete[] tasksHost;

  runSchedulerGpu<<<dim3(TASKS_SIZE, 1, 1), dim3(128, 1, 1), 0, 0>>>(TASKS_SIZE);
}