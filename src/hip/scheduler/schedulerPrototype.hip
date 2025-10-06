#include "../hipCommon.h"
#include "schedulerPrototype.h"

// Note: For now I will try approach with single global work queue for all
// workers. This might be probelmatic due to contention, but it is implemented
// with atomic load and add only(assuminghw support here...) If the contention
// will be a problem, I will switch to backup solution, but that depends on
// workload which will be determined in the future.

// This queue is unusual, since the assumption is that
// all items slots are preallocated and there is enough of them, so there is no
// need to reuse item memory again.
template <typename T, unsigned int SIZE>
struct WorkQueue {
  T* m_workQueue;
  unsigned int* m_headIdx;
  unsigned int* m_tailIdx;

  __host__ void Init() {
    HIP_CHECK(hipMalloc(&m_workQueue, SIZE * sizeof(T)));
    HIP_CHECK(hipMalloc(&m_headIdx, sizeof(int*)));
    HIP_CHECK(hipMalloc(&m_tailIdx, sizeof(int*)));

    HIP_CHECK(hipMemset(m_workQueue, 0, SIZE * sizeof(T)));

    const int zero = 0;

    HIP_CHECK(hipMemcpy(m_headIdx, &zero, sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(m_tailIdx, &zero, sizeof(int), hipMemcpyHostToDevice));
  }

  __host__ void Deinit() {
    HIP_CHECK(hipFree(m_tailIdx));
    HIP_CHECK(hipFree(m_headIdx));
    HIP_CHECK(hipFree(m_workQueue));
  }

  __device__ unsigned int AddNewWorkItem() {
    unsigned int prevHead = atomicAdd(m_headIdx, 1);

    if (prevHead >= SIZE) {
      assert(false);
    }

    return prevHead;
  }

  __device__ unsigned int PopWorkItemOrWait(unsigned int maxTries) {
    unsigned int prevTail = atomicAdd(m_tailIdx, 1);
    unsigned int head = __atomic_load_n(m_headIdx, __ATOMIC_RELAXED);
    unsigned int tries = 0;

    while (head <= prevTail && head < SIZE) {
      head = __atomic_load_n(m_headIdx, __ATOMIC_RELAXED);
      ++tries;
    }

    return prevTail;
  }

  __host__ __device__ T* GetWorkItemAtIdx(const unsigned int idx) {
    if (idx >= SIZE) {
      assert(false);
      return nullptr;
    }

    return &m_workQueue[idx];
  }

  __host__ __device__ unsigned int GetWorkItemSize() const { return SIZE; }
};

// TODO: align Task struct to cache line size to avoid cash trashing.
struct Task {
  uint32_t _index;
  uint32_t _size;
  uint32_t _blockIdx;


  __device__ uint32_t GetIndex() const { return _index; }

  __device__ uint32_t GetSize() const { return _size; }
};

struct Manager {
  Task* _tasksAlloc;
  uint32_t _tasksSize;
  WorkQueue<Task, 1024> _workQueue;

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

__global__ void testWorkItemQueue() {
  if (threadIdx.x == 0) {
    const unsigned int idx = globalTaskManager._workQueue.AddNewWorkItem();

    // WARNING! Race condition here: atomic add + work item fill...

    Task* task = globalTaskManager._workQueue.GetWorkItemAtIdx(idx);
    task->_index = idx;
    task->_blockIdx = blockIdx.x;

    // TODO: Move this to queue impl(but how?)
    __threadfence();

    printf("BlockIdx: %i, added task idx: %i\n", blockIdx.x, idx);
  }

  __syncthreads();

  if (threadIdx.x == 100) {
    const unsigned int idx = globalTaskManager._workQueue.PopWorkItemOrWait(1000);

    printf("BlockIdx: %i, reading task idx: %i\n", blockIdx.x, idx);

    Task* task = globalTaskManager._workQueue.GetWorkItemAtIdx(idx);
    const auto taskblockIdx = task->_blockIdx;
    const auto index = task->_index;

    printf("BlockIdx: %i, task idx: %i, task->_blockIdx: %i, task->_index: %i\n", blockIdx.x, idx, taskblockIdx, index);
  }

  
}

void runScheduler() {
  const uint32_t TASKS_SIZE = 124;
  Task* tasksHost = new Task[TASKS_SIZE];
  for (unsigned i = 0; i < TASKS_SIZE; i++) {
    tasksHost[i]._index = i;
    tasksHost[i]._size = i * 2;
  }

  Task* tasksDevice;
  HIP_CHECK(hipMalloc(&tasksDevice, TASKS_SIZE * sizeof(Task)));
  HIP_CHECK(hipMemcpy(tasksDevice, tasksHost, TASKS_SIZE * sizeof(Task),
                      hipMemcpyHostToDevice));

  Manager managerHost;
  managerHost._workQueue.Init();

  managerHost._tasksAlloc = tasksDevice;
  managerHost._tasksSize = TASKS_SIZE;

  HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(globalTaskManager), &managerHost,
                              sizeof(Manager), 0, hipMemcpyHostToDevice));

  delete[] tasksHost;

  // runSchedulerGpu<<<dim3(TASKS_SIZE, 1, 1), dim3(128, 1, 1), 0, 0>>>(
  //     TASKS_SIZE);

  testWorkItemQueue<<<dim3(TASKS_SIZE, 1, 1), dim3(128, 1, 1), 0, 0>>>();


  HIP_CHECK(hipDeviceSynchronize())
  managerHost._workQueue.Deinit();

}