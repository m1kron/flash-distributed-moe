#pragma once
#include "../utils/hipDeviceUtils.h"
#include "taskAllocator.h"
#include "workQueue.h"

// Note: For now I will try approach with single global work queue for all
// workers. This might be probelmatic due to contention, but it is implemented
// with atomic load and add only(assuminghw support here...) If the contention
// will be a problem, I will switch to backup solution, but that depends on
// workload which will be determined in the future.

template <typename TTask, uint32_t SIZE>
struct TaskManager {
  using TTaskType = TTask;

  // Initializes task manager on host.
  __host__ hipError_t Init(hipStream_t stream);

  // Deinitializes task manager on host.
  __host__ hipError_t Deinit(hipStream_t stream);

  // Prepares for next launch(clears the state).
  __host__ hipError_t PrepareForNextLaunch(hipStream_t stream,
                                           uint32_t expectedMaxTasks);

  // Returns queue size.
  __host__ uint32_t GetTaskQueueSize() const;

  // Gets the current number of executed tasks.
  __device__ uint32_t GetCurrentExecutedTasks() const;

  // Increments the current number of executed tasks.
  __device__ void IncrementCurrentExecutedTasks();

  // Checks if the current number of executed tasks reached the max.
  __device__ bool DidExecuteExpectedNumberOfTasks() const;

  // Allocates new task in device code, returns nullptr if out of memory.
  __device__ TTask* AllocateTask();

  // Frees task in device code.
  __device__ void FreeTask(TTask* task);

  // Pushes task to the queue in device code, executed by warp 0 of the block.
  __device__ void PushTask_warp(TTask* task);

  // Waits and pops new task from the queue in device code, executed by warp 0
  // of the block.
  // If there is a new task ready, warp loads it into task_sharedMem.
  // If reached excpected max of executed tasks, waiting loop is break and code
  // returns true. Otherwise return false, and then task_sharedMem contains
  // poped task.
  __device__ bool WaitAndPopTask_warp(TTask* task_sharedMem);

  TaskAllocator<TTask, SIZE> m_tasksAlloc;
  WorkQueue<TTask, SIZE> m_workQueue;
  uint32_t* m_expectedMaxTasks;
  uint32_t* m_currentExecutedTasks;
};

/////////////////////////////////////////////////////////////////
//
// IMPLEMENTATION:
//
/////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////
template <typename TTask, uint32_t SIZE>
inline __host__ hipError_t TaskManager<TTask, SIZE>::Init(hipStream_t stream) {
  // TODO: allocated all neeeded mem with single call.
  HIP_ERROR_CHECK(m_workQueue.Init(stream));
  HIP_ERROR_CHECK(m_tasksAlloc.Init(stream));
  HIP_ERROR_CHECK(
      hipMallocAsync(&m_expectedMaxTasks, sizeof(uint32_t), stream));
  HIP_ERROR_CHECK(
      hipMallocAsync(&m_currentExecutedTasks, sizeof(uint32_t), stream));
  return hipSuccess;
}

/////////////////////////////////////////////////////////////////
template <typename TTask, uint32_t SIZE>
inline __host__ hipError_t
TaskManager<TTask, SIZE>::Deinit(hipStream_t stream) {
  HIP_ERROR_CHECK(m_workQueue.Deinit(stream));
  HIP_ERROR_CHECK(m_tasksAlloc.Deinit(stream));
  HIP_ERROR_CHECK(hipFreeAsync(m_expectedMaxTasks, stream));
  HIP_ERROR_CHECK(hipFreeAsync(m_currentExecutedTasks, stream));
  return hipSuccess;
}

/////////////////////////////////////////////////////////////////
template <typename TTask, uint32_t SIZE>
inline __host__ hipError_t TaskManager<TTask, SIZE>::PrepareForNextLaunch(
    hipStream_t stream, uint32_t expectedMaxTasks) {
  if (expectedMaxTasks > GetTaskQueueSize()) {
    MOE_ERROR_LOG("Expected max tasks(%d) exceeds queue size(%d)\n",
                  expectedMaxTasks, SIZE);
    return hipErrorUnknown;
  }
  HIP_ERROR_CHECK(m_workQueue.PrepareForNextLaunch(stream));
  HIP_ERROR_CHECK(m_tasksAlloc.PrepareForNextLaunch(stream));
  HIP_ERROR_CHECK(hipMemcpyAsync(m_expectedMaxTasks, &expectedMaxTasks,
                                 sizeof(uint32_t), hipMemcpyHostToDevice,
                                 stream));
  HIP_ERROR_CHECK(
      hipMemsetAsync(m_currentExecutedTasks, 0, sizeof(uint32_t), stream));
  return hipSuccess;
}

////////////////////////////////////////////////////////////////
template <typename TTask, uint32_t SIZE>
inline __host__ uint32_t TaskManager<TTask, SIZE>::GetTaskQueueSize() const {
  return SIZE;
}

/////////////////////////////////////////////////////////////////
template <typename TTask, uint32_t SIZE>
inline __device__ uint32_t
TaskManager<TTask, SIZE>::GetCurrentExecutedTasks() const {
  return __atomic_load_n(m_currentExecutedTasks, __ATOMIC_RELAXED);
}

/////////////////////////////////////////////////////////////////
template <typename TTask, uint32_t SIZE>
inline __device__ void
TaskManager<TTask, SIZE>::IncrementCurrentExecutedTasks() {
  atomicAdd(m_currentExecutedTasks, 1);
}

/////////////////////////////////////////////////////////////////
template <typename TTask, uint32_t SIZE>
inline __device__ bool
TaskManager<TTask, SIZE>::DidExecuteExpectedNumberOfTasks() const {
  // read atomically and compare with max
  const uint32_t expectedMaxExecutedTasks =
      __atomic_load_n(m_expectedMaxTasks, __ATOMIC_RELAXED);
  return GetCurrentExecutedTasks() >= expectedMaxExecutedTasks;
}

/////////////////////////////////////////////////////////////////
template <typename TTask, uint32_t SIZE>
inline __device__ TTask* TaskManager<TTask, SIZE>::AllocateTask() {
  return m_tasksAlloc.Allocate();
}

/////////////////////////////////////////////////////////////////
template <typename TTask, uint32_t SIZE>
inline __device__ void TaskManager<TTask, SIZE>::FreeTask(TTask* task) {
  m_tasksAlloc.Free(task);
}

/////////////////////////////////////////////////////////////////
template <typename TTask, uint32_t SIZE>
inline __device__ void TaskManager<TTask, SIZE>::PushTask_warp(TTask* task) {
  if (threadIdx.x / warpSize == 0) {
    TTask* task_globalMem = nullptr;

    if (threadIdx.x == 0) {
      // Thread 0 allocates task in global mem:
      task_globalMem = AllocateTask();
      HIP_DEVICE_ASSERT(task_globalMem != nullptr);
    }

    // Warp 0 cooperatvely fill task in global mem:
    unsigned long long allocatedTaskInt =
        reinterpret_cast<unsigned long long>(task_globalMem);

    // Broadcast pointer to all threads in the warp.
    allocatedTaskInt = __shfl(allocatedTaskInt, 0);

    task_globalMem = reinterpret_cast<TTask*>(allocatedTaskInt);

    char* taskBlob_g = reinterpret_cast<char*>(task_globalMem);
    const char* taskBlob_r = reinterpret_cast<const char*>(task);
    for (int i = threadIdx.x; i < sizeof(TTask); i += warpSize) {
      taskBlob_g[i] = taskBlob_r[i];
    }

    __threadfence();  //< Mem for writing warp only

    // Task is ready in global mem.

    // Thread 0 pushes task to the queue:
    if (threadIdx.x == 0) {
      m_workQueue.Push(task_globalMem);
      HIP_DEVICE_LOG("Added task to the queue\n");
    }
  }
}

/////////////////////////////////////////////////////////////////
template <typename TTask, uint32_t SIZE>
inline __device__ bool TaskManager<TTask, SIZE>::WaitAndPopTask_warp(
    TTask* task_sharedMem) {
  // Get new task:
  if (threadIdx.x / warpSize == 0) {
    TTask* task_globalMem = nullptr;

    bool shouldInterrupt = false;
    if (threadIdx.x == 0) {
      const auto ticket = m_workQueue.ReserveSlotTicket();

      if (ticket == UINT_MAX) {
        shouldInterrupt = true;
        HIP_DEVICE_LOG("Breaking waiting loop, ticket == UINT_MAX\n");
      }

      while (!m_workQueue.TryToPop(ticket, task_globalMem)) {
        if (DidExecuteExpectedNumberOfTasks()) {
          shouldInterrupt = true;
          HIP_DEVICE_LOG(
              "Breaking waiting loop, since max expected tasks reached\n");
          break;
        }
      }
    }

    shouldInterrupt = __shfl(shouldInterrupt, 0);
    if (shouldInterrupt) {
      return true;
    }

    unsigned long long poppedTaskInt =
        reinterpret_cast<unsigned long long>(task_globalMem);

    // Broadcast pointer to all threads in the warp.
    poppedTaskInt = __shfl(poppedTaskInt, 0);

    task_globalMem = reinterpret_cast<TTask*>(poppedTaskInt);

    char* taskBlob_s = reinterpret_cast<char*>(task_sharedMem);
    const char* taskBlob_g = reinterpret_cast<char*>(task_globalMem);

    // Warp 0 cooperatvely copy task from global mem to shared mem:
    for (int i = threadIdx.x; i < sizeof(TTask); i += warpSize) {
      taskBlob_s[i] = taskBlob_g[i];
    }

    if (threadIdx.x == 0) {
      // Free task in global mem:
      FreeTask(task_globalMem);
      IncrementCurrentExecutedTasks();
    }
    return false;
  }
  return false;
}