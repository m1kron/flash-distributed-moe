#pragma once
#include "taskAllocator.h"
#include "workQueue.h"

// Note: For now I will try approach with single global work queue for all
// workers. This might be probelmatic due to contention, but it is implemented
// with atomic load and add only(assuminghw support here...) If the contention
// will be a problem, I will switch to backup solution, but that depends on
// workload which will be determined in the future.

template <typename TTask, unsigned int SIZE>
struct TaskManager {
  TaskAllocator<TTask, SIZE> _tasksAlloc;
  WorkQueue<TTask, SIZE> _workQueue;
  uint32_t* _maxTasks;
  uint32_t* _currentExecutedTasks;

  __host__ void Init(unsigned expectedMaxTasks) {
    _workQueue.Init();
    _tasksAlloc.Init();
    HIP_CHECK(hipMalloc(&_maxTasks, sizeof(uint32_t)));
    HIP_CHECK(hipMemcpy(_maxTasks, &expectedMaxTasks, sizeof(uint32_t),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMalloc(&_currentExecutedTasks, sizeof(uint32_t)));
    HIP_CHECK(hipMemset(_currentExecutedTasks, 0, sizeof(uint32_t)));
  }

  __host__ void Deinit() {
    _workQueue.Deinit();
    _tasksAlloc.Deinit();
    HIP_CHECK(hipFree(_maxTasks));
    HIP_CHECK(hipFree(_currentExecutedTasks));
  }

  __device__ uint32_t GetCurrentExecutedTasks() const {
    return __atomic_load_n(_currentExecutedTasks, __ATOMIC_RELAXED);
  }

  __device__ void IncrementCurrentExecutedTasks() {
    atomicAdd(_currentExecutedTasks, 1);
  }

  __device__ bool IsAtMaxExecuted() const {
    // read atomically and compare with max
    const uint32_t maxExecutedTasks =
        __atomic_load_n(_maxTasks, __ATOMIC_RELAXED);
    return GetCurrentExecutedTasks() >= maxExecutedTasks;
  }

  __device__ TTask* AllocateTask() { return _tasksAlloc.Allocate(); }

  __device__ void FreeTask(TTask* task) { _tasksAlloc.Free(task); }

  __device__ uint32_t GetSize() const { return _tasksAlloc.GetSize(); }

  __device__ void PushTask_warp(TTask* task) {
    if (threadIdx.x / warpSize == 0) {
      TTask* task_globalMem = nullptr;

      if (threadIdx.x == 0) {
        // Thread 0 allocates task in global mem:
        task_globalMem = AllocateTask();
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
        _workQueue.Push(task_globalMem);
        printf("Producer BlockIdx: %i, added task ID: %i\n", blockIdx.x,
               task->taskId);
      }
    }
  }

  __device__ bool PopTask_warp(TTask* task_sharedMem) {
    // Get new task:
    if (threadIdx.x / warpSize == 0) {
      TTask* task_globalMem = nullptr;

      bool shouldInterrupt = false;
      if (threadIdx.x == 0) {
        const auto ticket = _workQueue.ReserveSlotTicket();
        while (!_workQueue.TryToPop(ticket, task_globalMem)) {
          if (IsAtMaxExecuted()) {
            // We are at max executed tasks, return false - no task for now.
            shouldInterrupt = true;
            printf("Consumer BlockIdx: %i, thread0: interrupted!\n", blockIdx.x);
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
};