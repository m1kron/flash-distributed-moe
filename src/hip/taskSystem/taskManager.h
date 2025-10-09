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

  __host__ void Init() {
    _workQueue.Init();
    _tasksAlloc.Init();
  }

  __host__ void Deinit() {
    _workQueue.Deinit();
    _tasksAlloc.Deinit();
  }

  __device__ TTask* AllocateTask() { return _tasksAlloc.Allocate(); }

  __device__ void FreeTask(TTask* task) { _tasksAlloc.Free(task); }

  __device__ uint32_t GetSize() const { return _tasksAlloc.GetSize(); }

  __device__ void PushTask_warp(TTask* task) {
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
    if (threadIdx.x / warpSize == 0) {
      for (int i = threadIdx.x; i < sizeof(TTask); i += warpSize) {
        taskBlob_g[i] = taskBlob_r[i];
      }

      __threadfence();  //< Mem for writing warp only
    }
    // Task is ready in global mem.

    // Thread 0 pushes task to the queue:
    if (threadIdx.x == 0) {
      _workQueue.Push(task_globalMem);
      printf("Producer BlockIdx: %i, added task ID: %i\n", blockIdx.x,
             task->taskId);
    }
  }

  __device__ void PopTask_block(TTask* task_sharedMem) {
    // Get new task:
    TTask* task_globalMem = nullptr;
    if (threadIdx.x == 0) {
      task_globalMem = _workQueue.PopOrBusyWait();
      printf("Consumer BlockIdx: %i, got new task!\n", blockIdx.x);
    }
    unsigned long long poppedTaskInt =
        reinterpret_cast<unsigned long long>(task_globalMem);

    // Broadcast pointer to all threads in the warp.
    poppedTaskInt = __shfl(poppedTaskInt, 0);

    task_globalMem = reinterpret_cast<TTask*>(poppedTaskInt);

    char* taskBlob_s = reinterpret_cast<char*>(task_sharedMem);
    const char* taskBlob_g = reinterpret_cast<char*>(task_globalMem);

    // Warp 0 cooperatvely copy task from global mem to shared mem:
    if (threadIdx.x / warpSize == 0) {
      for (int i = threadIdx.x; i < sizeof(TTask); i += warpSize) {
        taskBlob_s[i] = taskBlob_g[i];
      }
    }

    if (threadIdx.x == 0) {
      // Free task in global mem:
      FreeTask(task_globalMem);
    }

    __syncthreads();
  }
};