#include <hip/hip_runtime_api.h>

#include <iostream>
#include <vector>

#include "../../../src/hip/hipCommon.h"
#include "../../../src/hip/taskSystem/taskManager.h"
#include "gtest/gtest.h"

#define HIP_ASSERT(condition)                           \
  {                                                     \
    const hipError_t error = condition;                 \
    ASSERT_EQ(error, hipSuccess) << " for " #condition; \
  }

// TODO: align Task struct to cache line size to avoid cache trashing.
struct __align__(16) Task {
  int32_t stage;
  int32_t* data;
};

__device__ void ExecuteTask(const Task& task) {
  switch (task.stage) {
    case 0: {
      // First stage: increment data by 1

      const int32_t val = task.data[threadIdx.x];
      task.data[threadIdx.x] = 2 * val;
      break;
    }
    case 1: {
      // Second stage: increment data by 1
      const int32_t val = task.data[threadIdx.x];
      task.data[threadIdx.x] = val + 1;
      break;
    }
    default:
      // Unknown stage
      printf("Unknown task stage: %d\n", task.stage);
      break;
  }
}

using TaskManagerType = TaskManager<Task, 2048>;
__constant__ TaskManagerType globalTaskManager;

__global__ void testWorkItemQueue(int32_t* data, int32_t dataSize) {
  __shared__ Task s_task;
  __shared__ bool globalInterrupted;
  if (threadIdx.x == 0) {
    globalInterrupted = false;
  }

  // All threads produce tasks to execute
  for (int i = blockIdx.x * blockDim.x; i < dataSize;
       i += gridDim.x * blockDim.x) {
    Task rTask;
    rTask.stage = 0;
    rTask.data = data + i;
    globalTaskManager.PushTask_warp(&rTask);
  }

  __syncthreads();

  while (true) {
    if (threadIdx.x == 65 || threadIdx.x == 1) {
      printf(
          "Consumer BlockIdx: %i, threadIdx: %i: Before pop task loop\n",
          blockIdx.x, threadIdx.x);
    }
    // Get new task:
    const bool shouldInterrupt = globalTaskManager.PopTask_warp(&s_task);
    if (shouldInterrupt && threadIdx.x == 0) {
      globalInterrupted = true;
      printf("BlockIdx: %i, interrupted!\n", blockIdx.x);
    }

    // if (threadIdx.x == 65 || threadIdx.x == 1) {
    //   printf(
    //       "Consumer BlockIdx: %i, threadIdx: %i: After pop task loop, wating
    //       on barrier...\n", blockIdx.x, threadIdx.x);
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
    // if (threadIdx.x == 65) {
    //   printf(
    //       "Consumer BlockIdx: %i, threadIdx: %i, executing task->blockIdx: %i, "
    //       "task->ID: "
    //       "%i\n",
    //       blockIdx.x, threadIdx.x, taskblockIdx, taskId);
    // }
    Task rTask;
    rTask.stage = s_task.stage;
    rTask.data = s_task.data;
    ExecuteTask(rTask);

    // All threads create the same task.
    
    ++rTask.stage;
    rTask.data = s_task.data;

    if (rTask.stage < 2) {
      globalTaskManager.PushTask_warp(&rTask);
    }

    // if (threadIdx.x == 65 || threadIdx.x == 1) {
    //   printf(
    //       "Consumer BlockIdx: %i, threadIdx: %i, waiting for other
    //       threads..\n", blockIdx.x, threadIdx.x);
    // }
    __syncthreads();
  }
}

void runLinearProblemInplace(int32_t* dataGpu, size_t dataSize) {
  TaskManagerType managerHost;
  const int blocksNum = dataSize / 128;
  managerHost.Init(blocksNum * 2);

  HIP_ASSERT(hipMemcpyToSymbol(HIP_SYMBOL(globalTaskManager), &managerHost,
                               sizeof(TaskManagerType), 0,
                               hipMemcpyHostToDevice));

  dim3 blockDim(128, 1, 1);
  dim3 gridDim(blocksNum, 1, 1);

  testWorkItemQueue<<<gridDim, blockDim, 0, 0>>>(dataGpu, dataSize);

  HIP_ASSERT(hipDeviceSynchronize());
  managerHost.Deinit();
}

constexpr int error_exit_code = -1;

template <typename T>
static void PrintBuffer(const T* buffer, int sizeToPrint) {
  std::cout << "First " << sizeToPrint
            << " elements of the buffer: " << std::endl;
  for (int i = 0; i < sizeToPrint; ++i) {
    std::cout << buffer[i] << ", ";
  }

  std::cout << std::endl;
}

template <typename T>
static void CheckConstValBuffer(const T* buffer, int size, T value) {
  for (int i = 0; i < size; ++i) {
    ASSERT_EQ(buffer[i], value) << "for i: " << i;
  }
}

TEST(LinearProblem, SchedulerTest) {
  int devices;
  HIP_ASSERT(hipGetDeviceCount(&devices));
  std::cout << "GPUS: " << devices << std::endl;

  int deviceId;
  HIP_ASSERT(hipGetDevice(&deviceId));
  std::cout << "Using GPU with id: " << deviceId << std::endl;

  // Prepare data:

  const size_t dataSize = 200 * 128;

  std::vector<int32_t> dataCpu(dataSize);

  for (size_t i = 0; i < dataSize; ++i) {
    dataCpu[i] = i;
  }

  int32_t* dataGpu = nullptr;
  HIP_ASSERT(hipMalloc(&dataGpu, dataSize * sizeof(int32_t)));
  HIP_ASSERT(hipMemcpy(dataGpu, dataCpu.data(), dataSize * sizeof(int32_t),
                       hipMemcpyHostToDevice));

  runLinearProblemInplace(dataGpu, dataSize);

  HIP_ASSERT(hipDeviceSynchronize());

  HIP_ASSERT(hipMemcpy(dataCpu.data(), dataGpu, dataSize * sizeof(int32_t),
                       hipMemcpyDeviceToHost));

  HIP_ASSERT(hipGetLastError());

  // Check results:
  for (size_t i = 0; i < dataSize; ++i) {
    ASSERT_EQ(dataCpu[i], 2 * i + 1) << " for i: " << i;
  }

  
}