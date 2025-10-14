#include <vector>

#include "../../../src/hip/hipHostCommon.h"
#include "../../../src/hip/taskSystem/taskManager.h"
#include "../../../src/hip/utils/hipDeviceUtils.h"
#include "gtest/gtest.h"

#define THREAD_BLOCK_SIZE 128
#define MIN_WARPS_PER_EXECUTION_UNIT 4

#define HIP_ERROR_ASSERT(condition)                     \
  {                                                     \
    const hipError_t error = condition;                 \
    ASSERT_EQ(error, hipSuccess) << " for " #condition; \
  }

struct __align__(16) SaxpyStageTask {
  int32_t stage;
  int32_t* data;
};

__device__ void ExecuteSaxpyStageTask(const SaxpyStageTask& task) {
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

using TaskManagerType = TaskManager<SaxpyStageTask, 8192>;
__constant__ TaskManagerType globalTaskManager;

__global__ void __launch_bounds__(THREAD_BLOCK_SIZE,
                                  MIN_WARPS_PER_EXECUTION_UNIT)
    inplaceSaxpyUsingTasksKernel(int32_t* data, int32_t dataSize) {
  __shared__ SaxpyStageTask task_s;
  __shared__ bool shouldExitNow_s;
  if (threadIdx.x == 0) {
    shouldExitNow_s = false;
  }

  // All threads produce initial tasks to execute
  for (int i = blockIdx.x * blockDim.x; i < dataSize;
       i += gridDim.x * blockDim.x) {
    SaxpyStageTask rTask;
    rTask.stage = 0;
    rTask.data = data + i;
    globalTaskManager.PushTask_warp(&rTask);
  }

  __syncthreads();

  while (true) {
    if (threadIdx.x == 65 || threadIdx.x == 1) {
      HIP_DEVICE_LOG("Consumer: Before pop task loop\n");
    }
    // Get new task:
    const bool noMoreTasks = globalTaskManager.PopTask_warp(&task_s);
    if (noMoreTasks && threadIdx.x == 0) {
      shouldExitNow_s = true;
    }

    __syncthreads();
    if (shouldExitNow_s) {
      break;
    }

    // Block cooperatively execute task:
    SaxpyStageTask rTask;
    rTask.stage = task_s.stage;
    rTask.data = task_s.data;
    ExecuteSaxpyStageTask(rTask);

    // And submit new one if needed:
    ++rTask.stage;
    rTask.data = task_s.data;

    if (rTask.stage < 2) {
      globalTaskManager.PushTask_warp(&rTask);
    }
  }
}

void runInplaceSaxpy(int32_t* dataGpu, size_t dataSize) {
  TaskManagerType managerHost;
  const int blocksNum = dataSize / THREAD_BLOCK_SIZE;
  managerHost.Init(blocksNum * 2);

  HIP_ERROR_ASSERT(hipMemcpyToSymbol(HIP_SYMBOL(globalTaskManager),
                                     &managerHost, sizeof(TaskManagerType), 0,
                                     hipMemcpyHostToDevice));

  dim3 blockDim(THREAD_BLOCK_SIZE, 1, 1);
  dim3 gridDim(608, 1, 1);

  inplaceSaxpyUsingTasksKernel<<<gridDim, blockDim, 0, 0>>>(dataGpu, dataSize);

  HIP_ERROR_ASSERT(hipDeviceSynchronize());
  managerHost.Deinit();
}

TEST(TaskSystem, SaxpyCoherenceTest) {
  // Prepare data:
  const size_t dataSize = 2048 * THREAD_BLOCK_SIZE;
  std::vector<int32_t> dataCpu(dataSize);

  for (size_t i = 0; i < dataSize; ++i) {
    dataCpu[i] = i;
  }

  int32_t* dataGpu = nullptr;
  HIP_ERROR_ASSERT(hipMalloc(&dataGpu, dataSize * sizeof(int32_t)));
  HIP_ERROR_ASSERT(hipMemcpy(dataGpu, dataCpu.data(),
                             dataSize * sizeof(int32_t),
                             hipMemcpyHostToDevice));

  runInplaceSaxpy(dataGpu, dataSize);

  HIP_ERROR_ASSERT(hipDeviceSynchronize());

  HIP_ERROR_ASSERT(hipMemcpy(dataCpu.data(), dataGpu,
                             dataSize * sizeof(int32_t),
                             hipMemcpyDeviceToHost));

  HIP_ERROR_ASSERT(hipGetLastError());

  // Check results:
  for (size_t i = 0; i < dataSize; ++i) {
    ASSERT_EQ(dataCpu[i], 2 * i + 1) << " for i: " << i;
  }
}