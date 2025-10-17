#include <hip/hip_runtime.h>

#include "../../../../src/hip/hipHostCommon.h"
#include "../../../../src/hip/taskSystem/taskManager.h"
#include "../../../../src/hip/utils/hipDeviceUtils.h"
#include "taskSystemGemmKernel.h"
#include "tileGemm.h"

#define THREAD_BLOCK_SIZE 256
#define MIN_WARPS_PER_EXECUTION_UNIT 1

struct __align__(16) GemmTileTask {
  const float* A;
  const float* B;
  float* C;
  int blockTileRowStartIdx;
  int blockTileColStartIdx;
};

template <int M, int N>
__device__ void ExecuteTask(const GemmTileTask& task) {
  gemm_64x64x2048_single_block<M, N>(task.A, task.B, task.C,
                                     task.blockTileRowStartIdx,
                                     task.blockTileColStartIdx);
}

using TaskManagerType = TaskManager<GemmTileTask, 8192>;
__constant__ TaskManagerType globalTaskManager;

template <int M, int N>
__global__ void __launch_bounds__(THREAD_BLOCK_SIZE,
                                  MIN_WARPS_PER_EXECUTION_UNIT)
    taskSystemGemmKernel(const float* __restrict__ A,
                         const float* __restrict__ B, float* __restrict__ C) {
  constexpr int TILE_M = 64;
  constexpr int TILE_N = 64;

  __shared__ GemmTileTask task_s;
  __shared__ bool shouldExitNow_s;
  if (threadIdx.x == 0) {
    shouldExitNow_s = false;
  }

  // Warp0 adds new tasks:
  for (int i = blockIdx.x; i < (M / TILE_M) * (N / TILE_N); i += gridDim.x) {
    GemmTileTask rTask;
    rTask.A = A;
    rTask.B = B;
    rTask.C = C;
    rTask.blockTileRowStartIdx = i / (N / TILE_N);
    rTask.blockTileColStartIdx = i % (N / TILE_N);
    globalTaskManager.PushTask_warp(&rTask);
  }

  __syncthreads();

  while (true) {
    if (threadIdx.x == 65 || threadIdx.x == 1) {
      HIP_DEVICE_LOG("Consumer: Before pop task loop\n");
    }
    // Get new task:
    const bool noMoreTasks = globalTaskManager.WaitAndPopTask_warp(&task_s);
    if (noMoreTasks && threadIdx.x == 0) {
      shouldExitNow_s = true;
    }

    __syncthreads();
    if (shouldExitNow_s) {
      break;
    }

    // Block cooperatively execute task:
    GemmTileTask rTask;
    rTask.A = task_s.A;
    rTask.B = task_s.B;
    rTask.C = task_s.C;
    rTask.blockTileRowStartIdx = task_s.blockTileRowStartIdx;
    rTask.blockTileColStartIdx = task_s.blockTileColStartIdx;
    ExecuteTask<M, N>(rTask);
  }
}

hipError_t taskSystemGemm(const float* A, const float* B, float* C, int M,
                          int N, int K) {
  constexpr int m_size = 2048;
  constexpr int n_size = 1536;
  const int k_size = 2048;
  const int m_tile_size = 64;
  const int n_tile_size = 64;
  assert(K == 2048);
  assert(M == m_size);
  assert(N == n_size);

  hipStream_t stream = 0;
  TaskManagerType managerHost;
  const int expectedNumTasks = (m_size / m_tile_size) * (n_size / n_tile_size);
  HIP_ERROR_CHECK(managerHost.Init(stream, expectedNumTasks));

  HIP_ERROR_CHECK(hipMemcpyToSymbolAsync(HIP_SYMBOL(globalTaskManager),
                                         &managerHost, sizeof(TaskManagerType),
                                         0, hipMemcpyHostToDevice, stream));

  dim3 blockDim(THREAD_BLOCK_SIZE, 1, 1);
  dim3 gridDim(304, 1, 1);

  taskSystemGemmKernel<m_size, n_size><<<gridDim, blockDim, 0, 0>>>(A, B, C);

  HIP_ERROR_CHECK(managerHost.Deinit(stream));
  return hipSuccess;
}