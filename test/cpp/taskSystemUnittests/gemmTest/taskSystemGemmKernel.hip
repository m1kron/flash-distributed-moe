#include <hip/hip_runtime.h>

#include "../../../../src/hip/hipHostCommon.h"
#include "../../../../src/hip/taskSystem/taskManager.h"
#include "../../../../src/hip/taskSystem/workerLoop.h"
#include "../../../../src/hip/utils/hipDeviceUtils.h"
#include "taskSystemGemmKernel.h"
#include "tileGemm.h"

#define MI_300X_SM_SIZE 304

struct __align__(16) GemmTileTask {
  const float* A;
  const float* B;
  float* C;
  int blockTileRowStartIdx;
  int blockTileColStartIdx;
};

template <typename TILE>
__device__ void ExecuteTask(const GemmTileTask& task, void* sharedMemPool) {
  TILE::gemm_64x64x2048_single_block(task.A, task.B, task.C,
                                     task.blockTileRowStartIdx,
                                     task.blockTileColStartIdx, sharedMemPool);
}

using TaskManagerType = TaskManager<GemmTileTask, 8192>;
__constant__ TaskManagerType globalTaskManager;

template <int M, int N>
__global__ void taskSystemGemmKernel(const float* __restrict__ A,
                                     const float* __restrict__ B,
                                     float* __restrict__ C) {
  using TestTile = TestTileGemm<N>;
  __shared__ char sharedMemPool[TestTile::SHARED_MEM_NEEDES_BYTES];

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

  workerTaskSystemLoop_block(globalTaskManager, ExecuteTask<TestTile>,
                             sharedMemPool);
}

struct TaskSystemGemmParams {
  TaskManagerType managerHost;
  int blocksPerKernel;
  dim3 blockDim;
  dim3 gridDim;
  hipStream_t stream = 0;
};

hipError_t initTaskSystemGemm(void** state) {
  constexpr int M_SIZE = 2048;
  constexpr int N_SIZE = 1536;
  assert(M == 2048);
  assert(N == 1536);
  TaskSystemGemmParams* params = new TaskSystemGemmParams();
  HIP_ERROR_CHECK(params->managerHost.Init(params->stream));

  HIP_ERROR_CHECK(hipMemcpyToSymbolAsync(
      HIP_SYMBOL(globalTaskManager), &params->managerHost,
      sizeof(TaskManagerType), 0, hipMemcpyHostToDevice, params->stream));

  HIP_ERROR_CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(
      &params->blocksPerKernel, taskSystemGemmKernel<M_SIZE, N_SIZE>, THREADS,
      0));

  params->blockDim = dim3(THREADS, 1, 1);
  params->gridDim = dim3(MI_300X_SM_SIZE * params->blocksPerKernel, 1, 1);

  *state = params;

  return hipSuccess;
}

hipError_t deinitTaskSystemGemm(void* state) {
  TaskSystemGemmParams* params = static_cast<TaskSystemGemmParams*>(state);
  HIP_ERROR_CHECK(params->managerHost.Deinit(params->stream));
  delete params;
  return hipSuccess;
}

hipError_t taskSystemGemm(void* state, const float* A, const float* B, float* C,
                          int M, int N, int K) {
  TaskSystemGemmParams* params = static_cast<TaskSystemGemmParams*>(state);
  constexpr int M_SIZE = 2048;
  constexpr int N_SIZE = 1536;
  const int k_size = 2048;
  assert(K == 2048);
  assert(M == M_SIZE);
  assert(N == N_SIZE);

  const int expectedNumTasks = (M / TILE_M) * (N / TILE_N);
  HIP_ERROR_CHECK(params->managerHost.PrepareForNextLaunch(params->stream,
                                                           expectedNumTasks));

  taskSystemGemmKernel<M_SIZE, N_SIZE>
      <<<params->gridDim, params->blockDim, 0, params->stream>>>(A, B, C);

  return hipSuccess;
}