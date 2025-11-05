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

hipError_t taskSystemGemm(const float* A, const float* B, float* C, int M,
                          int N, int K) {
  constexpr int m_size = 2048;
  constexpr int n_size = 1536;
  const int k_size = 2048;
  assert(K == 2048);
  assert(M == m_size);
  assert(N == n_size);

  hipStream_t stream = 0;
  TaskManagerType managerHost;
  const int expectedNumTasks = (m_size / TILE_M) * (n_size / TILE_N);
  HIP_ERROR_CHECK(managerHost.Init(stream, expectedNumTasks));

  HIP_ERROR_CHECK(hipMemcpyToSymbolAsync(HIP_SYMBOL(globalTaskManager),
                                         &managerHost, sizeof(TaskManagerType),
                                         0, hipMemcpyHostToDevice, stream));

  int blocksPerKernel;
  HIP_ERROR_CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(
      &blocksPerKernel, taskSystemGemmKernel<m_size, n_size>, THREADS, 0));

  // printf("blocksPerKernel: %i\n", blocksPerKernel);

  dim3 blockDim(THREADS, 1, 1);
  dim3 gridDim(MI_300X_SM_SIZE * blocksPerKernel, 1, 1);

  taskSystemGemmKernel<m_size, n_size><<<gridDim, blockDim, 0, 0>>>(A, B, C);

  HIP_ERROR_CHECK(managerHost.Deinit(stream));
  return hipSuccess;
}