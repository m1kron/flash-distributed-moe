#include <hip/hip_runtime.h>

#include "src/hip/tasks/internal/gemmTileBlock.h"
#include "src/hip/tasks/internal/softmaxBlock.h"
#include "src/hip/tasks/internal/topkBlock.h"
#include "test/cpp/moe/reference/refMoeGate.h"
#include "test/cpp/utils/utils.h"

namespace {
constexpr int M_SIZE = 48;
constexpr int N_SIZE = 128;
constexpr int K_SIZE = 2048;
constexpr int TOPK = 8;
constexpr int THREADS = 128;
constexpr int BLOCKS = 304;

template <typename TGEMM_TILE>
__global__ void refHipKernel(const float* __restrict__ A,
                             const float* __restrict__ B,
                             float* __restrict__ topkVals,
                             int* __restrict__ topIdx, int M_SIZE) {
  __shared__ char sharedMemPool[TGEMM_TILE::SHARED_MEM_NEEDES_BYTES];

  for (int tokenIdx = blockIdx.x; tokenIdx < M_SIZE; tokenIdx += gridDim.x) {
    typename TGEMM_TILE::TType out_regs[TGEMM_TILE::THREAD_OUTPUT_SIZE];

    moe::tasks::internal::GemmTile_block<TGEMM_TILE>(A, B, out_regs, tokenIdx,
                                                     0, sharedMemPool);
    out_regs[0] =
        moe::tasks::internal::Softmax_block<TGEMM_TILE::THREADS, float>(
            out_regs[0], sharedMemPool);

    char* sharedMemPoolBytes = reinterpret_cast<char*>(sharedMemPool);
    float* s_topkVals = reinterpret_cast<float*>(sharedMemPoolBytes);
    int* s_topkIdx =
        reinterpret_cast<int*>(sharedMemPoolBytes + sizeof(float) * TOPK);
    int* nextPool = s_topkIdx + TOPK;

    moe::tasks::internal::Topk8_block<TGEMM_TILE::THREADS, float>(
        out_regs[0], s_topkVals, s_topkIdx, nextPool);

    // Write to output:
    const int tid = threadIdx.x;
    if (threadIdx.x < TOPK) {
      topkVals[tokenIdx * 8 + tid] = s_topkVals[tid];
      topIdx[tokenIdx * 8 + tid] = s_topkIdx[tid];
    }
  }
}

hipError_t refHipGemm(const float* A, const float* B, float* topkVals,
                      int* topkIdx, int M, int N, int K) {
  constexpr int m_tile_size = 1;
  constexpr int n_tile_size = N_SIZE;
  assert(K == K_SIZE);
  assert(M == M_SIZE);
  assert(N == N_SIZE);

  using TGemmTileParams =
      GemmTileParams<N_SIZE, K_SIZE, m_tile_size, n_tile_size, 32, THREADS>;

  dim3 grid(BLOCKS);
  dim3 block(TGemmTileParams::THREADS);

  refHipKernel<TGemmTileParams>
      <<<grid, block, 0, 0>>>(A, B, topkVals, topkIdx, M);

  return hipSuccess;
}

TEST(MoeTasksInternalTests, MoeGate_basicTest) {
  constexpr int a_size = M_SIZE * K_SIZE;
  constexpr int b_size = K_SIZE * N_SIZE;
  constexpr int c_size = M_SIZE * N_SIZE;
  constexpr int topk_size = M_SIZE * TOPK;

  std::vector<float> host_a = RandomVector<float>(a_size, -0.1f, 0.1f);
  std::vector<float> host_b = RandomVector<float>(b_size, -0.1f, 0.1f);
  std::vector<float> host_topkVals(topk_size);
  std::vector<int> host_topkIdx(topk_size);

  float* d_a = nullptr;
  float* d_b = nullptr;
  float* d_topkVals = nullptr;
  int* d_topkIdx = nullptr;

  HIP_ERROR_ASSERT(hipMalloc(&d_a, a_size * sizeof(float)));
  HIP_ERROR_ASSERT(hipMalloc(&d_b, b_size * sizeof(float)));
  HIP_ERROR_ASSERT(hipMalloc(&d_topkVals, topk_size * sizeof(float)));
  HIP_ERROR_ASSERT(hipMalloc(&d_topkIdx, topk_size * sizeof(int)));

  HIP_ERROR_ASSERT(hipMemcpy(d_a, host_a.data(), a_size * sizeof(float),
                             hipMemcpyHostToDevice));
  HIP_ERROR_ASSERT(hipMemcpy(d_b, host_b.data(), b_size * sizeof(float),
                             hipMemcpyHostToDevice));
  HIP_ERROR_ASSERT(
      refHipGemm(d_a, d_b, d_topkVals, d_topkIdx, M_SIZE, N_SIZE, K_SIZE));

  HIP_ERROR_ASSERT(hipDeviceSynchronize())
  HIP_ERROR_ASSERT(hipGetLastError());

  HIP_ERROR_ASSERT(hipMemcpy(host_topkVals.data(), d_topkVals,
                             topk_size * sizeof(float), hipMemcpyDeviceToHost));
  HIP_ERROR_ASSERT(hipMemcpy(host_topkIdx.data(), d_topkIdx,
                             topk_size * sizeof(int), hipMemcpyDeviceToHost));

  // Free device memory.
  HIP_ERROR_ASSERT(hipFree(d_a));
  HIP_ERROR_ASSERT(hipFree(d_b));
  HIP_ERROR_ASSERT(hipFree(d_topkVals));
  HIP_ERROR_ASSERT(hipFree(d_topkIdx));

  auto refOut = test::refMoeGate(host_a.data(), host_b.data(), M_SIZE, N_SIZE,
                                 K_SIZE, TOPK);

  CheckAgainstRefBuffer(host_topkVals, refOut.topkVals);
  CheckAgainstRefBuffer(host_topkIdx, refOut.topkIdx);
}
}  // namespace