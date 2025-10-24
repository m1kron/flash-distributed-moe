#include <hip/hip_runtime.h>

#include "../../../src/hip/gemm/gemmTileBlock.h"
#include "../../../src/hip/softmax/softmaxTile.h"
#include "../reference/refGemm.h"
#include "../reference/refSoftmax.h"
#include "../utils/utils.h"

template <int M, int N>
__global__ void refHipKernel(const float* __restrict__ A,
                             const float* __restrict__ B,
                             float* __restrict__ C) {
  using TGemmTileParams = GemmTileParams<N, 2048, 1, 256, 32>;

  typename TGemmTileParams::TOutputType
      out_regs[TGemmTileParams::THREAD_OUTPUT_SIZE];

  GemmTileBlock<TGemmTileParams>(A, B, out_regs, blockIdx.x, blockIdx.y);
  out_regs[0] = softmax_row_256_warpopt(out_regs[0]);
  WriteGemmTileToGlobalMem<TGemmTileParams>(out_regs, C, blockIdx.x,
                                            blockIdx.y);
}

hipError_t refHipGemm(const float* A, const float* B, float* C, int M, int N,
                      int K) {
  constexpr int m_size = 48;
  constexpr int n_size = 256;
  const int k_size = 2048;
  const int m_tile_size = 1;
  const int n_tile_size = 256;
  assert(K == 2048);
  assert(M == m_size);
  assert(N == n_size);

  dim3 grid(M / m_tile_size, N / n_tile_size, 1);
  dim3 block(256);

  refHipKernel<m_size, n_size><<<grid, block, 0, 0>>>(A, B, C);

  return hipSuccess;
}

template <typename TFunc>
void PerformGemmCorrectnessTest2(const TFunc func) {
  // Prepare input buffers for module
  constexpr int m_size = 48;
  constexpr int n_size = 256;
  constexpr int k_size = 2048;

  constexpr int a_size = m_size * k_size;
  constexpr int b_size = k_size * n_size;
  constexpr int c_size = m_size * n_size;

  std::vector<float> host_a = RandomVector<float>(a_size, -0.1f, 0.1f);
  std::vector<float> host_b = RandomVector<float>(b_size, -0.1f, 0.1f);
  std::vector<float> host_c(c_size);

  float* d_a = nullptr;
  float* d_b = nullptr;
  float* d_c = nullptr;

  HIP_ERROR_ASSERT(hipMalloc(&d_a, a_size * sizeof(float)));
  HIP_ERROR_ASSERT(hipMalloc(&d_b, b_size * sizeof(float)));
  HIP_ERROR_ASSERT(hipMalloc(&d_c, c_size * sizeof(float)));
  HIP_ERROR_ASSERT(hipMemcpy(d_a, host_a.data(), a_size * sizeof(float),
                             hipMemcpyHostToDevice));
  HIP_ERROR_ASSERT(hipMemcpy(d_b, host_b.data(), b_size * sizeof(float),
                             hipMemcpyHostToDevice));
  HIP_ERROR_ASSERT(hipMemset(d_c, 0, c_size * sizeof(float)));

  HIP_ERROR_ASSERT(func(d_a, d_b, d_c, m_size, n_size, k_size));

  HIP_ERROR_ASSERT(hipDeviceSynchronize())
  HIP_ERROR_ASSERT(hipGetLastError());

  HIP_ERROR_ASSERT(hipMemcpy(host_c.data(), d_c, c_size * sizeof(float),
                             hipMemcpyDeviceToHost));

  // Free device memory.
  HIP_ERROR_ASSERT(hipFree(d_a));
  HIP_ERROR_ASSERT(hipFree(d_b));
  HIP_ERROR_ASSERT(hipFree(d_c));

  auto ref =
      test::refGemm(host_a.data(), host_b.data(), m_size, n_size, k_size);

  test::refSoftmaxRowOnlineInplace(ref.data(), m_size, n_size);

  CheckAgainstRefBuffer(host_c.data(), ref.data(), ref.size());
}

TEST(TaskSystemUnittests, refHipGemm) {
  PerformGemmCorrectnessTest2(refHipGemm);
}