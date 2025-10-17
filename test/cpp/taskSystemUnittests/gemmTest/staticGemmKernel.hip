#include <hip/hip_runtime.h>

#include "staticGemmKernel.h"
#include "tileGemm.h"

template <int M, int N>
__global__ void gemmStaticKernel(const float* __restrict__ A,
                                 const float* __restrict__ B,
                                 float* __restrict__ C) {
  gemm_64x64x2048_single_block<M, N>(A, B, C, blockIdx.x, blockIdx.y);
}

hipError_t staticGemm(const float* A, const float* B, float* C, int M, int N,
                      int K) {
  constexpr int m_size = 2048;
  constexpr int n_size = 1536;
  const int k_size = 2048;
  const int m_tile_size = 64;
  const int n_tile_size = 64;
  assert(K == 2048);
  assert(M == m_size);
  assert(N == n_size);

  dim3 grid(M / m_tile_size, N / n_tile_size, 1);
  dim3 block(256);

  gemmStaticKernel<m_size, n_size><<<grid, block, 0, 0>>>(A, B, C);

  return hipSuccess;
}