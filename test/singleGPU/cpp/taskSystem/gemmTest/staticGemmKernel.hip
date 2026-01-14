#include <hip/hip_runtime.h>

#include <cassert>

#include "staticGemmKernel.h"
#include "tileGemm.h"

template <int M, int N>
__global__ void gemmStaticKernel(const float* __restrict__ A,
                                 const float* __restrict__ B,
                                 float* __restrict__ C) {
  using TestTile = TestTileGemm<N>;
  __shared__ char sharedMemPool[TestTile::SHARED_MEM_NEEDES_BYTES];
  TestTile::gemm_64x64x2048_single_block(A, B, C, blockIdx.x, blockIdx.y,
                                         sharedMemPool);
}

hipError_t staticGemm(const float* A, const float* B, float* C, int M, int N,
                      int K) {
  constexpr int m_size = 2048;
  constexpr int n_size = 1536;
  constexpr int k_size = 2048;
  assert(K == k_size);
  assert(M == m_size);
  assert(N == n_size);

  dim3 grid(M / TILE_M, N / TILE_N, 1);
  dim3 block(THREADS);

  gemmStaticKernel<m_size, n_size><<<grid, block, 0, 0>>>(A, B, C);

  return hipSuccess;
}