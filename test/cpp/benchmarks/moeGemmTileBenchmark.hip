#include <hip/hip_runtime.h>

#include "src/hip/kernel/tasks/internal/gemmTileBlock.h"
#include "test/cpp/benchmarks/benchmarkUtils.h"
#include "test/cpp/moe/reference/refGemm.h"
#include "test/cpp/utils/utils.h"

namespace {
static constexpr float ABS_ERROR = 1e-3f;

template <typename TGEMM_TILE>
__global__ void refHipKernel(const float* __restrict__ tokens,
                             const float* __restrict__ weights, float* output) {
  __shared__ char sharedMemPool[TGEMM_TILE::SHARED_MEM_NEEDES_BYTES];

  constexpr int N_CHUNKS = TGEMM_TILE::N / TGEMM_TILE::TILE_N;

  const int tokenIdx = blockIdx.x / N_CHUNKS;

  const float* thisBlockTokenPtr = tokens + tokenIdx * TGEMM_TILE::K;

  const int thisBlockTileIdx = blockIdx.x % N_CHUNKS;

  typename TGEMM_TILE::TType out_regs[TGEMM_TILE::THREAD_OUTPUT_SIZE];

  moe::tasks::internal::GemmTile_block<TGEMM_TILE>(
      thisBlockTokenPtr, weights, out_regs, 0, thisBlockTileIdx, sharedMemPool);

  float* thisBlockOutputPtr = output + tokenIdx * TGEMM_TILE::N;
  moe::tasks::internal::WriteGemmTileToGlobalMem_block<TGEMM_TILE>(
      out_regs, thisBlockOutputPtr, 0, thisBlockTileIdx);
}

template <int M, int N, int K, int N_TILE, int K_TILE, int THREADS>
hipError_t gemmAsManyGemv(const float* A, const float* B, float* C) {
  // WARNING:
  // A [M,K]
  // B [N,K] (-> transposed from standard gemm)
  // C [M,N]
  // NOTE: We process gemm as many gemv, which simiulates moe case.
  constexpr int M_TILE = 1;

  using TGemmTileParams =
      GemmTileParams<N, K, M_TILE, N_TILE, K_TILE, THREADS, float>;

  constexpr int N_CHUNKS = N / N_TILE;
  constexpr int TILES_TO_PROCESS = N_CHUNKS * M;

  dim3 grid(TILES_TO_PROCESS);
  dim3 block(TGemmTileParams::THREADS);

  refHipKernel<TGemmTileParams><<<grid, block, 0, 0>>>(A, B, C);

  return hipSuccess;
}

template <int M, int N, int K, int N_TILE, int K_TILE, int THREADS>
void PerformBenchmark() {
  constexpr int a_size = M * K;
  constexpr int b_size = K * N;
  constexpr int c_size = M * N;

  std::vector<float> host_a = RandomVector<float>(a_size);
  std::vector<float> host_b = RandomVector<float>(b_size);
  std::vector<float> host_c = RandomVector<float>(c_size);

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

  const float avg_ms = test::bench::Benchmark(
      10, 10, gemmAsManyGemv<M, N, K, N_TILE, K_TILE, THREADS>, d_a, d_b, d_c);

  HIP_ERROR_ASSERT(
      gemmAsManyGemv<M, N, K, N_TILE, K_TILE, THREADS>(d_a, d_b, d_c));

  HIP_ERROR_ASSERT(hipDeviceSynchronize())
  HIP_ERROR_ASSERT(hipGetLastError());

  HIP_ERROR_ASSERT(hipMemcpy(host_c.data(), d_c, c_size * sizeof(float),
                             hipMemcpyDeviceToHost));

  // Free device memory.
  HIP_ERROR_ASSERT(hipFree(d_a));
  HIP_ERROR_ASSERT(hipFree(d_b));
  HIP_ERROR_ASSERT(hipFree(d_c));

  auto ref = test::refGemm(host_a.data(), host_b.data(), M, N, K);

  CheckAgainstRefBuffer(host_c, ref, ABS_ERROR);

  std::cout << "MOE-like gemm [" << M << "x" << N << "x" << K
            << "], with params: <THREADS: " << THREADS << ", N_TILE: " << N_TILE
            << ", K_TILE: " << K_TILE << "> kernel avg time is " << avg_ms
            << " ms." << std::endl;
}

#define DECLARE_BENCHMARK(M, N, K, N_TILE, K_TILE, THREADS)                             \
  TEST(                                                                                 \
      Benchmark,                                                                        \
      MoeLikeGemm_##M##x##N##K##_NTILE_##N_TILE##_KTILE_##K_TILE##_THREADS_##THREADS) { \
    PerformBenchmark<M, N, K, N_TILE, K_TILE, THREADS>();                               \
  }

DECLARE_BENCHMARK(128, 768, 2048, 256, 32, 128);
DECLARE_BENCHMARK(128, 2048, 768, 256, 32, 128);
}  // namespace