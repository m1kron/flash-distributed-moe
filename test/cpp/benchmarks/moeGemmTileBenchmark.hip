#include <hip/hip_runtime.h>

#include "src/hip/common/metadata.h"
#include "src/hip/kernel/tasks/internal/gemmImpl/gemmImplSelector.h"
#include "src/hip/kernel/tasks/internal/writeGemmTileBlock.h"
#include "test/cpp/benchmarks/benchmarkUtils.h"
#include "test/cpp/moe/reference/refGemm.h"
#include "test/cpp/utils/utils.h"

namespace {
static constexpr float ABS_ERROR = 1e-3f;

template <typename TGEMM_TILE>
__global__ void batchedGemvKernel(const float* __restrict__ A,
                                  const float* __restrict__ B,
                                  float* __restrict__ output) {
  using GemmImpl =
      typename moe::tasks::internal::GemmImplSelector<TGEMM_TILE>::type;

  __shared__ char sharedMemPool[GemmImpl::NeededSharedMemBytes()];

  constexpr int N_CHUNKS = TGEMM_TILE::N / TGEMM_TILE::TILE_N;

  const int thisBlockGemvIdx = blockIdx.x / N_CHUNKS;

  const float* thisBlockAPtr = A + thisBlockGemvIdx * TGEMM_TILE::K;
  const float* thisBlockBPtr =
      B + thisBlockGemvIdx * TGEMM_TILE::N * TGEMM_TILE::K;

  const int thisBlockTileIdx = blockIdx.x % N_CHUNKS;

  typename TGEMM_TILE::TType out_regs[TGEMM_TILE::THREAD_OUTPUT_SIZE];

  GemmImpl::Execute(thisBlockAPtr, thisBlockBPtr, out_regs, 0, thisBlockTileIdx,
                    sharedMemPool);

  float* thisBlockOutputPtr = output + thisBlockGemvIdx * TGEMM_TILE::N;
  moe::tasks::internal::WriteGemmTileToGlobalMem_block<TGEMM_TILE>(
      out_regs, thisBlockOutputPtr, 0, thisBlockTileIdx);
}

template <int M, int N, int K, int N_TILE, int K_TILE, int THREADS>
hipError_t batchedGemv(const float* A, const float* B, float* C) {
  // WARNING:
  // A [M,K]
  // B [M*N,K] (-> transposed from standard gemm)
  // C [M,N]
  // NOTE: We process gemm as many gemv, which simiulates moe case.
  constexpr int M_TILE = 1;

  using TGemmTileMetadata =
      GemmTileMetadata<N, K, M_TILE, N_TILE, K_TILE, THREADS, float>;

  constexpr int N_CHUNKS = N / N_TILE;
  constexpr int TILES_TO_PROCESS = N_CHUNKS * M;

  dim3 grid(TILES_TO_PROCESS);
  dim3 block(TGemmTileMetadata::THREADS);

  batchedGemvKernel<TGemmTileMetadata><<<grid, block, 0, 0>>>(A, B, C);

  return hipSuccess;
}

template <typename T>
std::vector<T> refBatchedGemv(const T* A, const T* B, int M, int N, int K) {
  std::vector<T> out;

  for (int gemvIdx = 0; gemvIdx < M; ++gemvIdx) {
    const T* vec = A + gemvIdx * K;
    const T* mat = B + gemvIdx * N * K;
    auto outRow = test::refGemm(vec, mat, 1, N, K);
    out.insert(out.end(), outRow.begin(), outRow.end());
  }

  return out;
}

template <int M, int N, int K, int N_TILE, int K_TILE, int THREADS>
void PerformBenchmark(const std::string& name = "MOE-like batched gemv") {
  // Batched gemv: Matrix A keeps M vectors, represented as single matrix of
  // size, matrix B keeps M Matrices, each of size K*N as a single matrix of
  // size MxNxK. Result is again stored as single matrix of size MxN. This kind
  // of simulates moe, where each token is passed to different expert weights.
  // NOTE that for matices A and B, last dim is always K.
  constexpr int a_size = M * K;
  constexpr int b_size = M * N * K;
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
      10, 10, batchedGemv<M, N, K, N_TILE, K_TILE, THREADS>, d_a, d_b, d_c);

  HIP_ERROR_ASSERT(
      batchedGemv<M, N, K, N_TILE, K_TILE, THREADS>(d_a, d_b, d_c));

  HIP_ERROR_ASSERT(hipDeviceSynchronize())
  HIP_ERROR_ASSERT(hipGetLastError());

  HIP_ERROR_ASSERT(hipMemcpy(host_c.data(), d_c, c_size * sizeof(float),
                             hipMemcpyDeviceToHost));

  // Free device memory.
  HIP_ERROR_ASSERT(hipFree(d_a));
  HIP_ERROR_ASSERT(hipFree(d_b));
  HIP_ERROR_ASSERT(hipFree(d_c));

  auto ref = refBatchedGemv(host_a.data(), host_b.data(), M, N, K);

  CheckAgainstRefBuffer(host_c, ref, ABS_ERROR);

  std::cout << name << ": [" << M << "x" << N << "x" << K
            << "], with params: <N_TILE: " << N_TILE << ", K_TILE: " << K_TILE
            << ", THREADS: " << THREADS << "> kernel avg time is " << avg_ms
            << " ms." << std::endl;
}

template <int TOKENS>
void PerformRealMoeGemms() {
  std::cout << std::endl
            << "Benchmarking real MOE gemms for " << TOKENS
            << " tokens:" << std::endl;
  // Gate gemm:
  {
    const int M = TOKENS;
    const int N = moe::MoeImplMetadata::TILES_CONFIG::GATE_TILE_METADATA::N;
    const int K = moe::MoeImplMetadata::TILES_CONFIG::GATE_TILE_METADATA::K;
    const int N_TILE =
        moe::MoeImplMetadata::TILES_CONFIG::GATE_TILE_METADATA::TILE_N;
    const int K_TILE =
        moe::MoeImplMetadata::TILES_CONFIG::GATE_TILE_METADATA::TILE_K;
    const int THREADS =
        moe::MoeImplMetadata::TILES_CONFIG::GATE_TILE_METADATA::THREADS;

    PerformBenchmark<M, N, K, N_TILE, K_TILE, THREADS>("MOE gate gemm");
  }

  // FFN1 gemm:
  {
    const int M = TOKENS * moe::MoeImplMetadata::MOE_PROBLEM_CONFIG::TOPK;
    const int N = moe::MoeImplMetadata::TILES_CONFIG::FFN1_TILE_METADATA::N;
    const int K = moe::MoeImplMetadata::TILES_CONFIG::FFN1_TILE_METADATA::K;
    const int N_TILE =
        moe::MoeImplMetadata::TILES_CONFIG::FFN1_TILE_METADATA::TILE_N;
    const int K_TILE =
        moe::MoeImplMetadata::TILES_CONFIG::FFN1_TILE_METADATA::TILE_K;
    const int THREADS =
        moe::MoeImplMetadata::TILES_CONFIG::FFN1_TILE_METADATA::THREADS;

    PerformBenchmark<M, N, K, N_TILE, K_TILE, THREADS>("MOE FFN1 gemm");
  }

  // FFN2 gemm:
  {
    const int M = TOKENS * moe::MoeImplMetadata::MOE_PROBLEM_CONFIG::TOPK;
    const int N = moe::MoeImplMetadata::TILES_CONFIG::FFN2_TILE_METADATA::N;
    const int K = moe::MoeImplMetadata::TILES_CONFIG::FFN2_TILE_METADATA::K;
    const int N_TILE =
        moe::MoeImplMetadata::TILES_CONFIG::FFN2_TILE_METADATA::TILE_N;
    const int K_TILE =
        moe::MoeImplMetadata::TILES_CONFIG::FFN2_TILE_METADATA::TILE_K;
    const int THREADS =
        moe::MoeImplMetadata::TILES_CONFIG::FFN2_TILE_METADATA::THREADS;

    PerformBenchmark<M, N, K, N_TILE, K_TILE, THREADS>("MOE FFN2 gemm");
  }
}
TEST(Benchmark, RealMoeGemms) {
  PerformRealMoeGemms<1>();
  PerformRealMoeGemms<16>();
  PerformRealMoeGemms<32>();
}

TEST(Benchmark, MoeLikeBatchedGemv) {
  // Benchmarking other metadata parameters, to see how it performs.
  PerformBenchmark<8, 768, 2048, 256, 32, 256>();
  PerformBenchmark<8, 2048, 768, 256, 32, 256>();
  PerformBenchmark<128, 2048, 768, 256, 16, 256>();

  // One can add here more specific configurations, to see how it performs.
}
}  // namespace