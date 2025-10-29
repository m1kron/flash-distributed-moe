#pragma once

template <int _N, int _K, int _TILE_M, int _TILE_N, int _TILE_K, int _THREADS>
struct GemmTileParams {
  using TOutputType = float;
  using TInputType = float;
  constexpr static int N = _N;
  constexpr static int K = _K;
  constexpr static int TILE_M = _TILE_M;
  constexpr static int TILE_N = _TILE_N;
  constexpr static int TILE_K = _TILE_K;
  constexpr static int THREADS = _THREADS;
  constexpr static int THREAD_OUTPUT_SIZE = (TILE_M * TILE_N) / THREADS;
  constexpr static int SHARED_MEM_NEEDES_BYTES =
      ((TILE_M * TILE_K) + (TILE_K * TILE_N)) * sizeof(TInputType);
};

// NOTE: GPT5-mini generated code with my minor changes.
// Device GEMM for single block.
// A: M x K (row-major), B: K x N (row-major), C: M x N (row-major).
template <typename GEMM_TILE_PARAMS>
__device__ void GemmTile_block(
    const float* __restrict__ A, const float* __restrict__ B,
    typename GEMM_TILE_PARAMS::TOutputType* __restrict__ CTile_thread_regs,
    int blockTileRowStartIdx, int blockTileColStartIdx) {
  constexpr int TILE_M = GEMM_TILE_PARAMS::TILE_M;
  constexpr int TILE_N = GEMM_TILE_PARAMS::TILE_N;
  constexpr int K = GEMM_TILE_PARAMS::K;
  constexpr int N = GEMM_TILE_PARAMS::N;
  // tile along K dimension
  constexpr int TILE_K = GEMM_TILE_PARAMS::TILE_K;
  constexpr int OUT_PER_THREAD = GEMM_TILE_PARAMS::THREAD_OUTPUT_SIZE;
  const int BLOCK_START_ROW = blockTileRowStartIdx * TILE_M;
  const int BLOCK_START_COL = blockTileColStartIdx * TILE_N;

  __shared__ float sA[TILE_M * TILE_K];  // [row, kLocal]
  __shared__ float sB[TILE_K * TILE_N];  // [kLocal, col]

  const int tid = threadIdx.x;
  const int baseLinear = tid * OUT_PER_THREAD;
#pragma unroll
  for (int i = 0; i < OUT_PER_THREAD; ++i) CTile_thread_regs[i] = 0.0f;

  for (int kt = 0; kt < K; kt += TILE_K) {
    // Cooperative load A tile: sA[row * TILE_K + kLocal] = A[row*K + (kt +
    // kLocal)]
    for (int idx = tid; idx < TILE_M * TILE_K; idx += blockDim.x) {
      const int row = idx / TILE_K;
      const int kLocal = idx % TILE_K;  // 0..TILE_K-1
      sA[row * TILE_K + kLocal] =
          A[(BLOCK_START_ROW + row) * K + (kt + kLocal)];
    }

    // Cooperative load B tile: sB[kLocal * N + col] = B[(kt + kLocal) * N +
    // col]
    for (int idx = tid; idx < TILE_K * TILE_N; idx += blockDim.x) {
      const int kLocal = idx / TILE_N;  // 0..TILE_K-1
      const int col = idx % TILE_N;
      sB[kLocal * TILE_N + col] =
          B[(kt + kLocal) * N + (col + BLOCK_START_COL)];
    }

    __syncthreads();

    // Multiply-accumulate over local k
#pragma unroll
    for (int kLocal = 0; kLocal < TILE_K; ++kLocal) {
      // For each output this thread owns, read A once and B once (per kLocal)
#pragma unroll
      for (int out = 0; out < OUT_PER_THREAD; ++out) {
        const int linear = baseLinear + out;
        const int r = linear / TILE_N;
        const int c = linear % TILE_N;
        const float aval = sA[r * TILE_K + kLocal];
        const float bval = sB[kLocal * TILE_N + c];
        CTile_thread_regs[out] += aval * bval;
      }
    }

    __syncthreads();
  }
}

// Writes tile output distributed in registers in all participating threads to
// global mem.
template <typename GEMM_TILE_PARAMS>
__device__ void WriteGemmTileToGlobalMem_block(
    const typename GEMM_TILE_PARAMS::
        TOutputType* __restrict__ outTile_thread_regs,
    typename GEMM_TILE_PARAMS::TOutputType* __restrict__ out_global,
    int blockTileRowStartIdx, int blockTileColStartIdx) {
  constexpr int OUT_PER_THREAD = GEMM_TILE_PARAMS::THREAD_OUTPUT_SIZE;
  constexpr int TILE_M = GEMM_TILE_PARAMS::TILE_M;
  constexpr int TILE_N = GEMM_TILE_PARAMS::TILE_N;
  constexpr int N = GEMM_TILE_PARAMS::N;
  const int BLOCK_START_ROW = blockTileRowStartIdx * TILE_M;
  const int BLOCK_START_COL = blockTileColStartIdx * TILE_N;
  const int baseLinear = threadIdx.x * OUT_PER_THREAD;
#pragma unroll
  for (int out = 0; out < OUT_PER_THREAD; ++out) {
    const int linear = baseLinear + out;
    const int r = linear / TILE_N;
    const int c = linear % TILE_N;
    out_global[(BLOCK_START_ROW + r) * N + c + BLOCK_START_COL] =
        outTile_thread_regs[out];
  }
}
