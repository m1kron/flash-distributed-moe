#pragma once
#include "src/hip/common/gemmTileMetadata.h"

namespace moe {
namespace tasks {
namespace internal {

// Writes tile output distributed in registers in all participating threads to
// global mem.
template <typename GEMM_TILE_PARAMS>
__device__ void WriteGemmTileToGlobalMem_block(
    const typename GEMM_TILE_PARAMS::TType* __restrict__ outTile_thread_regs,
    typename GEMM_TILE_PARAMS::TType* __restrict__ out_global,
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

}  // namespace internal
}  // namespace tasks
}  // namespace moe