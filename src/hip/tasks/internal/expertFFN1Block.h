#pragma once
#include "gemmTileBlock.h"

namespace moe {
namespace tasks {
namespace internal {
inline __device__ float silu(float a, float b) {
  float sig;
  if (a >= 0.0f) {
    sig = 1.0f / (1.0f + expf(-a));
  } else {
    float ea = expf(a);
    sig = ea / (1.0f + ea);
  }

  const float silu = a * sig;
  return silu * b;
}

// Task impl for expert's FFN1. rowIdx and colIdx refers to output tiles.
// Assumption is that output's N dimension equal to 2 * (expertWeights N
// dimension).
// Output is returned in register array of size
// T_OUTPUT_TILE::THREAD_OUTPUT_SIZE;
template <typename T_OUTPUT_TILE>
__device__ void expertFFN1_block(
    const typename T_OUTPUT_TILE::TType* __restrict__ tokens,
    const typename T_OUTPUT_TILE::TType* __restrict__ expertWeights,
    typename T_OUTPUT_TILE::TType* __restrict__ outRegs, int rowIdx, int colIdx,
    void* sharedMemPool) {
  constexpr int N_CHUNKS = T_OUTPUT_TILE::N / T_OUTPUT_TILE::TILE_N;

  const int tokenIdx = rowIdx;
  const int tileCol = colIdx;

  using T_FFN1_TILE =
      GemmTileParams<T_OUTPUT_TILE::N * 2, T_OUTPUT_TILE::K,
                     T_OUTPUT_TILE::TILE_M, T_OUTPUT_TILE::TILE_N,
                     T_OUTPUT_TILE::TILE_K, T_OUTPUT_TILE::THREADS,
                     typename T_OUTPUT_TILE::TType>;

  static_assert(T_FFN1_TILE::THREAD_OUTPUT_SIZE ==
                T_OUTPUT_TILE::THREAD_OUTPUT_SIZE);

  // TODO: Fuse gemms.
  typename T_FFN1_TILE::TType w1_regs[T_FFN1_TILE::THREAD_OUTPUT_SIZE];
  moe::tasks::internal::GemmTile_block<T_FFN1_TILE>(
      tokens, expertWeights, w1_regs, tokenIdx, tileCol, sharedMemPool);

  moe::tasks::internal::GemmTile_block<T_FFN1_TILE>(
      tokens, expertWeights, outRegs, tokenIdx, tileCol + N_CHUNKS,
      sharedMemPool);

  // Silu:
  for (int i = 0; i < T_FFN1_TILE::THREAD_OUTPUT_SIZE; ++i) {
    outRegs[i] = silu(w1_regs[i], outRegs[i]);
  }
}
}  // namespace internal
}  // namespace tasks
}  // namespace moe