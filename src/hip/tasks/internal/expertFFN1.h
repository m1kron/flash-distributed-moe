#pragma once
#include "gemmTileBlock.h"

namespace tasks {

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

// Task for expert's FFN1. rowIdx and colIdx refers to output tiles.
// Assumption is that output's N dimension equal to 2 * (expertWeights N
// dimension).
template <typename T_OUTPUT_TILE>
__device__ void expertFFN1Task(
    const typename T_OUTPUT_TILE::TInputType* __restrict__ tokens,
    const typename T_OUTPUT_TILE::TInputType* __restrict__ expertWeights,
    typename T_OUTPUT_TILE::TOutputType* __restrict__ output, int rowIdx,
    int colIdx, void* sharedMemPool) {
  constexpr int N_CHUNKS = T_OUTPUT_TILE::N / T_OUTPUT_TILE::TILE_N;

  const int tokenIdx = rowIdx;
  const int tileCol = colIdx;

  using T_FFN1_TILE =
      GemmTileParams<T_OUTPUT_TILE::N * 2, T_OUTPUT_TILE::K,
                     T_OUTPUT_TILE::TILE_M, T_OUTPUT_TILE::TILE_N,
                     T_OUTPUT_TILE::TILE_K, T_OUTPUT_TILE::THREADS>;

  // TODO: Fuse gemms.
  typename T_FFN1_TILE::TOutputType w1_regs[T_FFN1_TILE::THREAD_OUTPUT_SIZE];
  GemmTile_block<T_FFN1_TILE>(tokens, expertWeights, w1_regs, tokenIdx, tileCol,
                              sharedMemPool);

  typename T_FFN1_TILE::TOutputType w3_regs[T_FFN1_TILE::THREAD_OUTPUT_SIZE];
  GemmTile_block<T_FFN1_TILE>(tokens, expertWeights, w3_regs, tokenIdx,
                              tileCol + N_CHUNKS, sharedMemPool);

  // Silu:
  for (int i = 0; i < T_FFN1_TILE::THREAD_OUTPUT_SIZE; ++i) {
    w3_regs[i] = silu(w1_regs[i], w3_regs[i]);
  }

  // w3_regs contains output

  WriteGemmTileToGlobalMem_block<T_OUTPUT_TILE>(w3_regs, output, tokenIdx,
                                                tileCol);
}

}  // namespace tasks