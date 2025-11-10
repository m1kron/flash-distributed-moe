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
template <typename TGEMM_TILE>
__device__ void expertFFN1Task(const float* __restrict__ tokens,
                               const float* __restrict__ expertWeights,
                               float* __restrict__ output, int rowIdx,
                               int colIdx, void* sharedMemPool) {
  constexpr int N_CHUNKS = (TGEMM_TILE::N / 2) / TGEMM_TILE::TILE_N;

  const int tokenIdx = rowIdx;
  const int tileCol = colIdx;

  // TODO: Fuse gemms.
  typename TGEMM_TILE::TOutputType w1_regs[TGEMM_TILE::THREAD_OUTPUT_SIZE];
  GemmTile_block<TGEMM_TILE>(tokens, expertWeights, w1_regs, tokenIdx, tileCol,
                             sharedMemPool);

  typename TGEMM_TILE::TOutputType w3_regs[TGEMM_TILE::THREAD_OUTPUT_SIZE];
  GemmTile_block<TGEMM_TILE>(tokens, expertWeights, w3_regs, tokenIdx,
                             tileCol + N_CHUNKS, sharedMemPool);

  // Silu:
  for (int i = 0; i < TGEMM_TILE::THREAD_OUTPUT_SIZE; ++i) {
    w3_regs[i] = silu(w1_regs[i], w3_regs[i]);
  }

  // w3_regs contains output
  using T_OUTPUT_TILE = GemmTileParams<TGEMM_TILE::N / 2, TGEMM_TILE::K,
                                       TGEMM_TILE::TILE_M, TGEMM_TILE::TILE_N,
                                       TGEMM_TILE::TILE_K, TGEMM_TILE::THREADS>;

  WriteGemmTileToGlobalMem_block<T_OUTPUT_TILE>(w3_regs, output, tokenIdx,
                                                tileCol);
}

}  // namespace tasks