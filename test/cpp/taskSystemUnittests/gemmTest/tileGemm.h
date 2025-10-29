#pragma once
#include "../../../../src/hip/tasks/internal/gemmTileBlock.h"

constexpr int TILE_M = 64;
constexpr int TILE_N = 32;
constexpr int THREADS = 128;

template <int N>
__device__ void gemm_64x64x2048_single_block(const float* __restrict__ A,
                                             const float* __restrict__ B,
                                             float* __restrict__ C,
                                             int blockTileRowStartIdx,
                                             int blockTileColStartIdx) {
  using TGemmTileParams = GemmTileParams<N, 2048, TILE_M, TILE_N, 32, THREADS>;

  typename TGemmTileParams::TOutputType
      out_regs[TGemmTileParams::THREAD_OUTPUT_SIZE];

  GemmTile_block<TGemmTileParams>(A, B, out_regs, blockTileRowStartIdx,
                                  blockTileColStartIdx);
  WriteGemmTileToGlobalMem_block<TGemmTileParams>(
      out_regs, C, blockTileRowStartIdx, blockTileColStartIdx);
}
