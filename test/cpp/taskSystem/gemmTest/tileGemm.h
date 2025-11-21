#pragma once
#include "src/hip/tasks/internal/gemmTileBlock.h"

constexpr int TILE_M = 64;
constexpr int TILE_N = 32;
constexpr int THREADS = 128;

template <int N>
struct TestTileGemm {
  using TGemmTileParams = GemmTileParams<N, 2048, TILE_M, TILE_N, 32, THREADS>;

  static constexpr int SHARED_MEM_NEEDES_BYTES =
      TGemmTileParams::SHARED_MEM_NEEDES_BYTES;

  static __device__ void gemm_64x64x2048_single_block(
      const float* __restrict__ A, const float* __restrict__ B,
      float* __restrict__ C, int blockTileRowStartIdx, int blockTileColStartIdx,
      void* sharedMemPool) {
    typename TGemmTileParams::TType
        out_regs[TGemmTileParams::THREAD_OUTPUT_SIZE];

    moe::tasks::internal::GemmTile_block<TGemmTileParams>(
        A, B, out_regs, blockTileRowStartIdx, blockTileColStartIdx,
        sharedMemPool);
    moe::tasks::internal::WriteGemmTileToGlobalMem_block<TGemmTileParams>(
        out_regs, C, blockTileRowStartIdx, blockTileColStartIdx);
  }
};
