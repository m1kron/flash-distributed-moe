#pragma once
#include "src/hip/kernel/tasks/internal/gemmImpl/gemmImplSelector.h"
#include "src/hip/kernel/tasks/internal/gemmTileBlock.h"

constexpr int TILE_M = 64;
constexpr int TILE_N = 32;
constexpr int THREADS = 128;

template <int N>
struct TestTileGemm {
  using TGemmTileParams =
      GemmTileParams<N, 2048, TILE_M, TILE_N, 32, THREADS, float>;
  using GemmImpl =
      typename moe::tasks::internal::GemmImplSelector<TGemmTileParams>::type;

  static constexpr int SHARED_MEM_NEEDES_BYTES =
      GemmImpl::NeededSharedMemBytes();

  static __device__ void gemm_64x64x2048_single_block(
      const float* __restrict__ A, const float* __restrict__ B,
      float* __restrict__ C, int blockTileRowStartIdx, int blockTileColStartIdx,
      void* sharedMemPool) {
    typename GemmImpl::TILE_METADATA::TType
        out_regs[GemmImpl::TILE_METADATA::THREAD_OUTPUT_SIZE];

    GemmImpl::Execute(A, B, out_regs, blockTileRowStartIdx,
                      blockTileColStartIdx, sharedMemPool);
    moe::tasks::internal::WriteGemmTileToGlobalMem_block<
        typename GemmImpl::TILE_METADATA>(out_regs, C, blockTileRowStartIdx,
                                          blockTileColStartIdx);
  }
};
