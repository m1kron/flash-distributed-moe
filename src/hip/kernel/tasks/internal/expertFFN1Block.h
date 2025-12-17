#pragma once
#include "src/hip/utils/hipDeviceUtils.h"
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
// T_FFN1_TILE::THREAD_OUTPUT_SIZE;
template <typename T_FFN1_TILE_IMPL>
__device__ void expertFFN1_block(
    const typename T_FFN1_TILE_IMPL::TILE_METADATA::TType* __restrict__ tokens,
    const typename T_FFN1_TILE_IMPL::TILE_METADATA::
        TType* __restrict__ expertWeights,
    typename T_FFN1_TILE_IMPL::TILE_METADATA::TType* __restrict__ outRegs,
    int rowIdx, int colIdx, void* sharedMemPool) {
  HIP_DEVICE_ASSERT(rowIdx == 0);

  using FFN1_TILE = typename T_FFN1_TILE_IMPL::TILE_METADATA;

  const typename FFN1_TILE::TType* expertWeights2 =
      expertWeights + FFN1_TILE::K * FFN1_TILE::N;

  // TODO: Fuse gemms.
  typename FFN1_TILE::TType w1_regs[FFN1_TILE::THREAD_OUTPUT_SIZE];

  T_FFN1_TILE_IMPL::Execute(tokens, expertWeights, w1_regs, 0, colIdx,
                            sharedMemPool);

  T_FFN1_TILE_IMPL::Execute(tokens, expertWeights2, outRegs, 0, colIdx,
                            sharedMemPool);

  // Silu:
  for (int i = 0; i < FFN1_TILE::THREAD_OUTPUT_SIZE; ++i) {
    outRegs[i] = silu(w1_regs[i], outRegs[i]);
  }
}
}  // namespace internal
}  // namespace tasks
}  // namespace moe