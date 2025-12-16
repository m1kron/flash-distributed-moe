#pragma once
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
  const int tokenIdx = rowIdx;
  const int tileCol = colIdx;

  const typename T_FFN1_TILE_IMPL::TILE_METADATA::TType* expertWeights2 =
      expertWeights +
      T_FFN1_TILE_IMPL::TILE_METADATA::K * T_FFN1_TILE_IMPL::TILE_METADATA::N;

  // TODO: Fuse gemms.
  typename T_FFN1_TILE_IMPL::TILE_METADATA::TType
      w1_regs[T_FFN1_TILE_IMPL::TILE_METADATA::THREAD_OUTPUT_SIZE];

  T_FFN1_TILE_IMPL::Execute(tokens, expertWeights, w1_regs, tokenIdx, tileCol,
                            sharedMemPool);

  T_FFN1_TILE_IMPL::Execute(tokens, expertWeights2, outRegs, tokenIdx, tileCol,
                            sharedMemPool);

  // Silu:
  for (int i = 0; i < T_FFN1_TILE_IMPL::TILE_METADATA::THREAD_OUTPUT_SIZE;
       ++i) {
    outRegs[i] = silu(w1_regs[i], outRegs[i]);
  }
}
}  // namespace internal
}  // namespace tasks
}  // namespace moe