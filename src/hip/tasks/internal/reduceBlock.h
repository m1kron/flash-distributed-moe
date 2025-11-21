#pragma once

namespace moe {
namespace tasks {
namespace internal {
// GPT5-mini generated code with my minor changes.
/*
  Vectorized reduce for a tile TOPK x TILE_SIZE -> 1 x TILE_SIZE (column-wise
  sum) of a matrix of size TOPK * HIDDEN_SIZE.
  - Processes 4 columns at once using float4.
  - Assumes TILE_SIZE == 512 and TOPK == 8.
  - Assumes blockDim.x == 128 (but works with other thread counts).
*/

template <int TOPK, int TILE_SIZE, int THREADS, int HIDDEN_SIZE>
__device__ inline void Reduce_block(const float* __restrict__ in,
                                    float* __restrict__ out,
                                    int blockTileColStartIdx) {
  constexpr int ROWS = TOPK;
  constexpr int VEC_WIDTH = 4;
  static_assert((TILE_SIZE % VEC_WIDTH) == 0);
  constexpr int TOTAL_VEC = TILE_SIZE / VEC_WIDTH;  // 128
  constexpr int PER_THREAD_ITERS = (TOTAL_VEC + THREADS - 1) / THREADS;
  constexpr int VEC_HIDDEN_SIZE = HIDDEN_SIZE / VEC_WIDTH;

  const int tid = threadIdx.x;

  // reinterpret input/out as float4 arrays (safe because TILE_SIZE is multiple
  // of 4)
  const float4* in_vec_base =
      reinterpret_cast<const float4*>(in + blockTileColStartIdx * TILE_SIZE);
  float4* out_vec =
      reinterpret_cast<float4*>(out + blockTileColStartIdx * TILE_SIZE);

#pragma unroll
  for (int i = 0; i < PER_THREAD_ITERS; ++i) {
    const int vecIdx = tid + i * THREADS;

    // accumulate across 8 rows
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
      // each row has VEC_HIDDEN_SIZE float4 elements, row-major layout ensures
      // contiguous blocks
      const float4 v = in_vec_base[r * VEC_HIDDEN_SIZE + vecIdx];
      acc.x += v.x;
      acc.y += v.y;
      acc.z += v.z;
      acc.w += v.w;
    }

    out_vec[vecIdx] = acc;
  }
}
}  // namespace internal
}  // namespace tasks
}  // namespace moe