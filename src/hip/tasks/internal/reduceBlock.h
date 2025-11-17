#pragma once

#include <hip/hip_runtime.h>

namespace tasks {
// GPT5-mini generated code with my minor changes.
/*
  Vectorized reduce for a tile 8 x 512 -> 1 x 512 (column-wise sum).
  - Processes 4 columns at once using float4.
  - Assumes COLS == 512 and ROWS == 8.
  - Assumes blockDim.x == 128 (but works with other thread counts).
*/

template <int TOPK, int COLS, int THREADS>
__device__ inline void Reduce_block(const float* __restrict__ in,
                                    float* __restrict__ out) {
  constexpr int ROWS = TOPK;
  constexpr int VEC_WIDTH = 4;
  constexpr int TOTAL_VEC = COLS / VEC_WIDTH;  // 128
  constexpr int PER_THREAD_ITERS = (TOTAL_VEC + THREADS - 1) / THREADS;

  const int tid = threadIdx.x;

  // reinterpret input/out as float4 arrays (safe because COLS is multiple of 4)
  const float4* in_vec_base = reinterpret_cast<const float4*>(in);
  float4* out_vec = reinterpret_cast<float4*>(out);

#pragma unroll
  for (int i = 0; i < PER_THREAD_ITERS; ++i) {
    const int vecIdx = tid + i * THREADS;

    // accumulate across 8 rows
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
      // each row has TOTAL_VEC float4 elements, row-major layout ensures
      // contiguous blocks
      const float4 v = in_vec_base[r * TOTAL_VEC + vecIdx];
      acc.x += v.x;
      acc.y += v.y;
      acc.z += v.z;
      acc.w += v.w;
    }

    out_vec[vecIdx] = acc;
  }
}
}  // namespace tasks