#pragma once

#include <hip/hip_runtime.h>

#include <cmath>

// GPT5-mini generated code:
// Optimized device softmax for a single row where:
//  - row length (cols) == 256
//  - blockDim.x == 256 (one thread per column)
// Uses warp-level reductions (assumes warpSize == 64 on ROCm) to reduce
// shared-memory traffic.
//
// Notes:
//  - This implementation targets float inputs (template kept for minimal
//  flexibility).
//  - All intermediate reductions use float/warp shuffles and final accumulation
//  uses double
//    for a small number of warp partials to keep precision.
template <typename T = float>
__device__ float softmax_row_256_warpopt(T input) {
  constexpr int COLS = 256;
  constexpr int WARP_SIZE = warpSize;          // ROCm wavefront size
  constexpr int NUM_WARPS = COLS / WARP_SIZE;  // = 4

  const int tid = threadIdx.x;  // expected 0..255
  const int lane = tid & (WARP_SIZE - 1);
  const int warpId = tid >> 6;  // tid / 64

  // load value (use float for warp shuffles)
  float val_f = input;

  // warp-level max reduction using shfl_down
  float wmax = val_f;
  // offsets: 32,16,8,4,2,1 for 64-lane warp
  for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
    float other = __shfl_down(wmax, offset);
    wmax = fmaxf(wmax, other);
  }

  // shared per-warp maxima and per-warp sums
  __shared__ float s_warp_max[NUM_WARPS];
  __shared__ float s_warp_sum[NUM_WARPS];

  // warp leader writes warp max to shared
  if (lane == 0) s_warp_max[warpId] = wmax;
  __syncthreads();

  // thread 0 (or first warp) computes global max from per-warp maxima
  float global_max_f;
  if (tid == 0) {
    double gm = -INFINITY;
    for (int w = 0; w < NUM_WARPS; ++w) {
      double v = static_cast<double>(s_warp_max[w]);
      if (v > gm) gm = v;
    }
    global_max_f = static_cast<float>(gm);
    s_warp_max[0] = global_max_f;  // broadcast carrier
  }
  __syncthreads();
  global_max_f = s_warp_max[0];

  // compute exp shifted by global max (float)
  float e = expf(val_f - global_max_f);

  // warp-level sum of exponentials
  float wsum = e;
  for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
    float other = __shfl_down(wsum, offset);
    wsum += other;
  }

  // warp leader writes warp sum to shared
  if (lane == 0) s_warp_sum[warpId] = wsum;
  __syncthreads();

  // thread 0 reduces per-warp sums to global sum (double for safety)
  double global_sum_d;
  if (tid == 0) {
    double acc = 0.0;
    for (int w = 0; w < NUM_WARPS; ++w)
      acc += static_cast<double>(s_warp_sum[w]);
    global_sum_d = acc;
    s_warp_sum[0] = static_cast<float>(global_sum_d);  // carrier
  }
  __syncthreads();
  global_sum_d = static_cast<double>(s_warp_sum[0]);

  // write normalized output
  if (global_sum_d <= 0.0) {
    return static_cast<T>(1.0 / static_cast<double>(COLS));
  } else {
    return static_cast<T>((static_cast<double>(e) / global_sum_d));
  }
}