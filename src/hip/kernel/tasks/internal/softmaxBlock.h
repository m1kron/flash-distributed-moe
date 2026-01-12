#pragma once
#include "src/hip/utils/hipWarpSize.h"

namespace moe {
namespace tasks {
namespace internal {
// GPT5-mini generated code:
// Optimized device softmax for a single row where:
//  - row length (cols) == _SIZE
//  - blockDim.x == _SIZE (one thread per column)
// Uses warp-level reductions (assumes warpSize == 64 on ROCm) to reduce
// shared-memory traffic.
//
// Notes:
//  - This implementation targets float inputs (template kept for minimal
//  flexibility).
//  - All intermediate reductions use float/warp shuffles and final accumulation
//  uses double
//    for a small number of warp partials to keep precision.
template <int _SIZE, typename T = float>
__device__ float Softmax_block(T input, void* sharedMemPool) {
  constexpr int COLS = _SIZE;
  constexpr int NUM_WARPS = COLS / WARP_SIZE;

  const int tid = threadIdx.x;
  const int lane = tid & (WARP_SIZE - 1);
  const int warpId = tid >> 6;  // tid / 64

  // load value (use float for warp shuffles)
  T val_f = input;

  // warp-level max reduction using shfl_down
  T wmax = val_f;
  // offsets: 32,16,8,4,2,1 for 64-lane warp
  for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
    T other = __shfl_down(wmax, offset);
    wmax = fmaxf(wmax, other);
  }

  // shared per-warp maxima and per-warp sums
  T* sharedMemPoolFloat = reinterpret_cast<T*>(sharedMemPool);
  T* s_warp_max = sharedMemPoolFloat;
  T* s_warp_sum = sharedMemPoolFloat + NUM_WARPS;

  // warp leader writes warp max to shared
  if (lane == 0) s_warp_max[warpId] = wmax;
  __syncthreads();

  // thread 0 (or first warp) computes global max from per-warp maxima
  T global_max_f;
  if (tid == 0) {
    double gm = -INFINITY;
    for (int w = 0; w < NUM_WARPS; ++w) {
      double v = static_cast<double>(s_warp_max[w]);
      if (v > gm) gm = v;
    }
    global_max_f = static_cast<T>(gm);
    s_warp_max[0] = global_max_f;  // broadcast carrier
  }
  __syncthreads();
  global_max_f = s_warp_max[0];

  // compute exp shifted by global max (float)
  T e = expf(val_f - global_max_f);

  // warp-level sum of exponentials
  T wsum = e;
  for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
    T other = __shfl_down(wsum, offset);
    wsum += other;
  }

  // warp leader writes warp sum to shared
  if (lane == 0) s_warp_sum[warpId] = wsum;
  __syncthreads();

  // thread 0 reduces per-warp sums to global sum.
  T global_sum_d;
  if (tid == 0) {
    T acc = 0.0f;
    for (int w = 0; w < NUM_WARPS; ++w) acc += s_warp_sum[w];
    global_sum_d = acc;
    s_warp_sum[0] = global_sum_d;  // carrier
  }
  __syncthreads();
  global_sum_d = s_warp_sum[0];

  // write normalized output
  if (global_sum_d <= 0.0) {
    return static_cast<T>(1.0f / static_cast<T>(COLS));
  } else {
    return static_cast<T>(e / global_sum_d);
  }
}
}  // namespace internal
}  // namespace tasks
}  // namespace moe