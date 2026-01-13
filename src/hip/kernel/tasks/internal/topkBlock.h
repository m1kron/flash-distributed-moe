#pragma once

namespace moe {
namespace tasks {
namespace internal {

// GPT5-mini generated code with my minor changes.
/*
  Bitonic-based device top-8 for a _SIZE-item tile.

  - Assumes blockDim.x == _SIZE and each block handles one _SIZE-item tile.
  - Each thread initially owns one item: data[tid]
  - After the in-place bitonic sort in shared memory the array is sorted
    in descending order (largest first). Threads 0..7 then write top-8
    values/indices to out_vals/out_idx.

  Notes:
  - This implementation favors clarity and correctness of the bitonic network.
  - TOPK=8 and TILE=_SIZE are compile-time constants here.
*/

template <int _SIZE, typename T = float>
__device__ void Topk8_block(T input, T* __restrict__ out_vals,
                            int* __restrict__ out_idx,
                            void* __restrict__ sharedMemPool) {
  constexpr int THREADS = _SIZE;
  constexpr int TOPK = 8;

  const int tid = threadIdx.x;

  // Shared arrays for values and indices.
  char* sharedMemPoolBytes = reinterpret_cast<char*>(sharedMemPool);
  T* svals = reinterpret_cast<T*>(sharedMemPoolBytes);
  int* sidx = reinterpret_cast<int*>(sharedMemPoolBytes + sizeof(T) * THREADS);

  // Load
  svals[tid] = input;
  sidx[tid] = tid;
  __syncthreads();

  // Bitonic sort network (in-place) for _SIZE elements.
  // We adapt the classic bitonic compare-exchange to produce descending order.
  // Outer loop: size of subsequence to be bitonic-sorted (k)
  for (int k = 2; k <= THREADS; k <<= 1) {
    // Inner loop: compare distance
    for (int j = k >> 1; j > 0; j >>= 1) {
      int ixj = tid ^ j;
      if (ixj > tid) {
        // direction determined by bit k in tid:
        // when (tid & k) == 0 -> do one comparison type, else the other.
        // We choose comparisons so final order is descending (largest first).
        bool up = ((tid & k) == 0);  // "up" groups
        T a = svals[tid];
        T b = svals[ixj];
        int ia = sidx[tid];
        int ib = sidx[ixj];

        // For descending final order we swap when:
        //  - in "up" groups: a < b  -> swap (move larger to lower index)
        //  - in "down" groups: a > b -> swap
        bool doSwap = (up ? (a < b) : (a > b));
        if (doSwap) {
          svals[tid] = b;
          svals[ixj] = a;
          sidx[tid] = ib;
          sidx[ixj] = ia;
        }
      }
      __syncthreads();
    }
  }

  // After the sort: svals[0.._SIZE] are in descending order (largest at index
  // 0). First TOPK threads write indices and compute softmax of values.
  if (tid < TOPK) {
    constexpr long long TOPK_MASK = 0xFF;  // Only first 8 lanes participate

    // Cache value in register (single shared memory read)
    T val = svals[tid];
    out_idx[tid] = sidx[tid];

    // Softmax: numerically stable using warp shuffles (no shared memory).
    // Step 1: Broadcast max (lane 0 has max since sorted descending)
    T max_val = __shfl_sync(TOPK_MASK, val, 0);
    T exp_val = __expf(val - max_val);  // Fast single-precision exp

    // Step 2: Parallel reduction for sum of exponentials (log2(8) = 3 steps)
    T exp_sum = exp_val;
    exp_sum += __shfl_xor_sync(TOPK_MASK, exp_sum, 4);
    exp_sum += __shfl_xor_sync(TOPK_MASK, exp_sum, 2);
    exp_sum += __shfl_xor_sync(TOPK_MASK, exp_sum, 1);

    // Step 3: Normalize with fast reciprocal multiply
    out_vals[tid] = exp_val * __frcp_rn(exp_sum);
  }
  __syncthreads();
}
}  // namespace internal
}  // namespace tasks
}  // namespace moe