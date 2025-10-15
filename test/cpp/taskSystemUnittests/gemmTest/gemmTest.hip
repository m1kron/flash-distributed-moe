#include <hip/hip_runtime.h>

#include <vector>

#include "gtest/gtest.h"

#define HIP_ERROR_ASSERT(condition)                     \
  {                                                     \
    const hipError_t error = condition;                 \
    ASSERT_EQ(error, hipSuccess) << " for " #condition; \
  }

template <typename T>
static void CheckConstValBuffer(const T* buffer, int size, T value) {
  for (int i = 0; i < size; ++i) {
    ASSERT_EQ(buffer[i], value);
  }
}

// Device GEMM for single block: M=64, N=64, K=2048.
// Assumes blockDim.x == 256 and single block launches this kernel/function.
// A: M x K (row-major), B: K x N (row-major), C: M x N (row-major).
__device__ void gemm_64x64x2048_single_block(const float* __restrict__ A,
                                             const float* __restrict__ B,
                                             float* __restrict__ C) {
  constexpr int M = 64;
  constexpr int N = 64;
  constexpr int K = 2048;
  // tile along K dimension
  constexpr int TILE_K = 32;          // 32*2 arrays -> fits typical shared mem
  constexpr int OUT_PER_THREAD = 16;  // 256 threads * 16 = 4096 outputs

  __shared__ float sA[M * TILE_K];  // [row, kLocal]
  __shared__ float sB[TILE_K * N];  // [kLocal, col]

  const int tid = threadIdx.x;
  const int baseLinear = tid * OUT_PER_THREAD;  // linear index in 0..4095
  float acc[OUT_PER_THREAD];
#pragma unroll
  for (int i = 0; i < OUT_PER_THREAD; ++i) acc[i] = 0.0f;

  // Precompute row/col for this thread's outputs to avoid recompute inside k
  // loop
  int out_row[OUT_PER_THREAD];
  int out_col[OUT_PER_THREAD];
#pragma unroll
  for (int i = 0; i < OUT_PER_THREAD; ++i) {
    int linear = baseLinear + i;
    out_row[i] = linear / N;
    out_col[i] = linear % N;
  }

  const int loadCount =
      M * TILE_K;  // equals TILE_K * N for M==N==64 when TILE_K==32 => 2048

  for (int kt = 0; kt < K; kt += TILE_K) {
    // Cooperative load A tile: sA[row * TILE_K + kLocal] = A[row*K + (kt +
    // kLocal)]
    for (int idx = tid; idx < loadCount; idx += blockDim.x) {
      const int row = idx / TILE_K;     // 0..63
      const int kLocal = idx % TILE_K;  // 0..TILE_K-1
      sA[row * TILE_K + kLocal] = A[row * K + (kt + kLocal)];
    }

    // Cooperative load B tile: sB[kLocal * N + col] = B[(kt + kLocal) * N +
    // col]
    for (int idx = tid; idx < loadCount; idx += blockDim.x) {
      const int kLocal = idx / N;  // 0..TILE_K-1
      const int col = idx % N;     // 0..63
      sB[kLocal * N + col] = B[(kt + kLocal) * N + col];
    }

    __syncthreads();

    // Multiply-accumulate over local k
    for (int kLocal = 0; kLocal < TILE_K; ++kLocal) {
      // For each output this thread owns, read A once and B once (per kLocal)
#pragma unroll
      for (int out = 0; out < OUT_PER_THREAD; ++out) {
        const int r = out_row[out];
        const int c = out_col[out];
        const float aval = sA[r * TILE_K + kLocal];
        const float bval = sB[kLocal * N + c];
        acc[out] += aval * bval;
      }
    }

    __syncthreads();
  }

  // Write results
#pragma unroll
  for (int out = 0; out < OUT_PER_THREAD; ++out) {
    const int linear = baseLinear + out;
    if (linear < (M * N)) {
      const int r = out_row[out];
      const int c = out_col[out];
      C[r * N + c] = acc[out];
    }
  }
}

__global__ void gemmStatic(const float* __restrict__ A,
                           const float* __restrict__ B, float* __restrict__ C) {
  gemm_64x64x2048_single_block(A, B, C);
}

TEST(TaskSystem, GemmCoherenceTest) {
  // Prepare input buffers for module
  const int m_size = 64;
  const int n_size = 64;
  const int k_size = 2048;

  const int a_size = m_size * k_size;
  const int b_size = k_size * n_size;
  const int c_size = m_size * n_size;

  constexpr float alpha = 2.f;

  std::vector<float> host_a(a_size);
  std::fill(host_a.begin(), host_a.end(), 1.f);

  std::vector<float> host_b(b_size);
  std::fill(host_b.begin(), host_b.end(), 1.f);

  std::vector<float> host_c(c_size);
  std::fill(host_c.begin(), host_c.end(), 0.f);

  float* d_a = nullptr;
  float* d_b = nullptr;
  float* d_c = nullptr;

  HIP_ERROR_ASSERT(hipMalloc(&d_a, a_size * sizeof(float)));
  HIP_ERROR_ASSERT(hipMalloc(&d_b, b_size * sizeof(float)));
  HIP_ERROR_ASSERT(hipMalloc(&d_c, c_size * sizeof(float)));
  HIP_ERROR_ASSERT(hipMemcpy(d_a, host_a.data(), a_size * sizeof(float),
                             hipMemcpyHostToDevice));
  HIP_ERROR_ASSERT(hipMemcpy(d_b, host_b.data(), b_size * sizeof(float),
                             hipMemcpyHostToDevice));
  HIP_ERROR_ASSERT(hipMemset(d_c, 0, c_size * sizeof(float)));

  dim3 grid(1, 1, 1);
  dim3 block(256);

  gemmStatic<<<grid, block, 0, 0>>>(d_a, d_b, d_c);

  HIP_ERROR_ASSERT(hipDeviceSynchronize())
  HIP_ERROR_ASSERT(hipGetLastError());

  HIP_ERROR_ASSERT(hipMemcpy(host_c.data(), d_c, c_size * sizeof(float),
                             hipMemcpyDeviceToHost));

  // Free device memory.
  HIP_ERROR_ASSERT(hipFree(d_a));
  HIP_ERROR_ASSERT(hipFree(d_b));
  HIP_ERROR_ASSERT(hipFree(d_c));

  CheckConstValBuffer(host_c.data(), host_c.size(), 2048.0f);
}
