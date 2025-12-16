#include <hip/hip_runtime.h>

#include "src/hip/common/runtimeConfig.h"
#include "src/hip/kernel/tasks/internal/gateBlock.h"
#include "src/hip/kernel/tasks/internal/gemmImpl/gemmImplSelector.h"
#include "test/cpp/moe/reference/refMoeGate.h"
#include "test/cpp/utils/utils.h"

namespace {
constexpr int M_SIZE = 48;
constexpr int N_SIZE =
    moe::MoeRuntimeConfig::MOE_METADATA::MOE_PROBLEM_CONFIG::EXPERTS_NUM;
constexpr int K_SIZE =
    moe::MoeRuntimeConfig::MOE_METADATA::MOE_PROBLEM_CONFIG::HIDDEN_SIZE;
constexpr int TOPK =
    moe::MoeRuntimeConfig::MOE_METADATA::MOE_PROBLEM_CONFIG::TOPK;
constexpr int THREADS = moe::MoeRuntimeConfig::MOE_METADATA::HW_CONFIG::THREADS;
constexpr int BLOCKS = moe::MoeRuntimeConfig::MOE_METADATA::HW_CONFIG::BLOCKS;
using T_GATE_TILE_IMPL =
    moe::MoeRuntimeConfig::GEMM_RUNTIME_CONFIG::GATE_GEMM_TILE_IMPL;

template <typename TILE_GEMM_IMPL>
__global__ void moeGateKernel(const float* __restrict__ tokens,
                              const float* __restrict__ gateWeights,
                              float* __restrict__ topkVals,
                              int* __restrict__ topIdx, int M_SIZE) {
  __shared__ char sharedMemPool[TILE_GEMM_IMPL::NeededSharedMemBytes()];

  for (int tokenIdx = blockIdx.x; tokenIdx < M_SIZE; tokenIdx += gridDim.x) {
    float* topkVals_shared = nullptr;
    int* topkIdx_shared = nullptr;

    moe::tasks::internal::gate_block<TILE_GEMM_IMPL, TOPK>(
        tokens, gateWeights, tokenIdx, &topkVals_shared, &topkIdx_shared,
        sharedMemPool);

    // Write to output:
    const int tid = threadIdx.x;
    if (threadIdx.x < TOPK) {
      topkVals[tokenIdx * TOPK + tid] = topkVals_shared[tid];
      topIdx[tokenIdx * TOPK + tid] = topkIdx_shared[tid];
    }
  }
}

hipError_t moeGate(const float* A, const float* B, float* topkVals,
                   int* topkIdx, int M, int N, int K) {
  constexpr int m_tile_size = 1;
  assert(K == K_SIZE);
  assert(M == M_SIZE);
  assert(N == N_SIZE);

  dim3 grid(BLOCKS);
  dim3 block(THREADS);

  moeGateKernel<T_GATE_TILE_IMPL>
      <<<grid, block, 0, 0>>>(A, B, topkVals, topkIdx, M);

  return hipSuccess;
}

TEST(MoeTasksInternalTests, MoeGate_basicTest) {
  constexpr int a_size = M_SIZE * K_SIZE;
  constexpr int b_size = K_SIZE * N_SIZE;
  constexpr int c_size = M_SIZE * N_SIZE;
  constexpr int topk_size = M_SIZE * TOPK;

  std::vector<float> host_a = RandomVector<float>(a_size, -0.1f, 0.1f);
  std::vector<float> host_b = RandomVector<float>(b_size, -0.1f, 0.1f);
  std::vector<float> host_topkVals(topk_size);
  std::vector<int> host_topkIdx(topk_size);

  float* d_a = nullptr;
  float* d_b = nullptr;
  float* d_topkVals = nullptr;
  int* d_topkIdx = nullptr;

  HIP_ERROR_ASSERT(hipMalloc(&d_a, a_size * sizeof(float)));
  HIP_ERROR_ASSERT(hipMalloc(&d_b, b_size * sizeof(float)));
  HIP_ERROR_ASSERT(hipMalloc(&d_topkVals, topk_size * sizeof(float)));
  HIP_ERROR_ASSERT(hipMalloc(&d_topkIdx, topk_size * sizeof(int)));

  HIP_ERROR_ASSERT(hipMemcpy(d_a, host_a.data(), a_size * sizeof(float),
                             hipMemcpyHostToDevice));
  HIP_ERROR_ASSERT(hipMemcpy(d_b, host_b.data(), b_size * sizeof(float),
                             hipMemcpyHostToDevice));
  HIP_ERROR_ASSERT(
      moeGate(d_a, d_b, d_topkVals, d_topkIdx, M_SIZE, N_SIZE, K_SIZE));

  HIP_ERROR_ASSERT(hipDeviceSynchronize())
  HIP_ERROR_ASSERT(hipGetLastError());

  HIP_ERROR_ASSERT(hipMemcpy(host_topkVals.data(), d_topkVals,
                             topk_size * sizeof(float), hipMemcpyDeviceToHost));
  HIP_ERROR_ASSERT(hipMemcpy(host_topkIdx.data(), d_topkIdx,
                             topk_size * sizeof(int), hipMemcpyDeviceToHost));

  // Free device memory.
  HIP_ERROR_ASSERT(hipFree(d_a));
  HIP_ERROR_ASSERT(hipFree(d_b));
  HIP_ERROR_ASSERT(hipFree(d_topkVals));
  HIP_ERROR_ASSERT(hipFree(d_topkIdx));

  auto refOut = test::refMoeGate(host_a.data(), host_b.data(), M_SIZE, N_SIZE,
                                 K_SIZE, TOPK);

  CheckAgainstRefBuffer(host_topkVals, refOut.topkVals);
  CheckAgainstRefBuffer(host_topkIdx, refOut.topkIdx);
}
}  // namespace