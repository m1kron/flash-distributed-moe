#include <hip/hip_runtime.h>

#include "../../../src/hip/tasks/internal/gemmTileBlock.h"
#include "../../../src/hip/tasks/internal/softmaxBlock.h"
#include "../../../src/hip/tasks/internal/topkBlock.h"
#include "../reference/refGemm.h"
#include "../reference/refSoftmax.h"
#include "../reference/refTopk.h"
#include "../utils/utils.h"

template <int M, int N>
__global__ void refHipKernel(const float* __restrict__ A,
                             const float* __restrict__ B,
                             float* __restrict__ topkVals,
                             int* __restrict__ topIdx) {
  using TGemmTileParams = GemmTileParams<N, 2048, 1, 256, 32>;

  typename TGemmTileParams::TOutputType
      out_regs[TGemmTileParams::THREAD_OUTPUT_SIZE];

  GemmTile_block<TGemmTileParams>(A, B, out_regs, blockIdx.x, blockIdx.y);
  out_regs[0] = Softmax256_block(out_regs[0]);
  Topk8_256_block(out_regs[0], topkVals + blockIdx.x * 8,
                  topIdx + blockIdx.x * 8);
}

hipError_t refHipGemm(const float* A, const float* B, float* topkVals,
                      int* topkIdx, int M, int N, int K) {
  constexpr int m_size = 48;
  constexpr int n_size = 256;
  const int k_size = 2048;
  const int m_tile_size = 1;
  const int n_tile_size = 256;
  assert(K == 2048);
  assert(M == m_size);
  assert(N == n_size);

  dim3 grid(M / m_tile_size, N / n_tile_size, 1);
  dim3 block(256);

  refHipKernel<m_size, n_size><<<grid, block, 0, 0>>>(A, B, topkVals, topkIdx);

  return hipSuccess;
}

template <typename TFunc>
void PerformGemmCorrectnessTest2(const TFunc func) {
  // Prepare input buffers for module
  constexpr int m_size = 48;
  constexpr int n_size = 256;
  constexpr int k_size = 2048;
  constexpr int topk = 8;

  constexpr int a_size = m_size * k_size;
  constexpr int b_size = k_size * n_size;
  constexpr int c_size = m_size * n_size;
  constexpr int topk_size = m_size * topk;

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
      func(d_a, d_b, d_topkVals, d_topkIdx, m_size, n_size, k_size));

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

  auto ref =
      test::refGemm(host_a.data(), host_b.data(), m_size, n_size, k_size);

  test::refSoftmaxRowOnlineInplace(ref.data(), m_size, n_size);

  std::vector<float> topkVals_ref(topk_size, 0.0f);
  std::vector<int> topkIdx_ref(topk_size, 0);

  test::refTopK8PerRow(ref.data(), m_size, n_size, topkVals_ref.data(),
                       topkIdx_ref.data());

  CheckAgainstRefBuffer(host_topkVals.data(), topkVals_ref.data(),
                        topkVals_ref.size());
  CheckAgainstRefBuffer(host_topkIdx.data(), topkIdx_ref.data(),
                        topkIdx_ref.size());
}

TEST(TaskSystemUnittests, refHipGemm) {
  PerformGemmCorrectnessTest2(refHipGemm);
}