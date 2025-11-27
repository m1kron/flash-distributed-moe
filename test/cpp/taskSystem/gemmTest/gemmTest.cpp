#include <hip/hip_runtime.h>

#include "staticGemmKernel.h"
#include "taskSystemGemmKernel.h"
#include "test/cpp/moe/reference/refGemm.h"
#include "test/cpp/utils/utils.h"

namespace {

static constexpr float ABS_ERROR = 1e-3f;

hipError_t taskSystemGemmFunc(const float* A, const float* B, float* C, int M,
                              int N, int K) {
  void* state = nullptr;
  hipError_t error = hipSuccess;
  error = initTaskSystemGemm(&state);
  error = taskSystemGemm(state, A, B, C, M, N, K);
  error = deinitTaskSystemGemm(state);
  return error;
}

template <typename TFunc>
void PerformGemmCorrectnessTest(const TFunc func) {
  // Prepare input buffers for module
  constexpr int m_size = 2048;
  constexpr int n_size = 1536;
  constexpr int k_size = 2048;

  constexpr int a_size = m_size * k_size;
  constexpr int b_size = k_size * n_size;
  constexpr int c_size = m_size * n_size;

  std::vector<float> host_a = RandomVector<float>(a_size);
  std::vector<float> host_b = RandomVector<float>(b_size);
  std::vector<float> host_c = RandomVector<float>(c_size);

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

  HIP_ERROR_ASSERT(func(d_a, d_b, d_c, m_size, n_size, k_size));

  HIP_ERROR_ASSERT(hipDeviceSynchronize())
  HIP_ERROR_ASSERT(hipGetLastError());

  HIP_ERROR_ASSERT(hipMemcpy(host_c.data(), d_c, c_size * sizeof(float),
                             hipMemcpyDeviceToHost));

  // Free device memory.
  HIP_ERROR_ASSERT(hipFree(d_a));
  HIP_ERROR_ASSERT(hipFree(d_b));
  HIP_ERROR_ASSERT(hipFree(d_c));

  auto ref =
      test::refGemm(host_a.data(), host_b.data(), m_size, n_size, k_size);

  CheckAgainstRefBuffer(host_c, ref, ABS_ERROR);
}

TEST(TaskSystemTests, TaskSystemGemm) {
  PerformGemmCorrectnessTest(taskSystemGemmFunc);
}

TEST(TaskSystemTests, RefGemm) { PerformGemmCorrectnessTest(staticGemm); }
}  // namespace