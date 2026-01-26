#include <vector>

#include "gtest/gtest.h"
#include "include/iMoeKernelLauncher.h"
#include "src/hip/common/metadata.h"
#include "test/singleGPU/cpp/benchmarks/benchmarkUtils.h"
#include "test/common/utils.h"

namespace {
constexpr float ERROR_ABS = 1e-5f;
constexpr int EXPERTS_NUM =
    moe::MoeImplMetadata::MOE_PROBLEM_CONFIG::EXPERTS_NUM;
constexpr int HIDDEN_SIZE =
    moe::MoeImplMetadata::MOE_PROBLEM_CONFIG::HIDDEN_SIZE;
constexpr int EXPERT_INTERMEDIATE_SIZE =
    moe::MoeImplMetadata::MOE_PROBLEM_CONFIG::EXPERT_INTERMEDIATE_SIZE;

hipError_t run(moe::IMoeKernelLauncher* launcher, const void* tokens,
               void* output, int tokensNum, hipStream_t stream) {
  return launcher->Launch(tokens, output, tokensNum, stream);
}

void PerformBenchmark(int numTokens) {
  const int TOKENS_SIZE = numTokens * HIDDEN_SIZE;
  constexpr int GATE_WEIGHTS_SIZE = HIDDEN_SIZE * EXPERTS_NUM;
  constexpr int EXPERTS_FNN1_WEIGHTS_SIZE =
      EXPERTS_NUM * HIDDEN_SIZE * EXPERT_INTERMEDIATE_SIZE * 2;
  constexpr int EXPERTS_FNN2_WEIGHTS_SIZE =
      EXPERTS_NUM * HIDDEN_SIZE * EXPERT_INTERMEDIATE_SIZE;
  const int FINAL_OUTPUT_SIZE = numTokens * HIDDEN_SIZE;

  std::vector<float> tokens_host =
      RandomVector<float>(TOKENS_SIZE, -1.0f, 1.0f);

  float* tokens_device = nullptr;
  float* gateWeights_device = nullptr;
  float* expertFFN1Weights_device = nullptr;
  float* expertFFN2Weights_device = nullptr;
  float* finalOutput_device = nullptr;

  hipStream_t stream = 0;

  HIP_ERROR_ASSERT(hipMalloc(&tokens_device, numTokens * sizeof(float)));
  HIP_ERROR_ASSERT(
      hipMalloc(&gateWeights_device, GATE_WEIGHTS_SIZE * sizeof(float)));
  HIP_ERROR_ASSERT(hipMalloc(&expertFFN1Weights_device,
                             EXPERTS_FNN1_WEIGHTS_SIZE * sizeof(float)));
  HIP_ERROR_ASSERT(hipMalloc(&expertFFN2Weights_device,
                             EXPERTS_FNN2_WEIGHTS_SIZE * sizeof(float)));
  const moe::DistributedUniqueId duid = GetDistributedUniqueId(/*empty=*/true);
  moe::IMoeKernelLauncher* launcher = nullptr;
  HIP_ERROR_ASSERT(
      CreateLauncher(&launcher, gateWeights_device, expertFFN1Weights_device,
                     expertFFN2Weights_device, numTokens, stream, duid, 0, 1));
  HIP_ERROR_ASSERT(
      hipMalloc(&finalOutput_device, FINAL_OUTPUT_SIZE * sizeof(float)));

  HIP_ERROR_ASSERT(hipMemcpy(tokens_device, tokens_host.data(),
                             numTokens * sizeof(float), hipMemcpyHostToDevice));

  const float avg_ms =
      test::bench::Benchmark(10, 10, run, launcher, tokens_device,
                             finalOutput_device, numTokens, stream);

  HIP_ERROR_ASSERT(hipDeviceSynchronize())
  HIP_ERROR_ASSERT(hipGetLastError());

  // Free device memory.
  HIP_ERROR_ASSERT(hipFree(tokens_device));
  HIP_ERROR_ASSERT(hipFree(gateWeights_device));
  HIP_ERROR_ASSERT(hipFree(expertFFN1Weights_device));
  HIP_ERROR_ASSERT(hipFree(expertFFN2Weights_device));
  HIP_ERROR_ASSERT(hipFree(finalOutput_device));

  HIP_ERROR_ASSERT(DestroyLauncher(launcher, stream));

  std::cout << "Moe kernel avg time for num of tokens: " << numTokens << " is "
            << avg_ms << " ms." << std::endl;
}

TEST(Benchmark, FlashMoeSingleGPU) {
  PerformBenchmark(1);
  PerformBenchmark(16);
  PerformBenchmark(32);
}
}  // namespace