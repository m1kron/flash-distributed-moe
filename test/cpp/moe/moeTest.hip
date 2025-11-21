#include <hip/hip_runtime.h>

#include "include/iMoeKernelLauncher.h"
#include "test/cpp/moe/reference/refFullMoe.h"
#include "test/cpp/utils/utils.h"

namespace {
constexpr int TOKENS_NUM = 48;
constexpr int EXPERTS_NUM = 128;
constexpr int HIDDEN_SIZE = 2048;
constexpr int EXPERT_INTERMEDIATE_SIZE = 768;
constexpr int TOPK = 8;
constexpr int THREADS = 128;
constexpr int BLOCKS = 304;
constexpr int REDUCTION_TILE_SIZE = 512;
constexpr int REDUCTION_CHUNKS_PER_TOKEN = (HIDDEN_SIZE / REDUCTION_TILE_SIZE);

static void printResult(const float* result) {
  for (int tokenIdx = 0; tokenIdx < TOKENS_NUM; ++tokenIdx) {
    std::cout << "TokenIdx: " << tokenIdx << std::endl;
    for (int j = 0; j < 10; ++j) {
      std::cout << "|" << result[tokenIdx * HIDDEN_SIZE + j];
    }

    std::cout << std::endl;
  }
}

TEST(MoeTests, basic) {
  constexpr int TOKENS_SIZE = TOKENS_NUM * HIDDEN_SIZE;
  constexpr int GATE_WEIGHTS_SIZE = HIDDEN_SIZE * EXPERTS_NUM;
  constexpr int EXPERTS_FNN1_WEIGHTS_SIZE =
      EXPERTS_NUM * HIDDEN_SIZE * EXPERT_INTERMEDIATE_SIZE * 2;
  constexpr int EXPERTS_FNN2_WEIGHTS_SIZE =
      EXPERTS_NUM * HIDDEN_SIZE * EXPERT_INTERMEDIATE_SIZE;
  constexpr int FINAL_OUTPUT_SIZE = TOKENS_NUM * HIDDEN_SIZE;

  std::vector<float> tokens_host =
      RandomVector<float>(TOKENS_SIZE, -1.0f, 1.0f);
  std::vector<float> gateWeights_host =
      RandomVector<float>(GATE_WEIGHTS_SIZE, -0.1f, 0.1f);
  std::vector<float> expertFFN1Weights_host =
      RandomVector<float>(EXPERTS_FNN1_WEIGHTS_SIZE, -0.1f, 0.1f);
  std::vector<float> expertFFN2Weights_host =
      RandomVector<float>(EXPERTS_FNN2_WEIGHTS_SIZE, -0.1f, 0.1f);

  std::vector<float> output_host(FINAL_OUTPUT_SIZE, 0.0f);

  float* tokens_device = nullptr;
  float* gateWeights_device = nullptr;
  float* expertFFN1Weights_device = nullptr;
  float* expertFFN2Weights_device = nullptr;
  float* finalOutput_device = nullptr;

  hipStream_t stream = 0;
  moe::IMoeKernelLauncher* launcher = nullptr;
  HIP_ERROR_ASSERT(CreateLauncher(&launcher, stream, 2 * TOKENS_NUM));

  HIP_ERROR_ASSERT(hipMalloc(&tokens_device, TOKENS_SIZE * sizeof(float)));
  HIP_ERROR_ASSERT(
      hipMalloc(&gateWeights_device, GATE_WEIGHTS_SIZE * sizeof(float)));
  HIP_ERROR_ASSERT(hipMalloc(&expertFFN1Weights_device,
                             EXPERTS_FNN1_WEIGHTS_SIZE * sizeof(float)));
  HIP_ERROR_ASSERT(hipMalloc(&expertFFN2Weights_device,
                             EXPERTS_FNN2_WEIGHTS_SIZE * sizeof(float)));
  HIP_ERROR_ASSERT(
      hipMalloc(&finalOutput_device, FINAL_OUTPUT_SIZE * sizeof(float)));

  HIP_ERROR_ASSERT(hipMemcpy(tokens_device, tokens_host.data(),
                             TOKENS_SIZE * sizeof(float),
                             hipMemcpyHostToDevice));
  HIP_ERROR_ASSERT(hipMemcpy(gateWeights_device, gateWeights_host.data(),
                             GATE_WEIGHTS_SIZE * sizeof(float),
                             hipMemcpyHostToDevice));
  HIP_ERROR_ASSERT(hipMemcpy(
      expertFFN1Weights_device, expertFFN1Weights_host.data(),
      EXPERTS_FNN1_WEIGHTS_SIZE * sizeof(float), hipMemcpyHostToDevice));
  HIP_ERROR_ASSERT(hipMemcpy(
      expertFFN2Weights_device, expertFFN2Weights_host.data(),
      EXPERTS_FNN2_WEIGHTS_SIZE * sizeof(float), hipMemcpyHostToDevice));

  HIP_ERROR_ASSERT(launcher->Launch(
      tokens_device, gateWeights_device, expertFFN1Weights_device,
      expertFFN2Weights_device, finalOutput_device, TOKENS_NUM, stream));

  HIP_ERROR_ASSERT(hipDeviceSynchronize())
  HIP_ERROR_ASSERT(hipGetLastError());

  HIP_ERROR_ASSERT(hipMemcpy(output_host.data(), finalOutput_device,
                             FINAL_OUTPUT_SIZE * sizeof(float),
                             hipMemcpyDeviceToHost));

  // Free device memory.
  HIP_ERROR_ASSERT(hipFree(tokens_device));
  HIP_ERROR_ASSERT(hipFree(gateWeights_device));
  HIP_ERROR_ASSERT(hipFree(expertFFN1Weights_device));
  HIP_ERROR_ASSERT(hipFree(expertFFN2Weights_device));
  HIP_ERROR_ASSERT(hipFree(finalOutput_device));

  HIP_ERROR_ASSERT(DestroyLauncher(launcher, stream));

  auto refOut = test::refFullMoe(
      tokens_host.data(), gateWeights_host.data(),
      expertFFN1Weights_host.data(), expertFFN2Weights_host.data(), TOKENS_NUM,
      EXPERTS_NUM, HIDDEN_SIZE, TOPK, EXPERT_INTERMEDIATE_SIZE);

  // printResult(refOut.data());

  ASSERT_EQ(output_host.size(), refOut.size());
  CheckAgainstRefBuffer(output_host.data(), refOut.data(), refOut.size(),
                        1e-5f);
}
}  // namespace