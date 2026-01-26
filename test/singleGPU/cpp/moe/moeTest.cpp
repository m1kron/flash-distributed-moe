#include "include/iMoeKernelLauncher.h"
#include "src/hip/common/metadata.h"
#include "test/common/moeTools.h"
#include "test/singleGPU/cpp/moe/reference/refFullMoe.h"

namespace {
constexpr float ERROR_ABS = 1e-5f;
constexpr int TOKENS_NUM = 48;
constexpr int EXPERTS_NUM =
    moe::MoeImplMetadata::MOE_PROBLEM_CONFIG::EXPERTS_NUM;
constexpr int HIDDEN_SIZE =
    moe::MoeImplMetadata::MOE_PROBLEM_CONFIG::HIDDEN_SIZE;
constexpr int EXPERT_INTERMEDIATE_SIZE =
    moe::MoeImplMetadata::MOE_PROBLEM_CONFIG::EXPERT_INTERMEDIATE_SIZE;
constexpr int TOPK = moe::MoeImplMetadata::MOE_PROBLEM_CONFIG::TOPK;

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
  constexpr int FINAL_OUTPUT_SIZE = TOKENS_NUM * HIDDEN_SIZE;

  const test::MoeInputCPU input = test::GenerateMoeInputCPU(TOKENS_NUM);

  std::vector<float> output_host(FINAL_OUTPUT_SIZE, 0.0f);

  test::MoeInputGPU gpuAlloc;
  test::AllocateInputGPU(input, gpuAlloc);

  hipStream_t stream = 0;

  const moe::DistributedUniqueId duid = GetDistributedUniqueId(/*empty=*/true);
  moe::IMoeKernelLauncher* launcher = nullptr;
  HIP_ERROR_ASSERT(CreateLauncher(
      &launcher, gpuAlloc.gateWeights_device, gpuAlloc.expertFFN1Weights_device,
      gpuAlloc.expertFFN2Weights_device, TOKENS_NUM + 2, stream, duid, 0, 1));

  HIP_ERROR_ASSERT(launcher->Launch(
      gpuAlloc.tokens_device, gpuAlloc.finalOutput_device, TOKENS_NUM, stream));

  HIP_ERROR_ASSERT(hipDeviceSynchronize())
  HIP_ERROR_ASSERT(hipGetLastError());

  HIP_ERROR_ASSERT(hipMemcpy(output_host.data(), gpuAlloc.finalOutput_device,
                             FINAL_OUTPUT_SIZE * sizeof(float),
                             hipMemcpyDeviceToHost));

  // Free device memory.
  FreeInputGPU(gpuAlloc);

  HIP_ERROR_ASSERT(DestroyLauncher(launcher, stream));

  auto refOut = test::refFullMoe(
      input.tokens_host.data(), input.gateWeights_host.data(),
      input.expertFFN1Weights_host.data(), input.expertFFN2Weights_host.data(),
      TOKENS_NUM, EXPERTS_NUM, HIDDEN_SIZE, TOPK, EXPERT_INTERMEDIATE_SIZE);

  // printResult(refOut.data());

  CheckAgainstRefBuffer(output_host, refOut, ERROR_ABS);
}
}  // namespace