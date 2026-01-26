#include "src/hip/common/metadata.h"
#include "test/common/moeTools.h"
#include "test/multiGPU/cpp/moe/moeDistBaseTest.h"
#include "test/singleGPU/cpp/moe/reference/refFullMoe.h"

namespace {
constexpr float ERROR_ABS = 1e-5f;
constexpr int TOKENS_NUM = 1;
constexpr int EXPERTS_NUM =
    moe::MoeImplMetadata::MOE_PROBLEM_CONFIG::EXPERTS_NUM;
constexpr int HIDDEN_SIZE =
    moe::MoeImplMetadata::MOE_PROBLEM_CONFIG::HIDDEN_SIZE;
constexpr int EXPERT_INTERMEDIATE_SIZE =
    moe::MoeImplMetadata::MOE_PROBLEM_CONFIG::EXPERT_INTERMEDIATE_SIZE;
constexpr int TOPK = moe::MoeImplMetadata::MOE_PROBLEM_CONFIG::TOPK;

TEST_F(MoeDistBaseTest, BaseTest) {
  const test::MoeInputCPU inputCPU = test::GenerateMoeInputCPU(TOKENS_NUM);

  auto refOut = test::refFullMoe(
      inputCPU.tokens_host.data(), inputCPU.gateWeights_host.data(),
      inputCPU.expertFFN1Weights_host.data(),
      inputCPU.expertFFN2Weights_host.data(), TOKENS_NUM, EXPERTS_NUM,
      HIDDEN_SIZE, TOPK, EXPERT_INTERMEDIATE_SIZE);

  ExecuteInSeparateProcesses(1, [&inputCPU, &refOut](
                                    const moe::DistributedUniqueId& duid,
                                    int rank, int worldSize) {
    test::MoeInputGPU gpuAlloc;
    test::AllocateInputGPU(inputCPU, gpuAlloc, rank, worldSize);

    hipStream_t stream = 0;
    moe::IMoeKernelLauncher* launcher = nullptr;
    HIP_ERROR_ASSERT(CreateLauncher(
        &launcher, gpuAlloc.gateWeights_device,
        gpuAlloc.expertFFN1Weights_device, gpuAlloc.expertFFN2Weights_device,
        TOKENS_NUM + 2, stream, duid, rank, worldSize));

    HIP_ERROR_ASSERT(launcher->Launch(gpuAlloc.tokens_device,
                                      gpuAlloc.finalOutput_device,
                                      gpuAlloc.tokensNum, stream));

    HIP_ERROR_ASSERT(hipDeviceSynchronize())
    HIP_ERROR_ASSERT(hipGetLastError());

    std::vector<float> output_host(refOut.size(), 0);

    HIP_ERROR_ASSERT(hipMemcpy(output_host.data(), gpuAlloc.finalOutput_device,
                               output_host.size() * sizeof(float),
                               hipMemcpyDeviceToHost));

    // Free device memory.
    FreeInputGPU(gpuAlloc);

    HIP_ERROR_ASSERT(DestroyLauncher(launcher, stream));

    const auto refOutThisRank = test::ShardRefOutput(refOut, rank, worldSize);

    ASSERT_EQ(gpuAlloc.tokensNum, refOutThisRank.size() / HIDDEN_SIZE);

    std::cout << "Out tokens for rank " << rank << " is "
              << refOutThisRank.size() / HIDDEN_SIZE << std::endl;
    // CheckAgainstRefBuffer(output_host, refOutThisRank, ERROR_ABS);
  });
}
}  // namespace