#include "src/hip/common/metadata.h"
#include "test/common/moeTools.h"
#include "test/multiGPU/cpp/moe/moeDistBaseTest.h"
#include "test/singleGPU/cpp/moe/reference/refFullMoe.h"

namespace {
constexpr float ERROR_ABS = 1e-5f;

class MoeMultiGPUTests : public MoeDistBaseTest {
 public:
  void Execute(const test::MoeInputCPU& inputCPU, int worldSize) {
    constexpr int EXPERTS_NUM =
        moe::MoeImplMetadata::MOE_PROBLEM_CONFIG::EXPERTS_NUM;
    constexpr int HIDDEN_SIZE =
        moe::MoeImplMetadata::MOE_PROBLEM_CONFIG::HIDDEN_SIZE;
    constexpr int EXPERT_INTERMEDIATE_SIZE =
        moe::MoeImplMetadata::MOE_PROBLEM_CONFIG::EXPERT_INTERMEDIATE_SIZE;
    constexpr int TOPK = moe::MoeImplMetadata::MOE_PROBLEM_CONFIG::TOPK;

    const int tokensNum = inputCPU.tokens_host.size() / HIDDEN_SIZE;

    const auto refOut = test::refFullMoe(
        inputCPU.tokens_host.data(), inputCPU.gateWeights_host.data(),
        inputCPU.expertFFN1Weights_host.data(),
        inputCPU.expertFFN2Weights_host.data(), tokensNum, EXPERTS_NUM,
        HIDDEN_SIZE, TOPK, EXPERT_INTERMEDIATE_SIZE);

    ExecuteInSeparateProcesses(
        worldSize, [&inputCPU, &refOut](const moe::DistributedUniqueId& duid,
                                        int rank, int worldSize) {
          test::MoeInputGPU gpuAlloc;
          test::AllocateInputGPU(inputCPU, gpuAlloc, rank, worldSize);

          std::cout << "Rank " << rank << " will process " << gpuAlloc.tokensNum
                    << " tokens" << std::endl;

          hipStream_t stream = 0;
          moe::IMoeKernelLauncher* launcher = nullptr;
          HIP_ERROR_ASSERT(CreateLauncher(
              &launcher, gpuAlloc.gateWeights_device,
              gpuAlloc.expertFFN1Weights_device,
              gpuAlloc.expertFFN2Weights_device, gpuAlloc.tokensNum + 2, stream,
              duid, rank, worldSize));

          HIP_ERROR_ASSERT(launcher->Launch(gpuAlloc.tokens_device,
                                            gpuAlloc.finalOutput_device,
                                            gpuAlloc.tokensNum, stream));

          HIP_ERROR_ASSERT(hipDeviceSynchronize())
          HIP_ERROR_ASSERT(hipGetLastError());

          std::vector<float> output_host(gpuAlloc.tokensNum * HIDDEN_SIZE, 0);

          HIP_ERROR_ASSERT(hipMemcpy(
              output_host.data(), gpuAlloc.finalOutput_device,
              output_host.size() * sizeof(float), hipMemcpyDeviceToHost));

          FreeInputGPU(gpuAlloc);

          HIP_ERROR_ASSERT(DestroyLauncher(launcher, stream));

          const auto refOutThisRank =
              test::ShardRefOutput(refOut, rank, worldSize);

          ASSERT_EQ(gpuAlloc.tokensNum, refOutThisRank.size() / HIDDEN_SIZE);

          CheckAgainstRefBuffer(output_host, refOutThisRank, ERROR_ABS);
        });
  }
};

TEST_F(MoeMultiGPUTests, DISABLED_worldSize1_tokens1) {
  const int tokensNum = 1;
  const int worldSize = 1;
  const test::MoeInputCPU inputCPU = test::GenerateMoeInputCPU(tokensNum);

  Execute(inputCPU, worldSize);
}

TEST_F(MoeMultiGPUTests, worldSize4_tokensLocalForGPU) {
  const int tokensNum = 3;
  const int worldSize = 4;
  test::MoeInputCPU inputCPU = test::GenerateMoeInputCPU(tokensNum);

  test::SetTokenExpertRouting(inputCPU,
                              {{0, {0, 1, 2, 3, 4, 5, 6, 7}},
                               {1, {33, 34, 35, 36, 37, 38, 39, 40}},
                               {2, {64, 65, 66, 67, 68, 69, 70, 71}}});

  Execute(inputCPU, worldSize);
}

}  // namespace