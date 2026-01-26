#include <hip/hip_runtime.h>

#include "src/hip/common/runtimeConfig.h"
#include "src/hip/hipHostCommon.h"
#include "src/hip/kernel/tasks/internal/expertFFN1Block.h"
#include "src/hip/kernel/tasks/internal/writeGemmTileBlock.h"
#include "src/hip/utils/hipDeviceUtils.h"
#include "test/singleGPU/cpp/moe/reference/refExpertFFN1.h"
#include "test/common/utils.h"

namespace {
constexpr int TOKENS_NUM = 48;
constexpr int HIDDEN_SIZE =
    moe::MoeRuntimeConfig::MOE_METADATA::MOE_PROBLEM_CONFIG::HIDDEN_SIZE;
constexpr int EXPERT_INTERMEDIATE_SIZE = moe::MoeRuntimeConfig::MOE_METADATA::
    MOE_PROBLEM_CONFIG::EXPERT_INTERMEDIATE_SIZE;

constexpr float ABS_ERROR = 1e-4f;

using FFN1_GEMM_TILE_IMPL =
    moe::MoeRuntimeConfig::GEMM_RUNTIME_CONFIG::FFN1_GEMM_TILE_IMPL;

__global__ void hipExpertFFN1Kernel(const float* __restrict__ tokens,
                                    const float* __restrict__ expertWeights,
                                    float* __restrict__ output) {
  __shared__ char sharedMemPool[FFN1_GEMM_TILE_IMPL::NeededSharedMemBytes()];

  constexpr int N_CHUNKS = FFN1_GEMM_TILE_IMPL::TILE_METADATA::N /
                           FFN1_GEMM_TILE_IMPL::TILE_METADATA::TILE_N;

  const int tokenIdx = blockIdx.x / N_CHUNKS;
  const int tileCol = blockIdx.x % N_CHUNKS;

  const float* thisBlockTokens = tokens + tokenIdx * HIDDEN_SIZE;

  typename FFN1_GEMM_TILE_IMPL::TILE_METADATA::TType
      outRegs[FFN1_GEMM_TILE_IMPL::TILE_METADATA::THREAD_OUTPUT_SIZE];
  moe::tasks::internal::expertFFN1_block<FFN1_GEMM_TILE_IMPL>(
      thisBlockTokens, expertWeights, outRegs, 0, tileCol, sharedMemPool);
  moe::tasks::internal::WriteGemmTileToGlobalMem_block<
      FFN1_GEMM_TILE_IMPL::TILE_METADATA>(outRegs, output, tokenIdx, tileCol);
}

static hipError_t hipExpertFFN1(const float* tokens, const float* expertWeights,
                                float* output, int tokensNum,
                                int expertIntermediateSize, int hiddenSize) {
  assert(hiddenSize == HIDDEN_SIZE);
  assert(tokensNum == TOKENS_NUM);
  assert(expertIntermediateSize == EXPERT_INTERMEDIATE_SIZE);
  dim3 grid(tokensNum * (expertIntermediateSize /
                         FFN1_GEMM_TILE_IMPL::TILE_METADATA::TILE_N));
  dim3 block(FFN1_GEMM_TILE_IMPL::TILE_METADATA::THREADS);
  hipExpertFFN1Kernel<<<grid, block, 0, 0>>>(tokens, expertWeights, output);

  return hipSuccess;
}

TEST(MoeTasksInternalTests, MoeExpertFFN1_basicTest) {
  constexpr int TOKENS_SIZE = TOKENS_NUM * HIDDEN_SIZE;
  constexpr int EXPERTS_WEIGHTS_SIZE =
      HIDDEN_SIZE * 2 * EXPERT_INTERMEDIATE_SIZE;
  constexpr int OUTPUT_SIZE = TOKENS_NUM * EXPERT_INTERMEDIATE_SIZE;

  std::vector<float> tokens_host =
      RandomVector<float>(TOKENS_SIZE, -1.0f, 1.0f);
  std::vector<float> expertWeights_host =
      RandomVector<float>(EXPERTS_WEIGHTS_SIZE, -0.1f, 0.1f);
  std::vector<float> output_host(OUTPUT_SIZE, 0.0f);

  float* tokens_device = nullptr;
  float* expertWeights_device = nullptr;
  float* output_device = nullptr;

  HIP_ERROR_ASSERT(hipMalloc(&tokens_device, TOKENS_SIZE * sizeof(float)));
  HIP_ERROR_ASSERT(
      hipMalloc(&expertWeights_device, EXPERTS_WEIGHTS_SIZE * sizeof(float)));
  HIP_ERROR_ASSERT(hipMalloc(&output_device, OUTPUT_SIZE * sizeof(float)));

  HIP_ERROR_ASSERT(hipMemcpy(tokens_device, tokens_host.data(),
                             TOKENS_SIZE * sizeof(float),
                             hipMemcpyHostToDevice));
  HIP_ERROR_ASSERT(hipMemcpy(expertWeights_device, expertWeights_host.data(),
                             EXPERTS_WEIGHTS_SIZE * sizeof(float),
                             hipMemcpyHostToDevice));
  HIP_ERROR_ASSERT(hipExpertFFN1(tokens_device, expertWeights_device,
                                 output_device, TOKENS_NUM,
                                 EXPERT_INTERMEDIATE_SIZE, HIDDEN_SIZE));

  HIP_ERROR_ASSERT(hipDeviceSynchronize())
  HIP_ERROR_ASSERT(hipGetLastError());

  HIP_ERROR_ASSERT(hipMemcpy(output_host.data(), output_device,
                             OUTPUT_SIZE * sizeof(float),
                             hipMemcpyDeviceToHost));

  // Free device memory.
  HIP_ERROR_ASSERT(hipFree(tokens_device));
  HIP_ERROR_ASSERT(hipFree(expertWeights_device));
  HIP_ERROR_ASSERT(hipFree(output_device));

  auto refOut =
      test::refExpertFFN1(tokens_host.data(), expertWeights_host.data(),
                          TOKENS_NUM, HIDDEN_SIZE, EXPERT_INTERMEDIATE_SIZE);

  CheckAgainstRefBuffer(output_host, refOut, ABS_ERROR);
}
}  // namespace