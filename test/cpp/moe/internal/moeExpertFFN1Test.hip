#include <hip/hip_runtime.h>

#include "src/hip/hipHostCommon.h"
#include "src/hip/tasks/internal/expertFFN1Block.h"
#include "src/hip/utils/hipDeviceUtils.h"
#include "test/cpp/moe/reference/refExpertFFN1.h"
#include "test/cpp/utils/utils.h"

namespace {
constexpr int TOKENS_NUM = 48;
constexpr int EXPERTS_NUM = 128;
constexpr int HIDDEN_SIZE = 2048;
constexpr int EXPERT_INTERMEDIATE_SIZE = 768;
constexpr int TOPK = 8;
constexpr int THREADS = 128;
constexpr int BLOCKS = 304;

constexpr int EXPERT_TILE_SIZE = 256;
using TExpertGemmTileParams =
    GemmTileParams<EXPERT_INTERMEDIATE_SIZE, HIDDEN_SIZE, 1, EXPERT_TILE_SIZE,
                   32, THREADS, float>;

__global__ void hipExpertFFN1Kernel(const float* __restrict__ tokens,
                                    const float* __restrict__ expertWeights,
                                    float* __restrict__ output) {
  __shared__ char sharedMemPool[TExpertGemmTileParams::SHARED_MEM_NEEDES_BYTES];

  constexpr int N_CHUNKS =
      TExpertGemmTileParams::N / TExpertGemmTileParams::TILE_N;

  const int tokenIdx = blockIdx.x / N_CHUNKS;
  const int tileCol = blockIdx.x % N_CHUNKS;

  typename TExpertGemmTileParams::TType
      outRegs[TExpertGemmTileParams::THREAD_OUTPUT_SIZE];
  moe::tasks::internal::expertFFN1_block<TExpertGemmTileParams>(
      tokens, expertWeights, outRegs, tokenIdx, tileCol, sharedMemPool);
  moe::tasks::internal::WriteGemmTileToGlobalMem_block<TExpertGemmTileParams>(
      outRegs, output, tokenIdx, tileCol);
}

static hipError_t hipExpertFFN1(const float* tokens, const float* expertWeights,
                                float* output, int tokensNum,
                                int expertIntermediateSize, int hiddenSize) {
  assert(hiddenSize == HIDDEN_SIZE);
  assert(tokensNum == TOKENS_NUM);
  assert(expertIntermediateSize == EXPERT_INTERMEDIATE_SIZE);
  dim3 grid(tokensNum *
            (expertIntermediateSize / TExpertGemmTileParams::TILE_N));
  dim3 block(TExpertGemmTileParams::THREADS);
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

  CheckAgainstRefBuffer(output_host, refOut);
}
}  // namespace