#include <hip/hip_runtime.h>

#include "../../../src/hip/hipHostCommon.h"
#include "../../../src/hip/tasks/internal/gemmTileBlock.h"
#include "../../../src/hip/utils/hipDeviceUtils.h"
#include "../reference/refExpertFFN1.h"
#include "../utils/utils.h"

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
    GemmTileParams<2 * EXPERT_INTERMEDIATE_SIZE, HIDDEN_SIZE, 1,
                   EXPERT_TILE_SIZE, 32, THREADS>;

inline __device__ float Silu(float a, float b) {
  float sig;
  if (a >= 0.0f) {
    sig = 1.0f / (1.0f + expf(-a));
  } else {
    float ea = expf(a);
    sig = ea / (1.0f + ea);
  }

  const float silu = a * sig;
  return silu * b;
}

template <typename TGEMM_TILE>
__global__ void hipExpertFFN1Kernel(const float* __restrict__ tokens,
                                    const float* __restrict__ expertWeights,
                                    float* __restrict__ output) {
  __shared__ char sharedMemPool[TGEMM_TILE::SHARED_MEM_NEEDES_BYTES];

  constexpr int N_CHUNKS = (TGEMM_TILE::N / 2) / TGEMM_TILE::TILE_N;

  const int tokenIdx = blockIdx.x / N_CHUNKS;
  const int tileCol = blockIdx.x % N_CHUNKS;

  typename TGEMM_TILE::TOutputType w1_regs[TGEMM_TILE::THREAD_OUTPUT_SIZE];
  GemmTile_block<TGEMM_TILE>(tokens, expertWeights, w1_regs, tokenIdx, tileCol,
                             sharedMemPool);

  typename TGEMM_TILE::TOutputType w3_regs[TGEMM_TILE::THREAD_OUTPUT_SIZE];
  GemmTile_block<TGEMM_TILE>(tokens, expertWeights, w3_regs, tokenIdx,
                             tileCol + N_CHUNKS, sharedMemPool);

  // Silu:
  for (int i = 0; i < TGEMM_TILE::THREAD_OUTPUT_SIZE; ++i) {
    w3_regs[i] = Silu(w1_regs[i], w3_regs[i]);
  }

  // w3_regs contains output
  using T_OUTPUT_TILE = GemmTileParams<TGEMM_TILE::N / 2, TGEMM_TILE::K,
                                       TGEMM_TILE::TILE_M, TGEMM_TILE::TILE_N,
                                       TGEMM_TILE::TILE_K, TGEMM_TILE::THREADS>;

  WriteGemmTileToGlobalMem_block<T_OUTPUT_TILE>(w3_regs, output, tokenIdx,
                                                tileCol);
}

static hipError_t hipExpertFFN1(const float* tokens, const float* expertWeights,
                                float* output, int tokensNum,
                                int expertIntermediateSize, int hiddenSize) {
  assert(hiddenSize == HIDDEN_SIZE);
  assert(tokensNum == TOKENS_NUM);
  assert(expertIntermediateSize == EXPERT_INTERMEDIATE_SIZE);

  int bla = TExpertGemmTileParams::TILE_N;

  dim3 grid(tokensNum *
            (expertIntermediateSize / TExpertGemmTileParams::TILE_N));
  dim3 block(TExpertGemmTileParams::THREADS);
  hipExpertFFN1Kernel<TExpertGemmTileParams>
      <<<grid, block, 0, 0>>>(tokens, expertWeights, output);

  return hipSuccess;
}

TEST(MoeExpertFFN1, basicTest) {
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

  CheckAgainstRefBuffer(output_host.data(), refOut.data(), refOut.size());
}
}  // namespace