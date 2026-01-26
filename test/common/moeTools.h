#pragma once
#include "src/hip/common/metadata.h"
#include "utils.h"

namespace test {

struct MoeInputCPU {
  std::vector<float> tokens_host;
  std::vector<float> gateWeights_host;
  std::vector<float> expertFFN1Weights_host;
  std::vector<float> expertFFN2Weights_host;
};

// Generates ref cpu input.
MoeInputCPU GenerateMoeInputCPU(int tokensNum);

struct MoeInputGPU {
  const float* tokens_device = nullptr;
  const float* gateWeights_device = nullptr;
  const float* expertFFN1Weights_device = nullptr;
  const float* expertFFN2Weights_device = nullptr;
  float* finalOutput_device = nullptr;
};

void AllocateInputGPU(const MoeInputCPU& cpuInput, MoeInputGPU& gpuInput,
                      int rank, int worldSize);
void FreeInputGPU(MoeInputGPU& gpuInput);

////////////////////////////////////////////////////////////////////////
//
// IMPLEMENTATION:
//
////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////
inline MoeInputCPU GenerateMoeInputCPU(int tokensNum) {
  constexpr int EXPERTS_NUM =
      moe::MoeImplMetadata::MOE_PROBLEM_CONFIG::EXPERTS_NUM;
  constexpr int HIDDEN_SIZE =
      moe::MoeImplMetadata::MOE_PROBLEM_CONFIG::HIDDEN_SIZE;
  constexpr int EXPERT_INTERMEDIATE_SIZE =
      moe::MoeImplMetadata::MOE_PROBLEM_CONFIG::EXPERT_INTERMEDIATE_SIZE;

  const int TOKENS_SIZE = tokensNum * HIDDEN_SIZE;
  constexpr int GATE_WEIGHTS_SIZE = HIDDEN_SIZE * EXPERTS_NUM;
  constexpr int EXPERTS_FNN1_WEIGHTS_SIZE =
      EXPERTS_NUM * HIDDEN_SIZE * EXPERT_INTERMEDIATE_SIZE * 2;
  constexpr int EXPERTS_FNN2_WEIGHTS_SIZE =
      EXPERTS_NUM * HIDDEN_SIZE * EXPERT_INTERMEDIATE_SIZE;

  MoeInputCPU input;
  input.tokens_host = RandomVector<float>(TOKENS_SIZE, -1.0f, 1.0f);
  input.gateWeights_host = RandomVector<float>(GATE_WEIGHTS_SIZE, -0.1f, 0.1f);
  input.expertFFN1Weights_host =
      RandomVector<float>(EXPERTS_FNN1_WEIGHTS_SIZE, -0.1f, 0.1f);
  input.expertFFN2Weights_host =
      RandomVector<float>(EXPERTS_FNN2_WEIGHTS_SIZE, -0.1f, 0.1f);

  return input;
}

////////////////////////////////////////////////////////////////////////
inline void AllocateInputGPU(const MoeInputCPU& cpuInput, MoeInputGPU& gpuInput,
                             int rank, int worldSize) {
  constexpr int EXPERTS_NUM =
      moe::MoeImplMetadata::MOE_PROBLEM_CONFIG::EXPERTS_NUM;
  constexpr int HIDDEN_SIZE =
      moe::MoeImplMetadata::MOE_PROBLEM_CONFIG::HIDDEN_SIZE;
  constexpr int EXPERT_INTERMEDIATE_SIZE =
      moe::MoeImplMetadata::MOE_PROBLEM_CONFIG::EXPERT_INTERMEDIATE_SIZE;

  const int tokensNum = cpuInput.tokens_host.size() / HIDDEN_SIZE;

  // Calculate local (sharded) sizes
  const int localTokensNum = tokensNum / worldSize;
  const int tokenOffset = rank * localTokensNum;

  const int localExpertsNum = EXPERTS_NUM / worldSize;
  const int expertOffset = rank * localExpertsNum;

  // Sharded sizes
  const size_t localTokensSize = localTokensNum * HIDDEN_SIZE * sizeof(float);
  const size_t localFFN1WeightsSize = localExpertsNum * HIDDEN_SIZE *
                                      EXPERT_INTERMEDIATE_SIZE * 2 *
                                      sizeof(float);
  const size_t localFFN2WeightsSize =
      localExpertsNum * HIDDEN_SIZE * EXPERT_INTERMEDIATE_SIZE * sizeof(float);
  const size_t localOutputSize = localTokensNum * HIDDEN_SIZE * sizeof(float);

  // Gate weights are replicated (not sharded)
  const size_t gateWeightsSize =
      cpuInput.gateWeights_host.size() * sizeof(float);

  float* tokens_device = nullptr;
  float* gateWeights_device = nullptr;
  float* expertFFN1Weights_device = nullptr;
  float* expertFFN2Weights_device = nullptr;
  float* finalOutput_device = nullptr;

  HIP_ERROR_ASSERT(hipMalloc(&tokens_device, localTokensSize));
  HIP_ERROR_ASSERT(hipMalloc(&gateWeights_device, gateWeightsSize));
  HIP_ERROR_ASSERT(hipMalloc(&expertFFN1Weights_device, localFFN1WeightsSize));
  HIP_ERROR_ASSERT(hipMalloc(&expertFFN2Weights_device, localFFN2WeightsSize));
  HIP_ERROR_ASSERT(hipMalloc(&finalOutput_device, localOutputSize));

  // Copy sharded tokens
  const float* tokensOffsetPtr =
      cpuInput.tokens_host.data() + tokenOffset * HIDDEN_SIZE;
  HIP_ERROR_ASSERT(hipMemcpy(tokens_device, tokensOffsetPtr, localTokensSize,
                             hipMemcpyHostToDevice));

  // Copy full gate weights (replicated)
  HIP_ERROR_ASSERT(hipMemcpy(gateWeights_device,
                             cpuInput.gateWeights_host.data(), gateWeightsSize,
                             hipMemcpyHostToDevice));

  // Copy sharded FFN1 weights
  const size_t ffn1ExpertStride = HIDDEN_SIZE * EXPERT_INTERMEDIATE_SIZE * 2;
  const float* ffn1OffsetPtr =
      cpuInput.expertFFN1Weights_host.data() + expertOffset * ffn1ExpertStride;
  HIP_ERROR_ASSERT(hipMemcpy(expertFFN1Weights_device, ffn1OffsetPtr,
                             localFFN1WeightsSize, hipMemcpyHostToDevice));

  // Copy sharded FFN2 weights
  const size_t ffn2ExpertStride = HIDDEN_SIZE * EXPERT_INTERMEDIATE_SIZE;
  const float* ffn2OffsetPtr =
      cpuInput.expertFFN2Weights_host.data() + expertOffset * ffn2ExpertStride;
  HIP_ERROR_ASSERT(hipMemcpy(expertFFN2Weights_device, ffn2OffsetPtr,
                             localFFN2WeightsSize, hipMemcpyHostToDevice));

  gpuInput.tokens_device = tokens_device;
  gpuInput.gateWeights_device = gateWeights_device;
  gpuInput.expertFFN1Weights_device = expertFFN1Weights_device;
  gpuInput.expertFFN2Weights_device = expertFFN2Weights_device;
  gpuInput.finalOutput_device = finalOutput_device;
}

////////////////////////////////////////////////////////////////////////
inline void FreeInputGPU(MoeInputGPU& gpuInput) {
  HIP_ERROR_ASSERT(hipFree(const_cast<float*>(gpuInput.tokens_device)));
  HIP_ERROR_ASSERT(hipFree(const_cast<float*>(gpuInput.gateWeights_device)));
  HIP_ERROR_ASSERT(
      hipFree(const_cast<float*>(gpuInput.expertFFN1Weights_device)));
  HIP_ERROR_ASSERT(
      hipFree(const_cast<float*>(gpuInput.expertFFN2Weights_device)));
  HIP_ERROR_ASSERT(hipFree(const_cast<float*>(gpuInput.finalOutput_device)));
}

}  // namespace test