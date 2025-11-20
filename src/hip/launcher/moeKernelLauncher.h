#pragma once
#include "include/iMoeKernelLauncher.h"

namespace moe {

struct InternalState;

class MoeKernelLauncher : public IMoeKernelLauncher {
 public:
  // IMoeKernelLauncher interface
  virtual hipError_t Launch(const float* tokens, const float* gateWeights,
                            const float* ffn1ExpertWeights,
                            const float* ffn2ExpertWeights, float* output,
                            int tokensNum, hipStream_t stream) override;
  // ----

  // Initializes launcher.
  hipError_t Init(hipStream_t stream, int maxTokens);

  // Deinitializes launcher.
  hipError_t Deinit(hipStream_t stream);

 private:
  InternalState* m_internalState;
};
}  // namespace moe