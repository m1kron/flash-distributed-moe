#pragma once
#include "include/iMoeKernelLauncher.h"

namespace moe {

struct InternalState;

class MoeKernelLauncher : public IMoeKernelLauncher {
 public:
  // IMoeKernelLauncher interface
  virtual hipError_t Launch(const void* tokens, void* output, int tokensNum,
                            hipStream_t stream) override;
  // ----

  // Initializes launcher.
  hipError_t Init(const void* gateWeights, const void* ffn1ExpertWeights,
                  const void* ffn2ExpertWeights, int maxTokens,
                  hipStream_t stream, const moe::DistributedUniqueId& uid,
                  int rank, int worldSize);

  // Deinitializes launcher.
  hipError_t Deinit(hipStream_t stream);

 private:
  InternalState* m_internalState;
};
}  // namespace moe