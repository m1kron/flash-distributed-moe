#pragma once

#include <hip/hip_runtime.h>

namespace moe {

// Main interface for moe kernel launcher.
class IMoeKernelLauncher {
 public:
  // Launches the kernel.
  virtual hipError_t Launch(const void* tokens, const void* gateWeights,
                            const void* ffn1ExpertWeights,
                            const void* ffn2ExpertWeights, void* output,
                            int tokensNum, hipStream_t stream) = 0;

  // Dtor
  virtual ~IMoeKernelLauncher();
};

}  // namespace moe

// Creates launcher.
extern "C" hipError_t CreateLauncher(moe::IMoeKernelLauncher** launcher,
                                     hipStream_t stream, int maxTokens);

// Destroys launcher.
extern "C" hipError_t DestroyLauncher(moe::IMoeKernelLauncher* launcher,
                                      hipStream_t stream);
