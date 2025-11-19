#pragma once

#include <hip/hip_runtime.h>

namespace moe {

// Main interface for moe kernel launcher.
class IMoeKernelLauncher {
 public:
  // Launches the kernel.
  virtual hipError_t Launch(const float* tokens, const float* gateWeights,
                            const float* ffn1ExpertWeights,
                            const float* ffn2ExpertWeights, float* output,
                            int tokensNum, hipStream_t stream) = 0;

  // Dtor
  virtual ~IMoeKernelLauncher();
};

}  // namespace moe

// Creates launcher.
extern "C" hipError_t CreateLauncher(moe::IMoeKernelLauncher** launcher, hipStream_t stream);

// Destroys launcher.
extern "C" hipError_t DestroyLauncher(moe::IMoeKernelLauncher* launcher, hipStream_t stream);
