#pragma once

#include <hip/hip_runtime.h>

namespace moe {

// TODO: A POC version of API. Will be subject of changes in the future.
// Main interface for moe kernel launcher.
class IMoeKernelLauncher {
 public:
  // Launches the kernel.
  virtual hipError_t Launch(const void* tokens, void* output, int tokensNum,
                            hipStream_t stream) = 0;

  // Dtor
  virtual ~IMoeKernelLauncher();
};

struct DistributedUniqueId {
  uint8_t data[128];
};

}  // namespace moe

// Gets the distributed unique id. If empty is true, returns an empty unique id.
extern "C" moe::DistributedUniqueId GetDistributedUniqueId(bool empty);

// Creates launcher.
extern "C" hipError_t CreateLauncher(moe::IMoeKernelLauncher** launcher,
                                     const void* gateWeights,
                                     const void* ffn1ExpertWeights,
                                     const void* ffn2ExpertWeights,
                                     int maxTokens, hipStream_t stream,
                                     const moe::DistributedUniqueId& uid,
                                     int rank, int worldSize);

// Destroys launcher.
extern "C" hipError_t DestroyLauncher(moe::IMoeKernelLauncher* launcher,
                                      hipStream_t stream);
