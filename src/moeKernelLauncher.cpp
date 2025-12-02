#include "hip/launcher/moeKernelLauncher.h"

#include "hip/hipHostCommon.h"
#include "include/iMoeKernelLauncher.h"

namespace moe {

////////////////////////////////////////////////////////////////////
IMoeKernelLauncher::~IMoeKernelLauncher() {}

}  // namespace moe

////////////////////////////////////////////////////////////////////
extern "C" hipError_t CreateLauncher(moe::IMoeKernelLauncher** launcher,
                                     const void* gateWeights,
                                     const void* ffn1ExpertWeights,
                                     const void* ffn2ExpertWeights,
                                     int maxTokens, hipStream_t stream) {
  moe::MoeKernelLauncher* _launcher = new moe::MoeKernelLauncher();
  HIP_ERROR_CHECK(_launcher->Init(gateWeights, ffn1ExpertWeights,
                                  ffn2ExpertWeights, maxTokens, stream));
  *launcher = _launcher;
  return hipSuccess;
}

////////////////////////////////////////////////////////////////////
extern "C" hipError_t DestroyLauncher(moe::IMoeKernelLauncher* launcher,
                                      hipStream_t stream) {
  HIP_ERROR_CHECK(
      static_cast<moe::MoeKernelLauncher*>(launcher)->Deinit(stream));
  delete launcher;
  return hipSuccess;
}