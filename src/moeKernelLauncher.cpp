#include "hip/launcher/moeKernelLauncher.h"

#include <rocshmem/rocshmem.hpp>

#include "hip/hipHostCommon.h"
#include "include/iMoeKernelLauncher.h"

#define ROCSHMEM_ERROR_ASSERT(condition...)          \
  {                                                  \
    const int error = (condition);                   \
    if (error != 0) {                                \
      printf("ROCShMem failed on %s\n", #condition); \
    }                                                \
  }

namespace moe {

////////////////////////////////////////////////////////////////////
IMoeKernelLauncher::~IMoeKernelLauncher() {}

}  // namespace moe

////////////////////////////////////////////////////////////////////
extern "C" moe::DistributedUniqueId getDistributedUniqueId() {
  rocshmem::rocshmem_uniqueid_t uid;
  ROCSHMEM_ERROR_ASSERT(rocshmem::rocshmem_get_uniqueid(&uid));

  static_assert(sizeof(rocshmem::rocshmem_uniqueid_t) == sizeof(moe::DistributedUniqueId));

  moe::DistributedUniqueId dist_uid;
  memcpy(dist_uid.data, uid.data(), sizeof(dist_uid.data));
  return dist_uid;
}

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