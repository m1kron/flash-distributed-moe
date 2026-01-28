#include "hip/launcher/moeKernelLauncher.h"

#include "hip/hipHostCommon.h"
#include "include/iMoeKernelLauncher.h"
#include "src/hip/utils/rocshmem.h"

namespace moe {

////////////////////////////////////////////////////////////////////
IMoeKernelLauncher::~IMoeKernelLauncher() {}

}  // namespace moe

////////////////////////////////////////////////////////////////////
extern "C" moe::DistributedUniqueId GetDistributedUniqueId(bool empty) {
  moe::DistributedUniqueId dist_uid;
  if (!empty) {
    rocshmem::rocshmem_uniqueid_t uid;
    if (rocshmem::rocshmem_get_uniqueid(&uid) != 0)
      MOE_ERROR_LOG("rocshmem_get_uniqueid failed");

    static_assert(sizeof(rocshmem::rocshmem_uniqueid_t) ==
                  sizeof(moe::DistributedUniqueId));

    memcpy(dist_uid.data, uid.data(), sizeof(dist_uid.data));
  } else {
    memset(dist_uid.data, 0, sizeof(dist_uid.data));
  }
  return dist_uid;
}

////////////////////////////////////////////////////////////////////
extern "C" hipError_t CreateLauncher(moe::IMoeKernelLauncher** launcher,
                                     const void* gateWeights,
                                     const void* ffn1ExpertWeights,
                                     const void* ffn2ExpertWeights,
                                     int maxTokens, hipStream_t stream,
                                     const moe::DistributedUniqueId& uid,
                                     int rank, int worldSize) {
  moe::MoeKernelLauncher* _launcher = new moe::MoeKernelLauncher();
  HIP_ERROR_CHECK(_launcher->Init(gateWeights, ffn1ExpertWeights,
                                  ffn2ExpertWeights, maxTokens, stream, uid,
                                  rank, worldSize));
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