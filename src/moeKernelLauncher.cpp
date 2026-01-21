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

  static_assert(sizeof(rocshmem::rocshmem_uniqueid_t) ==
                sizeof(moe::DistributedUniqueId));

  moe::DistributedUniqueId dist_uid;
  memcpy(dist_uid.data, uid.data(), sizeof(dist_uid.data));
  return dist_uid;
}

////////////////////////////////////////////////////////////////////
extern "C" void InitializeDistributed(const moe::DistributedUniqueId& uid,
                                      int rank, int worldSize) {
  rocshmem::rocshmem_uniqueid_t roc_uid;
  memcpy(roc_uid.data(), uid.data, sizeof(uid.data));

  rocshmem::rocshmem_init_attr_t attr;

  ROCSHMEM_ERROR_ASSERT(rocshmem::rocshmem_set_attr_uniqueid_args(
      rank, worldSize, &roc_uid, &attr));
  ROCSHMEM_ERROR_ASSERT(rocshmem::rocshmem_init_attr(
      rocshmem::ROCSHMEM_INIT_WITH_UNIQUEID, &attr));

  const int rocshmem_rank = rocshmem::rocshmem_my_pe();
  const int rocshmem_size = rocshmem::rocshmem_n_pes();

  int GPU_id;
  hipGetDevice(&GPU_id);

  printf("[Rank %d/%d] Initialized rocSHMEM (rocSHMEM rank %d/%d) on GPU %d\n",
         rank, worldSize, rocshmem_rank, rocshmem_size, GPU_id);

  rocshmem::rocshmem_finalize();
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