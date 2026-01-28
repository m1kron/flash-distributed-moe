#pragma once
#include "src/hip/utils/hipDeviceUtils.h"
#include "src/hip/common/rocshmem/rocshmem.h"

namespace moe {

class RemoteComManager {
 public:
  __host__ hipError_t Init(const moe::DistributedUniqueId& uid, int rank,
                           int worldSize);
  __host__ hipError_t Deinit();
  __device__ int GetWorldSize() const;

 private:
  __host__ hipError_t InitRocshmem(const moe::DistributedUniqueId& uid,
                                   int rank, int worldSize);

  bool m_isActive = false;
  int m_worldSize = 1;
};

__constant__ RemoteComManager globalRemoteComManager;

//////////////////////////////////////////////////////////////////
//
// IMPLEMENTATION:
//
//////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////
inline __host__ hipError_t RemoteComManager::Init(
    const moe::DistributedUniqueId& uid, int rank, int worldSize) {
  MOE_ASSERT(worldSize > 0);
  m_isActive = worldSize > 1;
  m_worldSize = worldSize;

  if (m_isActive) {
    HIP_ERROR_CHECK(InitRocshmem(uid, rank, worldSize));
  }

  return hipSuccess;
}

//////////////////////////////////////////////////////////////////
inline __host__ hipError_t RemoteComManager::Deinit() {
  if (m_isActive) rocshmem_finalize();
  return hipSuccess;
}

////////////////////////////////////////////////////////////////////
inline __host__ hipError_t RemoteComManager::InitRocshmem(
    const moe::DistributedUniqueId& uid, int rank, int worldSize) {
  rocshmem::rocshmem_uniqueid_t roc_uid;
  memcpy(roc_uid.data(), uid.data, sizeof(uid.data));

  rocshmem::rocshmem_init_attr_t attr;

  if (rocshmem::rocshmem_set_attr_uniqueid_args(rank, worldSize, &roc_uid,
                                                &attr) != 0) {
    MOE_ERROR_LOG("rocshmem_set_attr_uniqueid_args failed");
    return hipErrorUnknown;
  }
  if (rocshmem::rocshmem_init_attr(rocshmem::ROCSHMEM_INIT_WITH_UNIQUEID,
                                   &attr) != 0) {
    MOE_ERROR_LOG("rocshmem_init_attr failed");
    return hipErrorUnknown;
  }

  const int rocshmem_rank = rocshmem::rocshmem_my_pe();
  const int rocshmem_size = rocshmem::rocshmem_n_pes();

  MOE_ASSERT(rocshmem_rank == rank);
  MOE_ASSERT(rocshmem_size == worldSize);

  int GPU_id;
  HIP_ERROR_CHECK(hipGetDevice(&GPU_id));

  MOE_ASSERT(GPU_id == rank);

  MOE_LOG("[Rank %d/%d] Initialized rocSHMEM (rocSHMEM rank %d/%d) on GPU %d\n",
          rank, worldSize, rocshmem_rank, rocshmem_size, GPU_id);

  return hipSuccess;
}

//////////////////////////////////////////////////////////////////
inline __device__ int RemoteComManager::GetWorldSize() const {
  return m_worldSize;
}

}  // namespace moe