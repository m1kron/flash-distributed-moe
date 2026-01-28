#pragma once
#include "src/hip/utils/hipDeviceUtils.h"

namespace moe {

class RemoteComManager {
 public:
  __host__ void Init(int worldSize);
  __host__ void Deinit();
  __device__ int GetWorldSize() const;

 private:
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
inline __host__ void RemoteComManager::Init(int worldSize) {
  m_worldSize = worldSize;
}

//////////////////////////////////////////////////////////////////
inline __host__ void RemoteComManager::Deinit() {}

//////////////////////////////////////////////////////////////////
inline __device__ int RemoteComManager::GetWorldSize() const {
  return m_worldSize;
}

}  // namespace moe