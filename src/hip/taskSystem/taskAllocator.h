#pragma once
#include "../hipHostCommon.h"
#include "../utils/hipDeviceUtils.h"

// Simple and fast linear allocator for tasks.
template <typename T, unsigned int SIZE>
struct TaskAllocator {
  // Initializes allocator on host, allocates memory for SIZE items of type T.
  __host__ hipError_t Init();

  // Deinitializes allocator on host, frees allocated memory.
  __host__ hipError_t Deinit();

  // Allocates new task in device code, returns nullptr if out of memory.
  __device__ T* Allocate();

  // Frees task in device code.
  __device__ void Free(T* task);

  T* m_buffer;
  uint32_t* m_allocatedIdx;
};

/////////////////////////////////////////////////////////////////
//
// IMPLEMENTATION:
//
/////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////
template <typename T, unsigned int SIZE>
inline __host__ hipError_t TaskAllocator<T, SIZE>::Init() {
  HIP_ERROR_CHECK(hipMalloc(&m_allocatedIdx, sizeof(uint32_t)));
  HIP_ERROR_CHECK(hipMemset(m_allocatedIdx, 0, sizeof(uint32_t)));
  HIP_ERROR_CHECK(hipMalloc(&m_buffer, SIZE * sizeof(T)));
  HIP_ERROR_CHECK(hipMemset(m_buffer, 0, SIZE * sizeof(T)));
  return hipSuccess;
}

/////////////////////////////////////////////////////////////////
template <typename T, unsigned int SIZE>
inline __host__ hipError_t TaskAllocator<T, SIZE>::Deinit() {
  HIP_ERROR_CHECK(hipFree(m_buffer));
  HIP_ERROR_CHECK(hipFree(m_allocatedIdx));
  return hipSuccess;
}

/////////////////////////////////////////////////////////////////
template <typename T, unsigned int SIZE>
inline __device__ T* TaskAllocator<T, SIZE>::Allocate() {
  const uint32_t idx = atomicAdd(m_allocatedIdx, 1);
  if (idx >= SIZE) {
    HIP_DEVICE_LOG("TaskAllocator: Out of memory!\n");
    return nullptr;
  }
  return m_buffer + idx;
}

/////////////////////////////////////////////////////////////////
template <typename T, unsigned int SIZE>
inline __device__ void TaskAllocator<T, SIZE>::Free(T* task) {
  // Nothing to do, allocator does not reuse memory.
}
