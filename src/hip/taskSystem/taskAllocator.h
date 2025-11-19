#pragma once
#include "../hipHostCommon.h"
#include "../utils/hipDeviceUtils.h"

// Simple and fast linear allocator for tasks.
template <typename T, unsigned int SIZE>
struct TaskAllocator {
  // Initializes allocator on host, allocates memory for SIZE items of type T.
  __host__ hipError_t Init(hipStream_t stream);

  // Deinitializes allocator on host, frees allocated memory.
  __host__ hipError_t Deinit(hipStream_t stream);

  // Prepares for next launch(clears the state).
  __host__ hipError_t PrepareForNextLaunch(hipStream_t stream);

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
inline __host__ hipError_t TaskAllocator<T, SIZE>::Init(hipStream_t stream) {
  HIP_ERROR_CHECK(hipMallocAsync(&m_allocatedIdx, sizeof(uint32_t), stream));
  HIP_ERROR_CHECK(hipMallocAsync(&m_buffer, SIZE * sizeof(T), stream));

  return hipSuccess;
}

/////////////////////////////////////////////////////////////////
template <typename T, unsigned int SIZE>
inline __host__ hipError_t TaskAllocator<T, SIZE>::Deinit(hipStream_t stream) {
  HIP_ERROR_CHECK(hipFreeAsync(m_buffer, stream));
  HIP_ERROR_CHECK(hipFreeAsync(m_allocatedIdx, stream));
  return hipSuccess;
}

/////////////////////////////////////////////////////////////////
template <typename T, unsigned int SIZE>
inline __host__ hipError_t
TaskAllocator<T, SIZE>::PrepareForNextLaunch(hipStream_t stream) {
  HIP_ERROR_CHECK(hipMemsetAsync(m_allocatedIdx, 0, sizeof(uint32_t), stream));
  HIP_ERROR_CHECK(hipMemsetAsync(m_buffer, 0, SIZE * sizeof(T), stream));
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
