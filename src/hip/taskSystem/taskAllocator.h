#pragma once
#include "../hipHostCommon.h"
#include "../utils/hipDeviceUtils.h"

template <typename T, unsigned int SIZE>
struct TaskAllocator {
  T* _items;
  uint32_t* _allocatedIdx;

  __host__ hipError_t Init();

  __host__ hipError_t Deinit() {
    HIP_ERROR_CHECK(hipFree(_items));
    HIP_ERROR_CHECK(hipFree(_allocatedIdx));
    return hipSuccess;
  }

  __device__ T* Allocate() {
    const uint32_t idx = atomicAdd(_allocatedIdx, 1);
    if (idx >= SIZE) {
      HIP_DEVICE_LOG("TaskAllocator: Out of memory!\n");
      return nullptr;
    }
    return _items + idx;
  }

  __device__ void Free(T* item);

  __device__ uint32_t GetSize() const { return SIZE; }
};

/////////////////////////////////////////////////////////////////
//
// IMPLEMENTATION:
//
/////////////////////////////////////////////////////////////////

template <typename T, unsigned int SIZE>
inline __device__ void TaskAllocator<T, SIZE>::Free(T* item) {
  // Nothing to do, allocator does not reuse memory.
}

/////////////////////////////////////////////////////////////////
template <typename T, unsigned int SIZE>
inline __host__ hipError_t TaskAllocator<T, SIZE>::Init() {
  HIP_ERROR_CHECK(hipMalloc(&_allocatedIdx, sizeof(uint32_t)));
  HIP_ERROR_CHECK(hipMemset(_allocatedIdx, 0, sizeof(uint32_t)));
  HIP_ERROR_CHECK(hipMalloc(&_items, SIZE * sizeof(T)));
  HIP_ERROR_CHECK(hipMemset(_items, 0, SIZE * sizeof(T)));
  return hipSuccess;
}