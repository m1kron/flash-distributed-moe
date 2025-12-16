#pragma once
#include "../hipHostCommon.h"
#include "../utils/hipDeviceUtils.h"

// Lockfree WorkQueue implementation: The main idea was to maximally reduce
// possible contention between threads. The implementation does not use CAS
// operation and does not reuse slots.
template <typename T, unsigned int SIZE>
struct WorkQueue {
  struct __align__(16) QueueSlot {
    int isComitted;
    T* data;
  };

  // Initializes queue on host, allocates memory for SIZE items of type T.
  __host__ hipError_t Init(hipStream_t stream);

  // Deinitializes queue on host, frees allocated memory.
  __host__ hipError_t Deinit(hipStream_t stream);

  // Prepares for next launch(clears the state).
  __host__ hipError_t PrepareForNextLaunch(hipStream_t stream);

  // Pushes item to the queue in device code, returns false if out of slots.
  __device__ bool Push(T* item_global_ptr);

  // Reserves slot ticket for popping item from the queue. Returns UINT_MAX if
  // there are no more slots to reserve(all used).
  __device__ unsigned int ReserveSlotTicket();

  // Tries to pop item from the queue for given slot ticket. Returns false if no
  // item is available for now.
  __device__ bool TryToPop(unsigned int slotTicket, T*& outItem);

  QueueSlot* m_workQueue;
  unsigned int* m_headIdx;
  unsigned int* m_tailIdx;
};

/////////////////////////////////////////////////////////////////
//
// IMPLEMENTATION:
//
/////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////
template <typename T, unsigned int SIZE>
inline __host__ hipError_t WorkQueue<T, SIZE>::Init(hipStream_t stream) {
  HIP_ERROR_CHECK(
      hipMallocAsync(&m_workQueue, SIZE * sizeof(QueueSlot), stream));
  HIP_ERROR_CHECK(hipMallocAsync(&m_headIdx, sizeof(int*), stream));
  HIP_ERROR_CHECK(hipMallocAsync(&m_tailIdx, sizeof(int*), stream));
  return hipSuccess;
}

/////////////////////////////////////////////////////////////////
template <typename T, unsigned int SIZE>
inline __host__ hipError_t WorkQueue<T, SIZE>::Deinit(hipStream_t stream) {
  HIP_ERROR_CHECK(hipFreeAsync(m_tailIdx, stream));
  HIP_ERROR_CHECK(hipFreeAsync(m_headIdx, stream));
  HIP_ERROR_CHECK(hipFreeAsync(m_workQueue, stream));
  return hipSuccess;
}

/////////////////////////////////////////////////////////////////
template <typename T, unsigned int SIZE>
inline __host__ hipError_t
WorkQueue<T, SIZE>::PrepareForNextLaunch(hipStream_t stream) {
  HIP_ERROR_CHECK(
      hipMemsetAsync(m_workQueue, 0, SIZE * sizeof(QueueSlot), stream));
  HIP_ERROR_CHECK(hipMemsetAsync(m_headIdx, 0, sizeof(unsigned int), stream));
  HIP_ERROR_CHECK(hipMemsetAsync(m_tailIdx, 0, sizeof(unsigned int), stream));
  return hipSuccess;
}

/////////////////////////////////////////////////////////////////
template <typename T, unsigned int SIZE>
inline __device__ bool WorkQueue<T, SIZE>::Push(T* item_global_ptr) {
  unsigned int slotIdx = atomicAdd(m_headIdx, 1);

  if (slotIdx >= SIZE) {
    HIP_DEVICE_LOG("Out of slots!\n");
    return false;
  }

  QueueSlot* slot = &m_workQueue[slotIdx];
  slot->data = item_global_ptr;

  __threadfence();
  const int prev = atomicExch(&slot->isComitted, 1);
  HIP_DEVICE_ASSERT(prev == 0);

  return true;
}

/////////////////////////////////////////////////////////////////
template <typename T, unsigned int SIZE>
inline __device__ unsigned int WorkQueue<T, SIZE>::ReserveSlotTicket() {
  const unsigned int slotIdx = atomicAdd(m_tailIdx, 1);
  if (slotIdx >= SIZE) {
    HIP_DEVICE_LOG("Out of slots!\n");
    return UINT_MAX;
  }
  return slotIdx;
}

/////////////////////////////////////////////////////////////////
template <typename T, unsigned int SIZE>
inline __device__ bool WorkQueue<T, SIZE>::TryToPop(unsigned int slotTicket,
                                                    T*& outItem) {
  const unsigned int slotIdx = slotTicket;
  if (slotIdx >= SIZE) {
    return false;
  }

  const unsigned int currentHeadIdx =
      __atomic_load_n(m_headIdx, __ATOMIC_RELAXED);

  if (currentHeadIdx <= slotIdx) {
    return false;
  }

  QueueSlot* slot = &m_workQueue[slotIdx];

  while (__atomic_load_n(&slot->isComitted, __ATOMIC_RELAXED) == 0) {
    // Wait until item is commited by another thread.
  }

  __threadfence();  //< This barrier might not be needed - check if can be
                    // replaces with isCommited load with acquire semantics.

  outItem = slot->data;
  return true;
}