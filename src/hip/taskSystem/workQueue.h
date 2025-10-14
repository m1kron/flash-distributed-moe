#pragma once
#include "../hipCommon.h"
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

  QueueSlot* m_workQueue;
  unsigned int* m_headIdx;
  unsigned int* m_tailIdx;

  __host__ void Init() {
    HIP_CHECK(hipMalloc(&m_workQueue, SIZE * sizeof(QueueSlot)));
    HIP_CHECK(hipMalloc(&m_headIdx, sizeof(int*)));
    HIP_CHECK(hipMalloc(&m_tailIdx, sizeof(int*)));

    HIP_CHECK(hipMemset(m_workQueue, 0, SIZE * sizeof(QueueSlot)));

    const int zero = 0;

    HIP_CHECK(hipMemcpy(m_headIdx, &zero, sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(m_tailIdx, &zero, sizeof(int), hipMemcpyHostToDevice));
  }

  __host__ void Deinit() {
    HIP_CHECK(hipFree(m_tailIdx));
    HIP_CHECK(hipFree(m_headIdx));
    HIP_CHECK(hipFree(m_workQueue));
  }

  __device__ bool Push(T* item_global_ptr) {
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

  __device__ unsigned int ReserveSlotTicket() {
    const unsigned int slotIdx = atomicAdd(m_tailIdx, 1);
    if (slotIdx >= SIZE) {
      HIP_DEVICE_LOG("Out of slots!\n");
      return 0;
    }
    return slotIdx;
  }

  __device__ bool TryToPop(unsigned int slotTicket, T*& outItem) {
    const unsigned int slotIdx = slotTicket;
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
                      //replaces with isCommited load with acquire semantics.

    outItem = slot->data;
    return true;
  }
};