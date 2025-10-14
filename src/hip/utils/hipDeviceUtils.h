#pragma once
#include <hip/hip_runtime.h>
#include <assert.h>

#ifdef NDEBUG
#define HIP_DEVICE_LOG(text, ...) (void)0
#else
#define HIP_DEVICE_LOG(STR, ARGS...) \
  printf("[BIdx:%d TIdX:%d]: " STR, blockIdx.x, threadIdx.x, ##ARGS);
#endif

#define HIP_DEVICE_ASSERT(CONDITION) assert(CONDITION);