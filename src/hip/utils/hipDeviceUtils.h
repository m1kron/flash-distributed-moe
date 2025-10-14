#pragma once

#ifdef NDEBUG
#define HIP_DEVICE_LOG(text, ...) (void)0
#else
#define HIP_DEVICE_LOG(STR, ARGS...) \
  printf("[BIdx:%d TIdX:%d]: " STR, blockIdx.x, threadIdx.x, ##ARGS);
#endif

#define HIP_DEVICE_ASSERT(CONDITION)                        \
  {                                                         \
    if (!(CONDITION)) {                                     \
      HIP_DEVICE_LOG("Assertion failed: %s\n", #CONDITION); \
    }                                                       \
  }