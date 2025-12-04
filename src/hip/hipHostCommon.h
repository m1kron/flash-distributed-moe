#pragma once

#include <assert.h>
#include <hip/hip_runtime.h>
#include <string.h>

#define __FILENAME__ \
  (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)

#define HIP_ERROR_CHECK(condition)                                   \
  {                                                                  \
    const hipError_t error = condition;                              \
    if (error != hipSuccess) {                                       \
      printf("[MOE HIP ERROR: %s:%i]: %s\n", __FILENAME__, __LINE__, \
             hipGetErrorString(error));                              \
      return error;                                                  \
    }                                                                \
  }

#ifdef NDEBUG
#define MOE_LOG(text, ...) (void)0
#else
#define MOE_LOG(STR, ARGS...) \
  printf("[MOE LOG: %s:%i]: " STR "\n", __FILENAME__, __LINE__, ##ARGS);
#endif

#define MOE_ASSERT(CONDITION) assert(CONDITION);