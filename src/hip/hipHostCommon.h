#pragma once

#include <hip/hip_runtime.h>

#include <iostream>

#define HIP_ERROR_CHECK(condition)                                        \
  {                                                                       \
    const hipError_t error = condition;                                   \
    if (error != hipSuccess) {                                            \
      std::cerr << "An error encountered: \"" << hipGetErrorString(error) \
                << "\" at " << __FILE__ << ':' << __LINE__ << std::endl;  \
      return error;                                                       \
    }                                                                     \
  }
