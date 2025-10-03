#include <hip/hip_runtime_api.h>

#include <iostream>
#include <vector>

#include "../../src/hip/scheduler/schedulerPrototype.h"
#include "gtest/gtest.h"

constexpr int error_exit_code = -1;

template <typename T>
static void PrintBuffer(const T* buffer, int sizeToPrint) {
  std::cout << "First " << sizeToPrint
            << " elements of the buffer: " << std::endl;
  for (int i = 0; i < sizeToPrint; ++i) {
    std::cout << buffer[i] << ", ";
  }

  std::cout << std::endl;
}

template <typename T>
static void CheckConstValBuffer(const T* buffer, int size, T value) {
  for (int i = 0; i < size; ++i) {
    ASSERT_EQ(buffer[i], value) << "for i: " << i;
  }
}

#define HIP_CHECK(condition)                                              \
  {                                                                       \
    const hipError_t error = condition;                                   \
    if (error != hipSuccess) {                                            \
      std::cerr << "An error encountered: \"" << hipGetErrorString(error) \
                << "\" at " << __FILE__ << ':' << __LINE__ << std::endl;  \
      std::exit(error_exit_code);                                         \
    }                                                                     \
  }

TEST(SchedulerTest, SchedulerTest) {
  int devices;
  HIP_CHECK(hipGetDeviceCount(&devices));
  std::cout << "GPUS: " << devices << std::endl;

  int deviceId;
  HIP_CHECK(hipGetDevice(&deviceId));
  std::cout << "Using GPU with id: " << deviceId << std::endl;

  runScheduler();

  HIP_CHECK(hipDeviceSynchronize())

  HIP_CHECK(hipGetLastError());
}