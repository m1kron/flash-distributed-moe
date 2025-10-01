#include <hip/hip_runtime_api.h>

#include <iostream>
#include <vector>

#include "../../src/hip/saxpy/saxpy.h"
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

TEST(StubTest, stubTest1) {
  int devices;
  HIP_CHECK(hipGetDeviceCount(&devices));
  std::cout << "GPUS: " << devices << std::endl;

  int deviceId;
  HIP_CHECK(hipGetDevice(&deviceId));
  std::cout << "Using GPU with id: " << deviceId << std::endl;

  const int size = 10000;
  constexpr float alpha = 2.f;

  std::vector<float> host_dx(size);
  std::fill(host_dx.begin(), host_dx.end(), 1.f);

  std::vector<float> host_dy(size);
  std::fill(host_dy.begin(), host_dy.end(), 1.f);

  float* d_dx = nullptr;
  float* d_dy = nullptr;

  HIP_CHECK(hipMalloc(&d_dx, size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_dy, size * sizeof(float)));
  HIP_CHECK(hipMemcpy(d_dx, host_dx.data(), size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_dy, host_dy.data(), size * sizeof(float),
                      hipMemcpyHostToDevice));

  saxpy(alpha, d_dx, d_dy, size);

  HIP_CHECK(hipDeviceSynchronize())

  HIP_CHECK(hipGetLastError());

  HIP_CHECK(hipMemcpy(host_dy.data(), d_dy, size * sizeof(float),
                      hipMemcpyDeviceToHost));

  HIP_CHECK(hipFree(d_dx));
  HIP_CHECK(hipFree(d_dy));

  CheckConstValBuffer(host_dy.data(), host_dy.size(), 3.0f);
}