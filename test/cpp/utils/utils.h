#pragma once
#include "gtest/gtest.h"

#include <random>

#define HIP_ERROR_ASSERT(condition)                     \
  {                                                     \
    const hipError_t error = condition;                 \
    ASSERT_EQ(error, hipSuccess) << " for " #condition; \
  }

template <typename T>
static void CheckConstValBuffer(const T* buffer, int size, T value) {
  for (int i = 0; i < size; ++i) {
    ASSERT_EQ(buffer[i], value);
  }
}

template <typename T>
static void CheckAgainstRefBuffer(const T* buffer, const T* ref, int size) {
  for (int i = 0; i < size; ++i) {
    ASSERT_FLOAT_EQ(buffer[i], ref[i]);
  }
}

// Generate a vector of random values of type T.
// T must be an arithmetic type (integral or floating point).
// default seed uses random_device.
template <typename T>
static std::vector<T> RandomVector(size_t size, T low = T(0), T high = T(1),
                                   uint32_t seed = 7907) {
  static_assert(std::is_arithmetic_v<T>,
                "RandomVector requires an arithmetic type");

  std::vector<T> out;
  out.reserve(size);

  std::mt19937 rng(seed);

  if constexpr (std::is_integral_v<T>) {
    std::uniform_int_distribution<T> dist(low, high);
    for (size_t i = 0; i < size; ++i) out.push_back(dist(rng));
  } else {  // floating point
    std::uniform_real_distribution<T> dist(low, high);
    for (size_t i = 0; i < size; ++i) out.push_back(dist(rng));
  }

  return out;
}