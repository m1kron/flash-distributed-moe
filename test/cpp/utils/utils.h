#pragma once
#include <random>
#include <vector>

#include "gtest/gtest.h"

static constexpr uint32_t SEED = 7907;

#define HIP_ERROR_ASSERT(condition)                     \
  {                                                     \
    const hipError_t error = condition;                 \
    ASSERT_EQ(error, hipSuccess) << " for " #condition; \
  }

// Checks if all values in buffer are equal to value.
template <typename T>
void CheckConstValBuffer(const std::vector<T>& buffer, T value);

// Compares buffer to ref.
template <typename T>
void CheckAgainstRefBuffer(const std::vector<T>& buffer,
                           const std::vector<T>& ref, float abs_error = 1e-5f);

// Generates single random val from range [low, high].
template <typename T>
T GetRandom(T low, T high);

// Generate a vector of random values of type T.
template <typename T>
std::vector<T> RandomVector(size_t size, T low = T(0), T high = T(1));

////////////////////////////////////////////////////////////////////////
//
// IMPLEMENTATION
//
////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////
template <typename T>
void CheckConstValBuffer(const std::vector<T>& buffer, T value) {
  for (size_t i = 0; i < buffer.size(); ++i) {
    ASSERT_EQ(buffer[i], value) << "for i = " << i;
  }
}

////////////////////////////////////////////////////////////////////////
template <typename T>
void CheckAgainstRefBuffer(const std::vector<T>& buffer,
                           const std::vector<T>& ref, float abs_error) {
  ASSERT_EQ(buffer.size(), ref.size());
  for (size_t i = 0; i < ref.size(); ++i) {
    ASSERT_NEAR(buffer[i], ref[i], abs_error) << "for i = " << i;
  }
}

////////////////////////////////////////////////////////////////////////
template <typename T>
T GetRandom(T low, T high) {
  struct RandWrapper {
    RandWrapper() { srand(SEED); }
    float getRandomFloat01() {
      return static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }
  };
  static RandWrapper randWrapper;
  return low + static_cast<T>(randWrapper.getRandomFloat01() *
                              static_cast<float>(high - low));
  ;
}

////////////////////////////////////////////////////////////////////////
template <typename T>
std::vector<T> RandomVector(size_t size, T low, T high) {
  static_assert(std::is_arithmetic_v<T>,
                "RandomVector requires an arithmetic type");

  std::vector<T> out;
  out.reserve(size);
  for (size_t i = 0; i < size; ++i) out.push_back(GetRandom<T>(low, high));

  return out;
}