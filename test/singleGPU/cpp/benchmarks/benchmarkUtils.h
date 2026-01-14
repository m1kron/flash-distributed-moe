#pragma once
#include <hip/hip_runtime.h>
#include <iostream>

namespace test {
namespace bench {
#define HIP_CHECK(call)                                              \
  do {                                                               \
    hipError_t err = call;                                           \
    if (err != hipSuccess) {                                         \
      std::cerr << "HIP error: " << hipGetErrorString(err) << " at " \
                << __FILE__ << ":" << __LINE__ << std::endl;         \
      std::exit(EXIT_FAILURE);                                       \
    }                                                                \
  } while (0)

/*
  Benchmark test that compares staticGemm and taskSystemGemm.
  - Allocates device buffers for A, B, C
  - Initializes A and B with 1.0f
  - Runs warmup then timed repetitions using hipEvent timing
  - Validates results (each C element should be ~K when A/B are all ones)
  - Prints average GPU kernel time (ms) for both implementations
*/

template <typename TFunc, typename... TArgs>
float Benchmark(int repetitions, int warmup, const TFunc& func,
                TArgs&&... args) {
  hipEvent_t start, stop;
  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&stop));

  // Warmup
  HIP_CHECK(hipDeviceSynchronize());
  for (int i = 0; i < repetitions; ++i) {
    HIP_CHECK(func(std::forward<TArgs>(args)...));
  }
  HIP_CHECK(hipDeviceSynchronize());

  HIP_CHECK(hipEventRecord(start, 0));
  for (int i = 0; i < repetitions; ++i) {
    HIP_CHECK(func(std::forward<TArgs>(args)...));
  }
  HIP_CHECK(hipEventRecord(stop, 0));
  HIP_CHECK(hipEventSynchronize(stop));

  float ms = 0.0f;
  HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
  HIP_CHECK(hipEventDestroy(start));
  HIP_CHECK(hipEventDestroy(stop));
  return ms / float(repetitions);
}
}  // namespace bench
}  // namespace test
