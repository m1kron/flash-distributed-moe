#pragma once
#include "gtest/gtest.h"
#include "include/iMoeKernelLauncher.h"
#include "test/singleGPU/cpp/utils/utils.h"

// Base gtest fixture for rocshmem fork based tests.
class MoeDistBaseTest : public ::testing::Test {
 protected:
  // Executes the given function in separate processes after initializing
  // rocshmem in each process.
  // Each process is assigned to different GPU device based on its rank.
  // Tfunc is assumed to be of the form: void(int rank, int size);
  template <typename Tfunc>
  void ExecuteInSeparateProcesses(int worldSize, const Tfunc& func);
};

////////////////////////////////////////////////////////////
//
// IMPLEMENTATION:
//
////////////////////////////////////////////////////////////

template <typename Tfunc>
inline void MoeDistBaseTest::ExecuteInSeparateProcesses(
    int worldSize, const Tfunc& func) {
  const moe::DistributedUniqueId duid = GetDistributedUniqueId(/*empty=*/false);

  bool master = true;
  int rank = 0;
  for (int i = 0; i < worldSize; i++) {
    if (fork() == 0) {
      master = false;
      break;
    }
    ++rank;
  }

  if (master) {
    while (wait(NULL) > 0);
  } else {
    int GpuCount;
    HIP_ERROR_ASSERT(hipGetDeviceCount(&GpuCount));
    ASSERT_GE(GpuCount, worldSize) << "for rank: " << rank;

    HIP_ERROR_ASSERT(hipSetDevice(rank));

    func(duid, rank, worldSize);

    exit(testing::Test::HasFailure());
  }
}