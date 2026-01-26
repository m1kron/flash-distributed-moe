#pragma once
#include <rocshmem/rocshmem.hpp>

#include "gtest/gtest.h"
#include "test/common/utils.h"

using namespace rocshmem;

#define ROCSHMEM_ERROR_ASSERT(condition...)    \
  {                                            \
    const int error = (condition);             \
    ASSERT_EQ(error, 0) << " for " #condition; \
  }

// Base gtest fixture for rocshmem fork based tests.
class RocshmemForkBasedTest : public ::testing::Test {
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
inline void RocshmemForkBasedTest::ExecuteInSeparateProcesses(
    int worldSize, const Tfunc& func) {
  rocshmem_uniqueid_t uid;

  ROCSHMEM_ERROR_ASSERT(rocshmem_get_uniqueid(&uid));

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
    // Init hip runtime for this process and init rocshmem
    int GpuCount;
    HIP_ERROR_ASSERT(hipGetDeviceCount(&GpuCount));
    ASSERT_GE(GpuCount, worldSize) << "for rank: " << rank;

    HIP_ERROR_ASSERT(hipSetDevice(rank));

    rocshmem_init_attr_t attr;

    ROCSHMEM_ERROR_ASSERT(
        rocshmem_set_attr_uniqueid_args(rank, worldSize, &uid, &attr));
    ROCSHMEM_ERROR_ASSERT(
        rocshmem_init_attr(ROCSHMEM_INIT_WITH_UNIQUEID, &attr));

    func(rank, worldSize);

    rocshmem_finalize();
    exit(testing::Test::HasFailure());
  }
}