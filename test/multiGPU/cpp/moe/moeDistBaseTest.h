#pragma once
#include "gtest/gtest.h"
#include "include/iMoeKernelLauncher.h"
#include "test/common/utils.h"
// POSIX for fork/wait
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>

// Base gtest fixture for rocshmem fork based tests.
class MoeDistBaseTest : public ::testing::Test {
 protected:
  // Executes the given function in separate processes after initializing
  // rocshmem in each process.
  // Each process is assigned to different GPU device based on its rank.
  // Tfunc is assumed to be of the form:
  //   void(const moe::DistributedUniqueId&, int rank, int worldSize);
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
  int createdChildren = 0;
  bool forkError = false;
  for (int i = 0; i < worldSize; i++) {
    pid_t cpid = fork();
    if (cpid == 0) {
      master = false;
      break;
    } else if (cpid > 0) {
      // Parent path: track the child created and continue for next rank
      ++createdChildren;
      ++rank;
    } else {  // cpid == -1
      forkError = true;
      // Do not attempt more forks; proceed to wait for already created children
      break;
    }
  }

  if (master) {
    int failures = 0;
    int status = 0;
    while (createdChildren > 0) {
      pid_t pid = waitpid(-1, &status, 0);
      if (pid == -1) {
        // If we expected children but waitpid failed, record failure.
        if (errno != EINTR) {
          ++failures;
          ADD_FAILURE() << "waitpid failed: " << strerror(errno);
          break;
        }
        continue;
      }
      --createdChildren;
      if (WIFEXITED(status)) {
        int code = WEXITSTATUS(status);
        if (code != 0) {
          ++failures;
          ADD_FAILURE() << "child pid " << pid
                        << " exited with code " << code;
        }
      } else if (WIFSIGNALED(status)) {
        ++failures;
        ADD_FAILURE() << "child pid " << pid
                      << " terminated by signal " << WTERMSIG(status);
      } else {
        ++failures;
        ADD_FAILURE() << "child pid " << pid << " ended abnormally";
      }
    }

    if (forkError) {
      ++failures;
      ADD_FAILURE() << "fork failed: " << strerror(errno);
    }

    ASSERT_EQ(failures, 0) << "One or more child processes failed";
  } else {
    int GpuCount;
    HIP_ERROR_ASSERT(hipGetDeviceCount(&GpuCount));
    ASSERT_GE(GpuCount, worldSize) << "for rank: " << rank;

    HIP_ERROR_ASSERT(hipSetDevice(rank));

    func(duid, rank, worldSize);

    // Return non-zero if any gtest assertion failed in this child.
    exit(testing::Test::HasFailure() ? 1 : 0);
  }
}