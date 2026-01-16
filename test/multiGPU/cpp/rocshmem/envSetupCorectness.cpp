#include "test/multiGPU/cpp/rocshmem/rocshmemCommon.h"

TEST_F(RocshmemForkBasedTest, EnvSetupCorrectness) {
  ExecuteInSeparateProcesses(3, [](int rank, int size) {
    const int rocshmem_rank = rocshmem_my_pe();
    const int rocshmem_size = rocshmem_n_pes();

    ASSERT_EQ(rocshmem_size, size);
    ASSERT_EQ(rocshmem_rank, rank);

    int GPU_id;
    HIP_ERROR_ASSERT(hipGetDevice(&GPU_id));
    ASSERT_EQ(GPU_id, rank);
  });
}