#include "test/multiGPU/cpp/moe/moeDistBaseTest.h"

TEST_F(MoeDistBaseTest, Bla) {

  std::vector<int> data{ 1, 2, 3, 4 };

  ExecuteInSeparateProcesses(
      4, [&data](const moe::DistributedUniqueId& duid, int rank, int size) {
        moe::IMoeKernelLauncher* launcher = nullptr;
        HIP_ERROR_ASSERT(CreateLauncher(&launcher, nullptr, nullptr, nullptr, 1,
                                        nullptr, duid, rank, size));

        int GPU_id;
        HIP_ERROR_ASSERT(hipGetDevice(&GPU_id));
        ASSERT_EQ(GPU_id, rank);

        std::cout << "Rank " << rank << " successfully created launcher."
                  << std::endl;

        for (const auto& data_item : data) {
          std::cout << "Rank " << rank << " data item: " << data_item
                    << std::endl;
        }

        HIP_ERROR_ASSERT(DestroyLauncher(launcher, nullptr));
      });
}