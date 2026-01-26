#include "include/iMoeKernelLauncher.h"
#include "test/singleGPU/cpp/utils/utils.h"
namespace {

TEST(MoeApiTests, maxHandledTokens) {
  const int TOKENS_NUM = 1024;
  const moe::DistributedUniqueId duid = getDistributedUniqueId(/*empty=*/true);
  moe::IMoeKernelLauncher* launcher = nullptr;
  ASSERT_EQ(CreateLauncher(&launcher, nullptr, nullptr, nullptr, TOKENS_NUM,
                           nullptr, duid, 0, 1),
            hipErrorUnknown);
}
}  // namespace