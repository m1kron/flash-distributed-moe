#include "include/iMoeKernelLauncher.h"
#include "test/common/utils.h"
namespace {

TEST(MoeApiTests, maxHandledTokens) {
  const int TOKENS_NUM = 1024;
  const moe::DistributedUniqueId duid = GetDistributedUniqueId(/*empty=*/true);
  moe::IMoeKernelLauncher* launcher = nullptr;
  ASSERT_EQ(CreateLauncher(&launcher, nullptr, nullptr, nullptr, TOKENS_NUM,
                           nullptr, duid, 0, 1),
            hipErrorUnknown);
}
}  // namespace