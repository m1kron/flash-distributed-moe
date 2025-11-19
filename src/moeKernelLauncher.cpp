#include "include/iMoeKernelLauncher.h"

namespace moe {

////////////////////////////////////////////////////////////////////
extern "C" hipError_t CreateLauncher(IMoeKernelLauncher** launcher) {
  return hipSuccess;
}

////////////////////////////////////////////////////////////////////
extern "C" hipError_t DestoryLauncher(IMoeKernelLauncher* launcher) {
  return hipSuccess;
}

}  // namespace moe