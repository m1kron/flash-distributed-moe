#pragma once

#include <hip/hip_runtime.h>

namespace moe {

class IMoeKernelLauncher {
  virtual hipError_t Init() = 0;
  virtual hipError_t Launch() = 0;
  virtual hipError_t Deinit() = 0;
};

extern "C" hipError_t CreateLauncher(IMoeKernelLauncher** launcher);
extern "C" hipError_t DestoryLauncher(IMoeKernelLauncher* launcher);

}  // namespace moe