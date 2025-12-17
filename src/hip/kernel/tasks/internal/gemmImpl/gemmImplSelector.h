#pragma once
#include "src/hip/kernel/tasks/internal/gemmImpl/basicGemmTileImpl.h"
#include "src/tools/compileTimeSelector.h"

namespace moe {
namespace tasks {
namespace internal {

template <typename GEMM_TILE_METADATA>
using GemmImplSelector =
    moe::tools::CompileTimeSelector<BasicGemmTileImpl<GEMM_TILE_METADATA>>;

}
}  // namespace tasks
}  // namespace moe