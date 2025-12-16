#pragma once
#include "src/hip/kernel/tasks/internal/gemmImpl/basicGemm.h"
#include "src/tools/compileTimeSelector.h"

namespace moe {
namespace tasks {
namespace internal {

template <typename GEMM_TILE_PARAMS>
using GemmImplSelector =
    moe::tools::CompileTimeSelector<BasicGemmTileImpl<GEMM_TILE_PARAMS>>;

}
}  // namespace tasks
}  // namespace moe