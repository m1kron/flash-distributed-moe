#pragma once
#include "src/hip/common/metadata.h"
#include "src/hip/kernel/tasks/internal/gemmImpl/gemmImplSelector.h"

namespace moe {
template <typename T>
constexpr T max(T a, T b) {
  return a > b ? a : b;
}

template <typename TILES_CONFIG>
struct GemmRuntimeConfig {
  using GATE_GEMM_TILE_IMPL = typename moe::tasks::internal::GemmImplSelector<
      typename TILES_CONFIG::GATE_TILE_METADATA>::type;

  using FFN1_GEMM_TILE_IMPL = typename moe::tasks::internal::GemmImplSelector<
      typename TILES_CONFIG::FFN1_TILE_METADATA>::type;

  using FFN2_GEMM_TILE_IMPL = typename moe::tasks::internal::GemmImplSelector<
      typename TILES_CONFIG::FFN2_TILE_METADATA>::type;
};

template <typename _MOE_METADATA>
struct MoeRuntimeConfigImpl {
  using MOE_METADATA = _MOE_METADATA;
  using GEMM_RUNTIME_CONFIG =
      GemmRuntimeConfig<typename MOE_METADATA::TILES_CONFIG>;

  static constexpr int SHARED_MEM_SIZE_BYTES = max(
      GEMM_RUNTIME_CONFIG::GATE_GEMM_TILE_IMPL::NeededSharedMemBytes(),
      max(GEMM_RUNTIME_CONFIG::FFN1_GEMM_TILE_IMPL::NeededSharedMemBytes(),
          GEMM_RUNTIME_CONFIG::FFN2_GEMM_TILE_IMPL::NeededSharedMemBytes()));
};

using MoeRuntimeConfig = MoeRuntimeConfigImpl<moe::MoeImplMetadata>;
}  // namespace moe