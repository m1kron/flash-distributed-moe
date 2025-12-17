#pragma once
#include "gemmTileMetadata.h"

namespace moe {

// Describes Moe problem.
struct MoeProblemConfig {
  static constexpr int EXPERTS_NUM = 128;
  static constexpr int HIDDEN_SIZE = 2048;
  static constexpr int EXPERT_INTERMEDIATE_SIZE = 768;
  static constexpr int TOPK = 8;
  using TDataType = float;
};

// Describes HW config
struct HWConfig {
  static constexpr int THREADS = 128;
  static constexpr int BLOCKS = 304;
};

template <typename HW_CONFIG, typename MOE_PROBLEM_CONFIG>
struct TilesConfig {
  static constexpr int REDUCTION_TILE_SIZE = 512;
  static constexpr int REDUCTION_CHUNKS_PER_TOKEN =
      (MOE_PROBLEM_CONFIG::HIDDEN_SIZE / REDUCTION_TILE_SIZE);

  static constexpr int EXPERTS_N_TILE_SIZE = 256;
  static constexpr int TILE_K_SIZE = 32;
  static constexpr int TILE_M_SIZE = 1;

  using GATE_TILE_METADATA = GemmTileMetadata<
      MOE_PROBLEM_CONFIG::EXPERTS_NUM, MOE_PROBLEM_CONFIG::HIDDEN_SIZE,
      TILE_M_SIZE, MOE_PROBLEM_CONFIG::EXPERTS_NUM, TILE_K_SIZE,
      HW_CONFIG::THREADS, typename MOE_PROBLEM_CONFIG::TDataType>;

  using FFN1_TILE_METADATA =
      GemmTileMetadata<MOE_PROBLEM_CONFIG::EXPERT_INTERMEDIATE_SIZE,
                       MOE_PROBLEM_CONFIG::HIDDEN_SIZE, TILE_M_SIZE,
                       EXPERTS_N_TILE_SIZE, TILE_K_SIZE, HW_CONFIG::THREADS,
                       typename MOE_PROBLEM_CONFIG::TDataType>;

  using FFN2_TILE_METADATA =
      GemmTileMetadata<MOE_PROBLEM_CONFIG::HIDDEN_SIZE,
                       MOE_PROBLEM_CONFIG::EXPERT_INTERMEDIATE_SIZE,
                       TILE_M_SIZE, EXPERTS_N_TILE_SIZE, TILE_K_SIZE,
                       HW_CONFIG::THREADS,
                       typename MOE_PROBLEM_CONFIG::TDataType>;
};

// Moe impl metadata.
struct MoeImplMetadata {
  using MOE_PROBLEM_CONFIG = MoeProblemConfig;
  using HW_CONFIG = HWConfig;
  using TILES_CONFIG = TilesConfig<HW_CONFIG, MOE_PROBLEM_CONFIG>;
};

}  // namespace moe