#pragma once

// gemm tile metadata.
template <int _N, int _K, int _TILE_M, int _TILE_N, int _TILE_K, int _THREADS>
struct GemmTileParams {
  using TOutputType = float;
  using TInputType = float;
  constexpr static int N = _N;
  constexpr static int K = _K;
  constexpr static int TILE_M = _TILE_M;
  constexpr static int TILE_N = _TILE_N;
  constexpr static int TILE_K = _TILE_K;
  constexpr static int THREADS = _THREADS;
  constexpr static int THREAD_OUTPUT_SIZE = (TILE_M * TILE_N) / THREADS;
  constexpr static int SHARED_MEM_NEEDES_BYTES =
      ((TILE_M * TILE_K) + (TILE_K * TILE_N)) * sizeof(TInputType);
};