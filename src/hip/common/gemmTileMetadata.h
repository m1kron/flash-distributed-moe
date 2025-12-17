#pragma once

template <int _N, int _K, int _TILE_M, int _TILE_N, int _TILE_K, int _THREADS,
          typename _TType>
struct GemmTileMetadata {
  using TType = _TType;
  constexpr static int N = _N;
  constexpr static int K = _K;
  constexpr static int TILE_M = _TILE_M;
  constexpr static int TILE_N = _TILE_N;
  constexpr static int TILE_K = _TILE_K;
  constexpr static int THREADS = _THREADS;
  constexpr static int THREAD_OUTPUT_SIZE = (TILE_M * TILE_N) / THREADS;

  static_assert(THREAD_OUTPUT_SIZE > 0, "Invalid thread output size");
};