#pragma once

#include <cstddef>
#include <cstring>
#include <vector>

namespace test {

template <typename T>
inline std::vector<T> refGemm(const T* A, const T* B, int M, int N, int K) {
  // Row-major:
  //  A: M x K
  //  B: K x N
  //  C: M x N

  std::vector<T> out(M * N, T(0));
  T* C = out.data();

  for (int i = 0; i < M; ++i) {
    T* Crow = C + size_t(i) * size_t(N);
    for (int k = 0; k < K; ++k) {
      const T a = A[size_t(i) * size_t(K) + size_t(k)];
      const T* Brow = B + size_t(k) * size_t(N);
      for (int j = 0; j < N; ++j) {
        Crow[j] += a * Brow[j];
      }
    }
  }

  return out;
}

}  // namespace test