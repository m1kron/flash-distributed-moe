#pragma once

#include <vector>

namespace test {

// A: M x K (row-major)
// B: N x K (row-major)  [note: transposed versus the usual K x N]
// C: M x N (row-major)
// Computes C = A * B^T
template <typename T>
inline std::vector<T> refGemm(const T* A, const T* B, int M, int N, int K) {
  std::vector<T> out(size_t(M) * size_t(N), T(0));
  T* C = out.data();

  for (int i = 0; i < M; ++i) {
    const T* Arow = A + size_t(i) * size_t(K);
    T* Crow = C + size_t(i) * size_t(N);
    for (int j = 0; j < N; ++j) {
      const T* Brow = B + size_t(j) * size_t(K);  // B row j (length K)
      T acc = T(0);
      for (int k = 0; k < K; ++k) {
        acc += Arow[k] * Brow[k];
      }
      Crow[j] = acc;
    }
  }

  return out;
}

}  // namespace test