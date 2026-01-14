#pragma once

#include <cmath>
#include <cstddef>
#include <type_traits>

#include "refGemm.h"

// GPT5 mini generated code.
namespace test {
// SiLU(x) = x * sigmoid(x), with numerically stable sigmoid
template <typename T>
inline T sigmoid_stable(T x) {
  if (x >= T(0)) {
    T e = std::exp(-x);
    return T(1) / (T(1) + e);
  } else {
    T e = std::exp(x);
    return e / (T(1) + e);
  }
}

// Case: input "in" is a matrix [rows x (2*D)], left half = A (cols D),
// right half = B (cols D). Compute out = SiLU(A) * B (elementwise), where
// out has shape [rows x D].
// Parameters:
//  - in: pointer to input matrix (row-major) with totalCols = 2*D
//  - out: pointer to output buffer, size rows * D
//  - rows: number of rows
//  - totalCols: total columns in "in" (must be even)
template <typename T>
inline void refSiluMul(const T* in, T* out, std::size_t rows,
                       std::size_t totalCols) {
  static_assert(std::is_floating_point<T>::value, "floating point required");
  if (totalCols % 2 != 0) return;
  const std::size_t D = totalCols / 2;
  for (std::size_t r = 0; r < rows; ++r) {
    const T* Arow = in + r * totalCols;
    const T* Brow = Arow + D;
    T* Orow = out + r * D;
    for (std::size_t c = 0; c < D; ++c) {
      T a = Arow[c];
      T b = Brow[c];
      T s = sigmoid_stable(a);
      Orow[c] = a * s * b;  // SiLU(a) * b
    }
  }
}

// Ref implementation of moe expert ffn1.
template <typename T>
std::vector<T> refExpertFFN1(const T* tokens, const T* expertWeights,
                             int tokensNum, int hiddenSize,
                             int expertIntermediateSize) {
  auto gemmOut = test::refGemm(tokens, expertWeights, tokensNum,
                               2 * expertIntermediateSize, hiddenSize);

  std::vector<T> out;
  out.resize(tokensNum * expertIntermediateSize);

  test::refSiluMul(gemmOut.data(), out.data(), tokensNum,
                   2 * expertIntermediateSize);

  return out;
}

}  // namespace test