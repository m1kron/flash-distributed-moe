#pragma once

#include <cmath>
#include <cstddef>
#include <limits>

namespace test {

// GPT5-mini generated code.
// In-place online (fused) per-row softmax.
// - data: pointer to row-major matrix [rows x cols]; results are written back
// into data Uses a single forward pass to compute running max & sum, then a
// second pass that reads original values and overwrites them with normalized
// probabilities.
template <typename T>
inline void refSoftmaxRowOnlineInplace(T* data, int rows, int cols) {
  if (rows == 0 || cols == 0) return;

  for (int r = 0; r < rows; ++r) {
    T* row = data + r * cols;

    // initialize from first element
    double maxv = static_cast<double>(row[0]);
    double sum = 1.0;  // exp(row[0] - maxv) == 1.0

    // Fused forward pass: compute running max and scaled sum (log-sum-exp
    // online)
    for (int j = 1; j < cols; ++j) {
      double x = static_cast<double>(row[j]);
      if (x > maxv) {
        // rescale accumulated sum to new max
        double exp_diff = std::exp(maxv - x);  // < 1.0
        sum = sum * exp_diff + 1.0;
        maxv = x;
      } else {
        sum += std::exp(x - maxv);
      }
    }

    // Second pass: read original values and write normalized probabilities
    // in-place
    if (sum <= 0.0) {
      // degenerate: produce uniform distribution
      T inv = static_cast<T>(1.0 / static_cast<double>(cols));
      for (int j = 0; j < cols; ++j) row[j] = inv;
    } else {
      for (int j = 0; j < cols; ++j) {
        double e = std::exp(static_cast<double>(row[j]) - maxv);
        row[j] = static_cast<T>(e / sum);
      }
    }
  }
}

}  // namespace test