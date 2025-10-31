#pragma once
#include "refGemm.h"
#include "refSoftmax.h"
#include "refTopk.h"

namespace test {

struct MoeGateOutput {
  std::vector<float> topkVals;
  std::vector<int> topkIdx;
};

template <typename T>
inline MoeGateOutput refMoeGate(const T* A, const T* B, int M, int N, int K,
                                int topk) {
  auto ref = test::refGemm(A, B, M, N, K);

  test::refSoftmaxRowOnlineInplace(ref.data(), M, N);

  MoeGateOutput out;
  out.topkIdx.resize(topk * M);
  out.topkVals.resize(topk * M);

  test::refTopK8PerRow(ref.data(), M, N, out.topkVals.data(),
                       out.topkIdx.data(), topk);

  return out;
}

}  // namespace test