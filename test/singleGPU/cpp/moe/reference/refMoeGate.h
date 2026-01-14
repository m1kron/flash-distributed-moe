#pragma once
#include "refGemm.h"
#include "refSoftmax.h"
#include "refTopk.h"

namespace test {

struct MoeGateOutput {
  std::vector<float> topkVals;
  std::vector<int> topkIdx;
};

// tokens -> [tokensNum, hiddenSize]
// gateWeights -> [hiddenSize, expertsNum]
// out -> 2x [tokensNum, topk]
template <typename T>
inline MoeGateOutput refMoeGate(const T* tokens, const T* gateWeights,
                                int tokensNum, int expertsNum, int hiddenSize,
                                int topk) {
  auto ref =
      test::refGemm(tokens, gateWeights, tokensNum, expertsNum, hiddenSize);

  MoeGateOutput out;
  out.topkIdx.resize(topk * tokensNum);
  out.topkVals.resize(topk * tokensNum);

  test::refTopK8PerRow(ref.data(), tokensNum, expertsNum, out.topkVals.data(),
                       out.topkIdx.data(), topk);

  test::refSoftmaxRowOnlineInplace(out.topkVals.data(), tokensNum, topk);

  return out;
}

}  // namespace test