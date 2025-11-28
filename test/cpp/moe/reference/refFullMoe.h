#pragma once
#include "refExpertFFN1.h"
#include "refMoeGate.h"

namespace test {
// Reference implementation of full moe kernel.
template <typename T>
static std::vector<T> refFullMoe(const T* tokens, const T* gateWeights,
                                 const T* ffn1ExpertWeights,
                                 const T* ffn2ExpertWeights, int tokensNum,
                                 int expertsNum, int hiddenSize, int topk,
                                 int expertIntermediateSize) {
  assert(tokens != nullptr);
  assert(gateWeights != nullptr);
  // ffn1ExpertWeights may be nullptr -> then projection fields will be zeros.

  // Run gating to obtain topk info
  auto gateOut = test::refMoeGate(tokens, gateWeights, tokensNum, expertsNum,
                                  hiddenSize, topk);
  // gateOut.topkVals and gateOut.topkIdx are expected to be vectors of size
  // tokensNum*topk
  const std::vector<T>& topkVals = gateOut.topkVals;
  const std::vector<int>& topkIdx = gateOut.topkIdx;
  const int rows = tokensNum;
  const int cols = hiddenSize;

  std::vector<T> out;
  out.assign(size_t(rows) * size_t(cols), 0.0f);

  for (int tokenIdx = 0; tokenIdx < tokensNum; ++tokenIdx) {
    const T* tokenPtr = tokens + size_t(tokenIdx) * size_t(hiddenSize);
    T* outTokenPtr = out.data() + size_t(tokenIdx) * size_t(hiddenSize);
    for (int k = 0; k < topk; ++k) {
      const int expert = topkIdx[tokenIdx * topk + k];
      const T gateScore = topkVals[tokenIdx * topk + k];

      // projection columns start at offset 2
      if (expert < 0 || expert >= expertsNum || ffn1ExpertWeights == nullptr) {
        // leave projection zeros
        continue;
      }

      // ffn1ExpertWeights layout assumed:
      // [expert][2*expertIntermediate][hidden]
      const T* expertFFN1W =
          ffn1ExpertWeights + size_t(expert) * size_t(hiddenSize) *
                                  size_t(2 * expertIntermediateSize);

      auto refFFN1Out = test::refExpertFFN1(tokenPtr, expertFFN1W, 1,
                                            hiddenSize, expertIntermediateSize);

      // ffn1ExpertWeights layout assumed: [expert][hidden][expertIntermediate]
      const T* expertFFN2W =
          ffn2ExpertWeights +
          size_t(expert) * size_t(hiddenSize) * size_t(expertIntermediateSize);
      auto refFFN2Out = test::refGemm(refFFN1Out.data(), expertFFN2W, 1,
                                      hiddenSize, expertIntermediateSize);

      // Copy projection into output row
      for (int d = 0; d < hiddenSize; ++d) {
        outTokenPtr[d] += refFFN2Out[size_t(d)] *
                          gateScore;  // proj is 1 x N stored row-major
      }
    }
  }

  return out;
}
}  // namespace test