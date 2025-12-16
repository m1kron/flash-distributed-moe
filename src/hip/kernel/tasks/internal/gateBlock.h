#pragma once
#include "src/hip/kernel/tasks/internal/softmaxBlock.h"
#include "src/hip/kernel/tasks/internal/topkBlock.h"

namespace moe {
namespace tasks {
namespace internal {

// Internal imple of moe gate. Returns result via internally allocated
// outTopkVals_shared and outTopkIdx_shared.
template <typename T_GATE_GEMM_IMPL, int TOPK>
__device__ void gate_block(
    const typename T_GATE_GEMM_IMPL::TILE_METADATA::TType* __restrict__ tokens,
    const typename T_GATE_GEMM_IMPL::TILE_METADATA::
        TType* __restrict__ gateWeights,
    int tokenIdx,
    typename T_GATE_GEMM_IMPL::TILE_METADATA::TType** outTopkVals_shared,
    int** outTopkIdx_shared, void* sharedMemPool) {
  using T_GATE_GEMM_TILE = typename T_GATE_GEMM_IMPL::TILE_METADATA;
  using TType = typename T_GATE_GEMM_TILE::TType;

  static_assert(
      T_GATE_GEMM_TILE::THREAD_OUTPUT_SIZE == 1,
      "Gemm tile for gate must have output size 1 per thread(softmax + topk).");

  TType out_regs[T_GATE_GEMM_TILE::THREAD_OUTPUT_SIZE];

  T_GATE_GEMM_IMPL::Execute(tokens, gateWeights, out_regs, tokenIdx, 0,
                            sharedMemPool);
  out_regs[0] =
      moe::tasks::internal::Softmax_block<T_GATE_GEMM_TILE::THREADS, TType>(
          out_regs[0], sharedMemPool);

  char* sharedMemPoolBytes = reinterpret_cast<char*>(sharedMemPool);
  TType* _outTopkVals_shared = reinterpret_cast<TType*>(sharedMemPoolBytes);
  int* _outTopkIdx_shared =
      reinterpret_cast<int*>(sharedMemPoolBytes + sizeof(TType) * TOPK);
  int* nextPool = _outTopkIdx_shared + TOPK;

  moe::tasks::internal::Topk8_block<T_GATE_GEMM_TILE::THREADS, TType>(
      out_regs[0], _outTopkVals_shared, _outTopkIdx_shared, nextPool);

  *outTopkVals_shared = _outTopkVals_shared;
  *outTopkIdx_shared = _outTopkIdx_shared;
}

}  // namespace internal
}  // namespace tasks
}  // namespace moe