#include "moeKernelLauncher.h"
#include "src/hip/hipHostCommon.h"
#include "src/hip/kernel/moeKernel.h"

namespace moe {

extern __constant__ TaskManagerType globalTaskManager;
extern __constant__ int* globalFFN1SyncArray;
extern __constant__ int* globalChunkReduceSyncArray;
struct InternalState {
  TaskManagerType taskManagerHost;
  int* ffn1SyncArrayDevice;
  int* chunkReduceSyncArrayDevice;
  void* ffn1Output_device;
  void* ffn2Output_device;
};

template <typename MOE_METADATA>
hipError_t moeKernelLaunchFunc(InternalState* state, const void* tokens,
                               const void* gateWeights,
                               const void* ffn1ExpertWeights,
                               const void* ffn2ExpertWeights, void* output,
                               int tokensNum, hipStream_t stream) {
  using TMoeDataType = typename MOE_METADATA::MOE_PROBLEM_CONFIG::TDataType;
  dim3 grid(MOE_METADATA::HW_CONFIG::BLOCKS);
  dim3 block(MOE_METADATA::HW_CONFIG::THREADS);

  const int ffn1TasksNum =
      tokensNum * MOE_METADATA::MOE_PROBLEM_CONFIG::TOPK *
      (MOE_METADATA::MOE_PROBLEM_CONFIG::EXPERT_INTERMEDIATE_SIZE /
       MOE_METADATA::TILES_CONFIG::FFN1_TILE_METADATA::TILE_N);
  const int ffn2TasksNum =
      tokensNum * MOE_METADATA::MOE_PROBLEM_CONFIG::TOPK *
      (MOE_METADATA::MOE_PROBLEM_CONFIG::HIDDEN_SIZE /
       MOE_METADATA::TILES_CONFIG::FFN2_TILE_METADATA::TILE_N);
  const int reductionTasksNum =
      tokensNum * MOE_METADATA::TILES_CONFIG::REDUCTION_CHUNKS_PER_TOKEN;

  const int expectedNumTasks = ffn1TasksNum + ffn2TasksNum + reductionTasksNum;
  HIP_ERROR_CHECK(
      state->taskManagerHost.PrepareForNextLaunch(stream, expectedNumTasks));

  HIP_ERROR_CHECK(hipMemsetAsync(
      state->ffn1SyncArrayDevice, 0,
      sizeof(int) * tokensNum * MOE_METADATA::MOE_PROBLEM_CONFIG::TOPK,
      stream));

  HIP_ERROR_CHECK(
      hipMemsetAsync(state->chunkReduceSyncArrayDevice, 0,
                     sizeof(int) * tokensNum *
                         MOE_METADATA::TILES_CONFIG::REDUCTION_CHUNKS_PER_TOKEN,
                     stream));

  moeKernel<MOE_METADATA><<<grid, block, 0, stream>>>(
      static_cast<const TMoeDataType*>(tokens),
      static_cast<const TMoeDataType*>(gateWeights),
      static_cast<const TMoeDataType*>(ffn1ExpertWeights),
      static_cast<const TMoeDataType*>(ffn2ExpertWeights),
      static_cast<TMoeDataType*>(state->ffn1Output_device),
      static_cast<TMoeDataType*>(state->ffn2Output_device),
      static_cast<TMoeDataType*>(output), tokensNum);

  return hipSuccess;
}

////////////////////////////////////////////////////////////////
hipError_t MoeKernelLauncher::Init(hipStream_t stream, int maxTokens) {
  m_internalState = new InternalState();

  const int ffn1OutputSize =
      maxTokens * moe::MoeImplMetadata::MOE_PROBLEM_CONFIG::TOPK *
      moe::MoeImplMetadata::MOE_PROBLEM_CONFIG::EXPERT_INTERMEDIATE_SIZE;
  const int ffn2OutputSize =
      maxTokens * moe::MoeImplMetadata::MOE_PROBLEM_CONFIG::TOPK *
      moe::MoeImplMetadata::MOE_PROBLEM_CONFIG::HIDDEN_SIZE;

  HIP_ERROR_CHECK(m_internalState->taskManagerHost.Init(stream));
  HIP_ERROR_CHECK(hipMemcpyToSymbolAsync(
      HIP_SYMBOL(globalTaskManager), &m_internalState->taskManagerHost,
      sizeof(TaskManagerType), 0, hipMemcpyHostToDevice, stream));

  HIP_ERROR_CHECK(hipMallocAsync(
      &m_internalState->ffn1SyncArrayDevice,
      sizeof(int) * maxTokens * moe::MoeImplMetadata::MOE_PROBLEM_CONFIG::TOPK,
      stream));
  HIP_ERROR_CHECK(hipMemcpyToSymbolAsync(
      HIP_SYMBOL(globalFFN1SyncArray), &m_internalState->ffn1SyncArrayDevice,
      sizeof(int*), 0, hipMemcpyHostToDevice, stream));

  HIP_ERROR_CHECK(hipMallocAsync(
      &m_internalState->chunkReduceSyncArrayDevice,
      sizeof(int) * maxTokens *
          moe::MoeImplMetadata::TILES_CONFIG::REDUCTION_CHUNKS_PER_TOKEN,
      stream));
  HIP_ERROR_CHECK(
      hipMemcpyToSymbolAsync(HIP_SYMBOL(globalChunkReduceSyncArray),
                             &m_internalState->chunkReduceSyncArrayDevice,
                             sizeof(int*), 0, hipMemcpyHostToDevice, stream));

  HIP_ERROR_CHECK(hipMalloc(
      &m_internalState->ffn1Output_device,
      ffn1OutputSize *
          sizeof(moe::MoeImplMetadata::MOE_PROBLEM_CONFIG::TDataType)));
  HIP_ERROR_CHECK(hipMalloc(
      &m_internalState->ffn2Output_device,
      ffn2OutputSize *
          sizeof(moe::MoeImplMetadata::MOE_PROBLEM_CONFIG::TDataType)));

  HIP_ERROR_CHECK(hipStreamSynchronize(stream));

  return hipSuccess;
}

////////////////////////////////////////////////////////////////
hipError_t MoeKernelLauncher::Deinit(hipStream_t stream) {
  HIP_ERROR_CHECK(hipFree(m_internalState->ffn1Output_device));
  HIP_ERROR_CHECK(hipFree(m_internalState->ffn2Output_device));

  HIP_ERROR_CHECK(hipFreeAsync(m_internalState->ffn1SyncArrayDevice, stream));
  HIP_ERROR_CHECK(
      hipFreeAsync(m_internalState->chunkReduceSyncArrayDevice, stream));
  HIP_ERROR_CHECK(m_internalState->taskManagerHost.Deinit(stream));

  HIP_ERROR_CHECK(hipStreamSynchronize(stream));

  delete m_internalState;
  return hipSuccess;
}

////////////////////////////////////////////////////////////////
hipError_t MoeKernelLauncher::Launch(const void* tokens,
                                     const void* gateWeights,
                                     const void* ffn1ExpertWeights,
                                     const void* ffn2ExpertWeights,
                                     void* output, int tokensNum,
                                     hipStream_t stream) {
  HIP_ERROR_CHECK(moeKernelLaunchFunc<moe::MoeImplMetadata>(
      m_internalState, tokens, gateWeights, ffn1ExpertWeights,
      ffn2ExpertWeights, output, tokensNum, stream));
  return hipSuccess;
}

}  // namespace moe