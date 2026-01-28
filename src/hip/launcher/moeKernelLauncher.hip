#include "moeKernelLauncher.h"
#include "src/hip/common/runtimeConfig.h"
#include "src/hip/hipHostCommon.h"
#include "src/hip/kernel/moeKernel.h"
#include "src/hip/kernel/remote/remoteComManager.h"

namespace moe {

extern __constant__ TaskManagerType globalTaskManager;
extern __constant__ RemoteComManager globalRemoteComManager;
extern __constant__ int* globalFFN1SyncArray;
extern __constant__ int* globalChunkReduceSyncArray;
struct InternalState {
  TaskManagerType taskManagerHost;
  RemoteComManager remoteComManagerHost;
  const void* gateWeights;
  const void* ffn1ExpertWeights;
  const void* ffn2ExpertWeights;
  int* ffn1SyncArrayDevice;
  int* chunkReduceSyncArrayDevice;
  void* ffn1Output_device;
  void* ffn2Output_device;
  int maxBlocksPerSM;
  bool rocshmemInitialized = false;
};

template <typename MOE_METADATA>
int GetExpectedNumOfTasksForOneToken() {
  const int ffn1TasksNum =
      MOE_METADATA::MOE_PROBLEM_CONFIG::TOPK *
      (MOE_METADATA::MOE_PROBLEM_CONFIG::EXPERT_INTERMEDIATE_SIZE /
       MOE_METADATA::TILES_CONFIG::FFN1_TILE_METADATA::TILE_N);
  const int ffn2TasksNum =
      MOE_METADATA::MOE_PROBLEM_CONFIG::TOPK *
      (MOE_METADATA::MOE_PROBLEM_CONFIG::HIDDEN_SIZE /
       MOE_METADATA::TILES_CONFIG::FFN2_TILE_METADATA::TILE_N);
  const int reductionTasksNum =
      MOE_METADATA::TILES_CONFIG::REDUCTION_CHUNKS_PER_TOKEN;

  return ffn1TasksNum + ffn2TasksNum + reductionTasksNum;
}

template <typename MOE_METADATA>
int GetExpectedNumOfTasks(int tokensNum) {
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

  return GetExpectedNumOfTasksForOneToken<MOE_METADATA>() * tokensNum;
}

template <typename MOE_RUNTIME_CONFIG>
hipError_t moeKernelLaunchFunc(InternalState* state, const void* tokens,
                               void* output, int tokensNum,
                               hipStream_t stream) {
  using MOE_METADATA = typename MOE_RUNTIME_CONFIG::MOE_METADATA;
  using TMoeDataType = typename MOE_METADATA::MOE_PROBLEM_CONFIG::TDataType;
  dim3 grid(MOE_METADATA::HW_CONFIG::BLOCKS * state->maxBlocksPerSM);
  dim3 block(MOE_METADATA::HW_CONFIG::THREADS);

  const int expectedNumTasks = GetExpectedNumOfTasks<MOE_METADATA>(tokensNum);

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

  moeKernel<MOE_RUNTIME_CONFIG><<<grid, block, 0, stream>>>(
      static_cast<const TMoeDataType*>(tokens),
      static_cast<const TMoeDataType*>(state->gateWeights),
      static_cast<const TMoeDataType*>(state->ffn1ExpertWeights),
      static_cast<const TMoeDataType*>(state->ffn2ExpertWeights),
      static_cast<TMoeDataType*>(state->ffn1Output_device),
      static_cast<TMoeDataType*>(state->ffn2Output_device),
      static_cast<TMoeDataType*>(output), tokensNum);

  return hipSuccess;
}

////////////////////////////////////////////////////////////////////
hipError_t InitRocshmem(const moe::DistributedUniqueId& uid, int rank,
                        int worldSize) {
  rocshmem::rocshmem_uniqueid_t roc_uid;
  memcpy(roc_uid.data(), uid.data, sizeof(uid.data));

  rocshmem::rocshmem_init_attr_t attr;

  if (rocshmem::rocshmem_set_attr_uniqueid_args(rank, worldSize, &roc_uid,
                                                &attr) != 0) {
    MOE_ERROR_LOG("rocshmem_set_attr_uniqueid_args failed");
    return hipErrorUnknown;
  }
  if (rocshmem::rocshmem_init_attr(rocshmem::ROCSHMEM_INIT_WITH_UNIQUEID,
                                   &attr) != 0) {
    MOE_ERROR_LOG("rocshmem_init_attr failed");
    return hipErrorUnknown;
  }

  const int rocshmem_rank = rocshmem::rocshmem_my_pe();
  const int rocshmem_size = rocshmem::rocshmem_n_pes();

  MOE_ASSERT(rocshmem_rank == rank);
  MOE_ASSERT(rocshmem_size == worldSize);

  int GPU_id;
  HIP_ERROR_CHECK(hipGetDevice(&GPU_id));

  MOE_ASSERT(GPU_id == rank);

  MOE_LOG("[Rank %d/%d] Initialized rocSHMEM (rocSHMEM rank %d/%d) on GPU %d\n",
          rank, worldSize, rocshmem_rank, rocshmem_size, GPU_id);

  return hipSuccess;
}

////////////////////////////////////////////////////////////////
hipError_t MoeKernelLauncher::Init(const void* gateWeights,
                                   const void* ffn1ExpertWeights,
                                   const void* ffn2ExpertWeights, int maxTokens,
                                   hipStream_t stream,
                                   const moe::DistributedUniqueId& uid,
                                   int rank, int worldSize) {
  m_internalState = new InternalState();

  MOE_ASSERT(worldSize > 0);

  if (worldSize > 1) {
    HIP_ERROR_CHECK(InitRocshmem(uid, rank, worldSize));
    m_internalState->rocshmemInitialized = true;
  }
  m_internalState->remoteComManagerHost.Init(worldSize);

  m_internalState->gateWeights = gateWeights;
  m_internalState->ffn1ExpertWeights = ffn1ExpertWeights;
  m_internalState->ffn2ExpertWeights = ffn2ExpertWeights;

  HIP_ERROR_CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(
      &m_internalState->maxBlocksPerSM, moeKernel<moe::MoeRuntimeConfig>,
      moe::MoeImplMetadata::HW_CONFIG::THREADS, 0));
  MOE_LOG("Calculated max block per sm: %i", m_internalState->maxBlocksPerSM);

  // Task + tickets
  const int expectedMaxNumTasks =
      GetExpectedNumOfTasks<moe::MoeImplMetadata>(maxTokens) +
      m_internalState->maxBlocksPerSM * moe::MoeImplMetadata::HW_CONFIG::BLOCKS;

  if (expectedMaxNumTasks >
      m_internalState->taskManagerHost.GetTaskQueueSize()) {
    const int maxPossibleTokens =
        (m_internalState->taskManagerHost.GetTaskQueueSize() -
         m_internalState->maxBlocksPerSM *
             moe::MoeImplMetadata::HW_CONFIG::BLOCKS) /
        GetExpectedNumOfTasksForOneToken<moe::MoeImplMetadata>();

    MOE_ERROR_LOG(
        "Task queue size is not enough for max tokens: %i, current "
        "configuration will handle %i tokens.",
        maxTokens, maxPossibleTokens);

    return hipErrorUnknown;
  }

  // TODO: Allocate all mem using as a single mem block.

  const int ffn1OutputSize =
      maxTokens * moe::MoeImplMetadata::MOE_PROBLEM_CONFIG::TOPK *
      moe::MoeImplMetadata::MOE_PROBLEM_CONFIG::EXPERT_INTERMEDIATE_SIZE;
  const int ffn2OutputSize =
      maxTokens * moe::MoeImplMetadata::MOE_PROBLEM_CONFIG::TOPK *
      moe::MoeImplMetadata::MOE_PROBLEM_CONFIG::HIDDEN_SIZE;

  HIP_ERROR_CHECK(m_internalState->taskManagerHost.Init(stream));
  HIP_ERROR_CHECK(hipMallocAsync(
      &m_internalState->ffn1SyncArrayDevice,
      sizeof(int) * maxTokens * moe::MoeImplMetadata::MOE_PROBLEM_CONFIG::TOPK,
      stream));
  HIP_ERROR_CHECK(hipMallocAsync(
      &m_internalState->chunkReduceSyncArrayDevice,
      sizeof(int) * maxTokens *
          moe::MoeImplMetadata::TILES_CONFIG::REDUCTION_CHUNKS_PER_TOKEN,
      stream));
  HIP_ERROR_CHECK(hipMallocAsync(
      &m_internalState->ffn1Output_device,
      ffn1OutputSize *
          sizeof(moe::MoeImplMetadata::MOE_PROBLEM_CONFIG::TDataType),
      stream));
  HIP_ERROR_CHECK(hipMallocAsync(
      &m_internalState->ffn2Output_device,
      ffn2OutputSize *
          sizeof(moe::MoeImplMetadata::MOE_PROBLEM_CONFIG::TDataType),
      stream));

  HIP_ERROR_CHECK(hipMemcpyToSymbolAsync(HIP_SYMBOL(globalRemoteComManager),
                                         &m_internalState->remoteComManagerHost,
                                         sizeof(RemoteComManager), 0,
                                         hipMemcpyHostToDevice, stream));

  HIP_ERROR_CHECK(hipMemcpyToSymbolAsync(
      HIP_SYMBOL(globalTaskManager), &m_internalState->taskManagerHost,
      sizeof(TaskManagerType), 0, hipMemcpyHostToDevice, stream));

  HIP_ERROR_CHECK(
      hipMemcpyToSymbolAsync(HIP_SYMBOL(globalChunkReduceSyncArray),
                             &m_internalState->chunkReduceSyncArrayDevice,
                             sizeof(int*), 0, hipMemcpyHostToDevice, stream));
  HIP_ERROR_CHECK(hipMemcpyToSymbolAsync(
      HIP_SYMBOL(globalFFN1SyncArray), &m_internalState->ffn1SyncArrayDevice,
      sizeof(int*), 0, hipMemcpyHostToDevice, stream));

  HIP_ERROR_CHECK(hipStreamSynchronize(stream));

  return hipSuccess;
}

////////////////////////////////////////////////////////////////
hipError_t MoeKernelLauncher::Deinit(hipStream_t stream) {
  HIP_ERROR_CHECK(hipFreeAsync(m_internalState->ffn1Output_device, stream));
  HIP_ERROR_CHECK(hipFreeAsync(m_internalState->ffn2Output_device, stream));

  HIP_ERROR_CHECK(hipFreeAsync(m_internalState->ffn1SyncArrayDevice, stream));
  HIP_ERROR_CHECK(
      hipFreeAsync(m_internalState->chunkReduceSyncArrayDevice, stream));
  HIP_ERROR_CHECK(m_internalState->taskManagerHost.Deinit(stream));
  m_internalState->remoteComManagerHost.Deinit();

  HIP_ERROR_CHECK(hipStreamSynchronize(stream));

  if (m_internalState->rocshmemInitialized) {
    rocshmem_finalize();
  }

  delete m_internalState;
  return hipSuccess;
}

////////////////////////////////////////////////////////////////
hipError_t MoeKernelLauncher::Launch(const void* tokens, void* output,
                                     int tokensNum, hipStream_t stream) {
  HIP_ERROR_CHECK(moeKernelLaunchFunc<moe::MoeRuntimeConfig>(
      m_internalState, tokens, output, tokensNum, stream));
  return hipSuccess;
}

}  // namespace moe