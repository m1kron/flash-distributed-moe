#include "hip/launcher/moeKernelLauncher.h"

#include <rocshmem/rocshmem.hpp>

#include "hip/hipHostCommon.h"
#include "include/iMoeKernelLauncher.h"

#define ROCSHMEM_ERROR_ASSERT(condition...)          \
  {                                                  \
    const int error = (condition);                   \
    if (error != 0) {                                \
      printf("ROCShMem failed on %s\n", #condition); \
    }                                                \
  }

namespace moe {

////////////////////////////////////////////////////////////////////
IMoeKernelLauncher::~IMoeKernelLauncher() {}

}  // namespace moe

using namespace rocshmem;

__global__ void sendMsgToPeerKernel(uint64_t* data, uint64_t* message,
                                    size_t nelem, uint64_t* sig_addr, int my_pe,
                                    int dst_pe) {
  int threadId = blockIdx.x * blockDim.x + threadIdx.x;

  if (my_pe == 0) {
    rocshmem_ulong_put_signal_wg(data, message, nelem, sig_addr, 1,
                                 ROCSHMEM_SIGNAL_SET, dst_pe);
    if (threadId == 0) {
      rocshmem_ulong_wait_until(sig_addr, ROCSHMEM_CMP_EQ, 1);
    }
  } else {
    if (threadId == 0) {
      rocshmem_ulong_wait_until(sig_addr, ROCSHMEM_CMP_EQ, 1);
    }
    __syncthreads();
    data[threadIdx.x] = data[threadIdx.x] + 1;
    rocshmem_ulong_put_signal_wg(data, data, nelem, sig_addr, 1,
                                 ROCSHMEM_SIGNAL_SET, dst_pe);
  }
  __syncthreads();
}

#define MAX_ELEM 1024

#define CHECK_HIP(condition)                                        \
  {                                                                 \
    hipError_t error = condition;                                   \
    if (error != hipSuccess) {                                      \
      fprintf(stderr, "HIP error: %d line: %d\n", error, __LINE__); \
      MPI_Abort(MPI_COMM_WORLD, error);                             \
    }                                                               \
  }

////////////////////////////////////////////////////////////////////
extern "C" moe::DistributedUniqueId getDistributedUniqueId(bool empty) {
  moe::DistributedUniqueId dist_uid;
  if (!empty) {
    rocshmem::rocshmem_uniqueid_t uid;
    if (rocshmem::rocshmem_get_uniqueid(&uid) != 0)
      MOE_ERROR_LOG("rocshmem_get_uniqueid failed");

    static_assert(sizeof(rocshmem::rocshmem_uniqueid_t) ==
                  sizeof(moe::DistributedUniqueId));

    memcpy(dist_uid.data, uid.data(), sizeof(dist_uid.data));
  } else {
    memset(dist_uid.data, 0, sizeof(dist_uid.data));
  }
  return dist_uid;
}

////////////////////////////////////////////////////////////////////
extern "C" void InitializeDistributed(const moe::DistributedUniqueId& uid,
                                      int rank, int worldSize) {
  rocshmem::rocshmem_uniqueid_t roc_uid;
  memcpy(roc_uid.data(), uid.data, sizeof(uid.data));

  rocshmem::rocshmem_init_attr_t attr;

  ROCSHMEM_ERROR_ASSERT(rocshmem::rocshmem_set_attr_uniqueid_args(
      rank, worldSize, &roc_uid, &attr));
  ROCSHMEM_ERROR_ASSERT(rocshmem::rocshmem_init_attr(
      rocshmem::ROCSHMEM_INIT_WITH_UNIQUEID, &attr));

  const int rocshmem_rank = rocshmem::rocshmem_my_pe();
  const int rocshmem_size = rocshmem::rocshmem_n_pes();

  int GPU_id;
  CHECK_HIP(hipGetDevice(&GPU_id));

  printf("[Rank %d/%d] Initialized rocSHMEM (rocSHMEM rank %d/%d) on GPU %d\n",
         rank, worldSize, rocshmem_rank, rocshmem_size, GPU_id);

  // ---------------------------------------------------------
  const int nelem = MAX_ELEM;
  const int npes = rocshmem::rocshmem_n_pes();
  const int dst_pe = (rank + 1) % npes;
  const int prev_pe = (rank - 1 + npes) % npes;
  uint64_t* message_host = (uint64_t*)malloc(nelem * sizeof(uint64_t));
  constexpr int msgVal = 12345;

  for (int i = 0; i < nelem; i++) {
    message_host[i] = msgVal;
  }

  uint64_t* message_device;
  CHECK_HIP(hipMalloc(&message_device, nelem * sizeof(uint64_t)));
  CHECK_HIP(hipMemcpy(message_device, message_host, nelem * sizeof(uint64_t),
                      hipMemcpyHostToDevice));

  uint64_t* data =
      (uint64_t*)rocshmem::rocshmem_malloc(nelem * sizeof(uint64_t));
  uint64_t* sig_addr = (uint64_t*)rocshmem::rocshmem_malloc(sizeof(uint64_t));
  if (NULL == data || NULL == message_device || NULL == sig_addr) {
    rocshmem::rocshmem_global_exit(1);
  }

  CHECK_HIP(hipMemset(data, 0, (nelem * sizeof(uint64_t))));
  CHECK_HIP(hipMemset(sig_addr, 0, (sizeof(uint64_t))));
  CHECK_HIP(hipDeviceSynchronize());

  int threadsPerBlock = MAX_ELEM;
  sendMsgToPeerKernel<<<dim3(1), dim3(threadsPerBlock), 0, 0>>>(
      data, message_device, nelem, sig_addr, rank, dst_pe);
  rocshmem::rocshmem_barrier_all();
  CHECK_HIP(hipDeviceSynchronize());

  bool pass = true;
  const int expected = rank == 0 ? msgVal + npes - 1 : msgVal + rank;
  for (int i = 0; i < nelem; i++) {
    if (data[i] != expected) {
      pass = false;
      printf("[%d] Error in element %d expected %d got %lu\n", rank, i,
             expected, data[i]);
    }
  }
  printf("[%d] Test %s \t %s\n", rank, "vllm", pass ? "[PASS]" : "[FAIL]");

  free(message_host);
  CHECK_HIP(hipFree(message_device));
  rocshmem::rocshmem_free(data);
  // ---------------------------------------------------------

  rocshmem::rocshmem_finalize();
}

////////////////////////////////////////////////////////////////////
extern "C" hipError_t CreateLauncher(moe::IMoeKernelLauncher** launcher,
                                     const void* gateWeights,
                                     const void* ffn1ExpertWeights,
                                     const void* ffn2ExpertWeights,
                                     int maxTokens, hipStream_t stream,
                                     const moe::DistributedUniqueId& uid,
                                     int rank, int worldSize) {
  moe::MoeKernelLauncher* _launcher = new moe::MoeKernelLauncher();
  HIP_ERROR_CHECK(_launcher->Init(gateWeights, ffn1ExpertWeights,
                                  ffn2ExpertWeights, maxTokens, stream, uid,
                                  rank, worldSize));
  *launcher = _launcher;

  if (worldSize > 1) {
    // ---------------------------------------------------------
    const int nelem = MAX_ELEM;
    const int npes = rocshmem::rocshmem_n_pes();
    const int dst_pe = (rank + 1) % npes;
    const int prev_pe = (rank - 1 + npes) % npes;
    uint64_t* message_host = (uint64_t*)malloc(nelem * sizeof(uint64_t));
    constexpr int msgVal = 12345;

    for (int i = 0; i < nelem; i++) {
      message_host[i] = msgVal;
    }

    uint64_t* message_device;
    CHECK_HIP(hipMalloc(&message_device, nelem * sizeof(uint64_t)));
    CHECK_HIP(hipMemcpy(message_device, message_host, nelem * sizeof(uint64_t),
                        hipMemcpyHostToDevice));

    uint64_t* data =
        (uint64_t*)rocshmem::rocshmem_malloc(nelem * sizeof(uint64_t));
    uint64_t* sig_addr = (uint64_t*)rocshmem::rocshmem_malloc(sizeof(uint64_t));
    if (NULL == data || NULL == message_device || NULL == sig_addr) {
      rocshmem::rocshmem_global_exit(1);
    }

    CHECK_HIP(hipMemset(data, 0, (nelem * sizeof(uint64_t))));
    CHECK_HIP(hipMemset(sig_addr, 0, (sizeof(uint64_t))));
    CHECK_HIP(hipDeviceSynchronize());

    int threadsPerBlock = MAX_ELEM;
    sendMsgToPeerKernel<<<dim3(1), dim3(threadsPerBlock), 0, 0>>>(
        data, message_device, nelem, sig_addr, rank, dst_pe);
    rocshmem::rocshmem_barrier_all();
    CHECK_HIP(hipDeviceSynchronize());

    bool pass = true;
    const int expected = rank == 0 ? msgVal + npes - 1 : msgVal + rank;
    for (int i = 0; i < nelem; i++) {
      if (data[i] != expected) {
        pass = false;
        printf("[%d] Error in element %d expected %d got %lu\n", rank, i,
               expected, data[i]);
      }
    }
    printf("[%d] Test %s \t %s\n", rank, "vllm", pass ? "[PASS]" : "[FAIL]");

    free(message_host);
    CHECK_HIP(hipFree(message_device));
    rocshmem::rocshmem_free(data);
    // ---------------------------------------------------------
  }

  return hipSuccess;
}

////////////////////////////////////////////////////////////////////
extern "C" hipError_t DestroyLauncher(moe::IMoeKernelLauncher* launcher,
                                      hipStream_t stream) {
  HIP_ERROR_CHECK(
      static_cast<moe::MoeKernelLauncher*>(launcher)->Deinit(stream));
  delete launcher;
  return hipSuccess;
}