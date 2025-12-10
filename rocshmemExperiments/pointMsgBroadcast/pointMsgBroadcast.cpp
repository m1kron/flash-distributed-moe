#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#include <iostream>
#include <rocshmem/rocshmem.hpp>

#define CHECK_HIP(condition)                                        \
  {                                                                 \
    hipError_t error = condition;                                   \
    if (error != hipSuccess) {                                      \
      fprintf(stderr, "HIP error: %d line: %d\n", error, __LINE__); \
      MPI_Abort(MPI_COMM_WORLD, error);                             \
    }                                                               \
  }

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

int main(int argc, char** argv) {
  const int rank = rocshmem_my_pe();
  int ndevices, my_device = 0;
  CHECK_HIP(hipGetDeviceCount(&ndevices));
  my_device = rank % ndevices;
  CHECK_HIP(hipSetDevice(my_device));
  int nelem = MAX_ELEM;

  if (argc > 1) {
    nelem = atoi(argv[1]);
  }

  rocshmem_init();
  const int npes = rocshmem_n_pes();
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

  uint64_t* data = (uint64_t*)rocshmem_malloc(nelem * sizeof(uint64_t));
  uint64_t* sig_addr = (uint64_t*)rocshmem_malloc(sizeof(uint64_t));
  if (NULL == data || NULL == message_device || NULL == sig_addr) {
    std::cout << "Error allocating memory from symmetric heap" << std::endl;
    std::cout << "data: " << data << ", message: " << message_device
              << ", size: " << sizeof(uint64_t) * nelem
              << ", sig_addr: " << sig_addr << std::endl;
    rocshmem_global_exit(1);
  }

  CHECK_HIP(hipMemset(data, 0, (nelem * sizeof(uint64_t))));
  CHECK_HIP(hipMemset(sig_addr, 0, (sizeof(uint64_t))));
  CHECK_HIP(hipDeviceSynchronize());

  int threadsPerBlock = MAX_ELEM;
  sendMsgToPeerKernel<<<dim3(1), dim3(threadsPerBlock), 0, 0>>>(
      data, message_device, nelem, sig_addr, rank, dst_pe);
  rocshmem_barrier_all();
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
  printf("[%d] Test %s \t %s\n", rank, argv[0], pass ? "[PASS]" : "[FAIL]");

  free(message_host);
  CHECK_HIP(hipFree(message_device));
  rocshmem_free(data);
  rocshmem_finalize();
  return 0;
}