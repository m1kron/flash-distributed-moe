#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#include <iostream>
#include <rocshmem/rocshmem.hpp>

const int TOKENS_PER_GPU = 4;
const int TOKEN_SIZE = 2048;
const int THREADS = 1024;
const int BLOCKS = 5;
const int ALLOC_SLOT_SIZE = 16;
const int TOKEN_HEADER_SIZE = 2;  // -> pe | pe's token index.

void PrintOutput(const std::vector<float>& output_host, int rank,
                 int stride = TOKEN_SIZE) {
  std::string outputString;
  outputString += "\nRank " + std::to_string(rank) + ":\n";
  for (int tokenIdx = 0; tokenIdx < TOKENS_PER_GPU; tokenIdx++) {
    outputString += "Token " + std::to_string(tokenIdx) + ":\n";
    for (int i = 0; i < 5; i++) {
      int globalIdx = tokenIdx * stride + i;
      outputString += std::to_string(output_host[globalIdx]) + " ";
    }
    outputString += "\n";
  }
  outputString += "\n";

  std::cout << outputString;
}

void VerifyOutput(const std::vector<float>& input,
                  const std::vector<int>& routingTable,
                  const std::vector<float>& output_host, int rank) {
  bool allCorrect = true;
  for (int tokenIdx = 0; tokenIdx < TOKENS_PER_GPU; tokenIdx++) {
    const float routeVal = (float)routingTable[tokenIdx];
    for (int i = 0; i < TOKEN_SIZE; i++) {
      int globalIdx = tokenIdx * TOKEN_SIZE + i;
      const float expectedValue = routeVal + input[globalIdx];
      if (output_host[globalIdx] != expectedValue) {
        allCorrect = false;
        std::cout << "Rank " << rank << " found incorrect value at token "
                  << tokenIdx << " index " << i << ": "
                  << output_host[globalIdx] << " != " << expectedValue << "\n";
        break;
      }
    }
  }
  if (allCorrect) {
    std::cout << "Rank " << rank << " output is correct.\n";
  }
}

#define CHECK_HIP(condition)                                        \
  {                                                                 \
    hipError_t error = condition;                                   \
    if (error != hipSuccess) {                                      \
      fprintf(stderr, "HIP error: %d line: %d\n", error, __LINE__); \
      MPI_Abort(MPI_COMM_WORLD, error);                             \
    }                                                               \
  }

using namespace rocshmem;

__global__ void moeLikeCommKernel(const float* input, float* output,
                                  const int* routingTable,
                                  float* remoteTokenMemPool_sym,
                                  int* remoteTokenMemPoolFirstFreeIdx_sym,
                                  uint64_t* remoteTokenMemPoolSlotStatus_sym,
                                  int* doneSendingFlag_sym,
                                  int* sendTokens_global, int* tokenIdx_global,
                                  int numTokens, int my_pe, int npes) {
  // 1. Send input to destination PEs based on routing table.
  for (int thisBlockTokenIdx = blockIdx.x; thisBlockTokenIdx < numTokens;
       thisBlockTokenIdx += gridDim.x) {
    const int dst_pe = routingTable[thisBlockTokenIdx];

    if (threadIdx.x == 0) {
      // "Allocate" a slot in the remote PE's token memory pool.
      const int idx = rocshmem_int_atomic_fetch_inc(
          remoteTokenMemPoolFirstFreeIdx_sym, dst_pe);
      const float* thisBlockInput = input + thisBlockTokenIdx * TOKEN_SIZE;
      printf("PE %d sending token %d to PE %d into slot %d\n", my_pe,
             thisBlockTokenIdx, dst_pe, idx);

      float* remotePoolPtr =
          remoteTokenMemPool_sym + idx * (TOKEN_SIZE + TOKEN_HEADER_SIZE);
      const float my_pe_f = (float)my_pe;
      const float tokenIdx_f = (float)(thisBlockTokenIdx);
      rocshmem_float_put(remotePoolPtr, &my_pe_f, 1, dst_pe);
      rocshmem_float_put(remotePoolPtr + 1, &tokenIdx_f, 1, dst_pe);
      rocshmem_float_put_signal(remotePoolPtr + TOKEN_HEADER_SIZE,
                                thisBlockInput, TOKEN_SIZE,
                                remoteTokenMemPoolSlotStatus_sym + idx, 1,
                                ROCSHMEM_SIGNAL_SET, dst_pe);

      rocshmem_fence();
      // const int prev = atomicAdd(sendTokens_global, 1);

      const int prev = __hip_atomic_fetch_add(
          sendTokens_global, 1, __ATOMIC_SEQ_CST, __HIP_MEMORY_SCOPE_AGENT);

      // Last block informs all PEs that sending is done.
      if (prev == (numTokens - 1)) {
        int dst = my_pe;
        for (int i = 0; i < npes - 1; ++i) {
          dst = (dst + 1) % npes;
          rocshmem_int_atomic_inc(doneSendingFlag_sym, dst);
        }
        printf("PE %d done sending all tokens. \n", my_pe);
      }
    }
    __syncthreads();
  }

  // 2. Each block processes received tokens.
  if (threadIdx.x == 0) {
    bool shouldStop = false;
    while (!shouldStop) {
      printf("PE %d block %d starting to receive tokens.\n", my_pe, blockIdx.x);

      const int slot = atomicAdd(tokenIdx_global, 1);

      int slotStatus = rocshmem_uint64_atomic_fetch(
          remoteTokenMemPoolSlotStatus_sym + slot, my_pe);
      while (slotStatus == 0 && !shouldStop) {
        slotStatus = rocshmem_uint64_atomic_fetch(
            remoteTokenMemPoolSlotStatus_sym + slot, my_pe);

        shouldStop = rocshmem_int_test(doneSendingFlag_sym, ROCSHMEM_CMP_EQ,
                                       npes - 1) == 1;
      }

      if (shouldStop) break;

      printf("PE %d block %d received token in slot %d.\n", my_pe, blockIdx.x,
             slot);
    }
  }

  __syncthreads();

  // 2. Wait until all PEs are done sending.
  if (threadIdx.x == 0) {
    rocshmem_int_wait_until(doneSendingFlag_sym, ROCSHMEM_CMP_EQ, npes - 1);
    const int receivedTokens =
        rocshmem_int_atomic_fetch(remoteTokenMemPoolFirstFreeIdx_sym, my_pe);
    printf("PE %d detected all PEs done sending. Received %d tokens.\n", my_pe,
           receivedTokens);
  }

  __syncthreads();
}

int main(int argc, char** argv) {
  const int rank = rocshmem_my_pe();
  int ndevices, my_device = 0;
  CHECK_HIP(hipGetDeviceCount(&ndevices));
  my_device = rank % ndevices;
  CHECK_HIP(hipSetDevice(my_device));
  rocshmem_init();
  const int worldSize = rocshmem_n_pes();

  const int INPUT_SIZE = TOKENS_PER_GPU * TOKEN_SIZE;

  std::vector<int> routingTable(TOKENS_PER_GPU);
  for (int i = 0; i < TOKENS_PER_GPU; i++) {
    int proposition = i % worldSize;
    if (proposition == rank) {
      proposition = (proposition + 1) % worldSize;
    }
    routingTable[i] = proposition;
  }

  std::vector<float> input_host(INPUT_SIZE, (float)rank);

  int* routingTable_device;
  CHECK_HIP(hipMalloc(&routingTable_device, TOKENS_PER_GPU * sizeof(int)));
  CHECK_HIP(hipMemcpy(routingTable_device, routingTable.data(),
                      TOKENS_PER_GPU * sizeof(int), hipMemcpyHostToDevice));

  float* input_device;
  CHECK_HIP(hipMalloc(&input_device, INPUT_SIZE * sizeof(float)));
  CHECK_HIP(hipMemcpy(input_device, input_host.data(),
                      INPUT_SIZE * sizeof(float), hipMemcpyHostToDevice));

  int* sendTokens_global;
  CHECK_HIP(hipMalloc(&sendTokens_global, sizeof(int)));
  CHECK_HIP(hipMemset(sendTokens_global, 0, sizeof(int)));

  float* remoteTokenMemPool_sym = (float*)rocshmem_malloc(
      ALLOC_SLOT_SIZE * (TOKEN_SIZE + TOKEN_HEADER_SIZE) * sizeof(float));

  int* remoteTokenMemPoolFirstFreeIdx_sym = (int*)rocshmem_malloc(sizeof(int));
  CHECK_HIP(hipMemset(remoteTokenMemPoolFirstFreeIdx_sym, 0, sizeof(int)));

  uint64_t* remoteTokenMemPoolSlotStatus_sym =
      (uint64_t*)rocshmem_malloc(ALLOC_SLOT_SIZE * sizeof(uint64_t));
  CHECK_HIP(hipMemset(remoteTokenMemPoolSlotStatus_sym, 0,
                      ALLOC_SLOT_SIZE * sizeof(uint64_t)));

  int* tokenIdx_global;
  CHECK_HIP(hipMalloc(&tokenIdx_global, sizeof(int)));
  CHECK_HIP(hipMemset(tokenIdx_global, 0, sizeof(int)));

  int* doneSendingFlag_sym = (int*)rocshmem_malloc(sizeof(int));
  CHECK_HIP(hipMemset(doneSendingFlag_sym, 0, sizeof(int)));

  float* output_device;
  CHECK_HIP(hipMalloc(&output_device, INPUT_SIZE * sizeof(float)));

  if (rank == 0) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    std::cout << "Rank " << rank << " launching kernel on device " << my_device
              << "\n";
  }

  moeLikeCommKernel<<<BLOCKS, THREADS>>>(
      input_device, output_device, routingTable_device, remoteTokenMemPool_sym,
      remoteTokenMemPoolFirstFreeIdx_sym, remoteTokenMemPoolSlotStatus_sym,
      doneSendingFlag_sym, sendTokens_global, tokenIdx_global, TOKENS_PER_GPU,
      rank, worldSize);

  rocshmem_barrier_all();
  CHECK_HIP(hipDeviceSynchronize());

  //  std::vector<float> output_host(INPUT_SIZE, 0);
  // CHECK_HIP(hipMemcpy(output_host.data(), output_device,
  //                    INPUT_SIZE * sizeof(float), hipMemcpyDeviceToHost));
  std::vector<float> output_host(
      TOKENS_PER_GPU * (TOKEN_SIZE + TOKEN_HEADER_SIZE), 0);
  CHECK_HIP(hipMemcpy(
      output_host.data(), remoteTokenMemPool_sym,
      TOKENS_PER_GPU * (TOKEN_SIZE + TOKEN_HEADER_SIZE) * sizeof(float),
      hipMemcpyDeviceToHost));

  CHECK_HIP(hipFree(output_device));
  rocshmem_free(remoteTokenMemPool_sym);
  CHECK_HIP(hipFree(input_device));
  CHECK_HIP(hipFree(routingTable_device));

  rocshmem_finalize();

  PrintOutput(output_host, rank, TOKEN_SIZE + TOKEN_HEADER_SIZE);
  // VerifyOutput(input_host, routingTable, output_host, rank);
  return 0;
}