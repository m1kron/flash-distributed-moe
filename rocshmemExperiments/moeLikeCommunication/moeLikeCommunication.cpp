#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#include <iostream>
#include <rocshmem/rocshmem.hpp>

const int TOKENS_PER_GPU = 4;
const int TOKEN_SIZE = 2048;
const int THREADS = 1024;
const int BLOCKS = 8;
const int ALLOC_SLOT_SIZE = 64;
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

__device__ void sendMsg(int dst_pe, int msg_pe, int tokenIdx,
                        const float* tokenData, float* remoteTokenMemPool_sym,
                        int* remoteTokenMemPoolFirstFreeIdx_sym,
                        uint64_t* remoteTokenMemPoolSlotStatus_sym) {
  if (threadIdx.x == 0) {
    // "Allocate" a slot in the remote PE's token memory pool.
    const int idx = rocshmem_int_atomic_fetch_inc(
        remoteTokenMemPoolFirstFreeIdx_sym, dst_pe);

    printf("PE %d sending token %d to PE %d into slot %d\n", rocshmem_my_pe(),
           tokenIdx, dst_pe, idx);

    float* remotePoolPtr =
        remoteTokenMemPool_sym + idx * (TOKEN_SIZE + TOKEN_HEADER_SIZE);
    const float my_pe_f = (float)msg_pe;
    const float tokenIdx_f = (float)(tokenIdx);
    rocshmem_float_put(remotePoolPtr, &my_pe_f, 1, dst_pe);
    rocshmem_float_put(remotePoolPtr + 1, &tokenIdx_f, 1, dst_pe);
    rocshmem_float_put_signal(
        remotePoolPtr + TOKEN_HEADER_SIZE, tokenData, TOKEN_SIZE,
        remoteTokenMemPoolSlotStatus_sym + idx, 1, ROCSHMEM_SIGNAL_SET, dst_pe);

    rocshmem_fence();
  }
}

__device__ void processMsg(int src_pe, int tokenIdx, float* token,
                           float* output, float* remoteTokenMemPool_sym,
                           int* remoteTokenMemPoolFirstFreeIdx_sym,
                           uint64_t* remoteTokenMemPoolSlotStatus_sym,
                           int* numOutputTokens_global) {
  if (threadIdx.x == 0) {
    if (src_pe == -1) {
      printf("PE %d: Received processed token, %d, saving to output.\n",
             rocshmem_my_pe(), tokenIdx);

      // Save the processed token to the output buffer.
      rocshmem_float_put(output + tokenIdx * TOKEN_SIZE, token, TOKEN_SIZE,
                         rocshmem_my_pe());

      __hip_atomic_fetch_add(numOutputTokens_global, 1, __ATOMIC_SEQ_CST,
                             __HIP_MEMORY_SCOPE_AGENT);

      return;
    } else {
      printf("PE %d: Received token to process form %d.\n", rocshmem_my_pe(),
             src_pe);

      for (int i = 0; i < TOKEN_SIZE; i++) {
        token[i] += (float)rocshmem_my_pe();
      }

      printf("PE %d: DONE processing token, %d, sending back to PE %d.\n",
             rocshmem_my_pe(), tokenIdx, src_pe);

      sendMsg(src_pe, -1, tokenIdx, token, remoteTokenMemPool_sym,
              remoteTokenMemPoolFirstFreeIdx_sym,
              remoteTokenMemPoolSlotStatus_sym);
      return;
    }
  }
}

__global__ void moeLikeCommKernel(
    const float* input, float* output, const int* routingTable,
    float* remoteTokenMemPool_sym, int* remoteTokenMemPoolFirstFreeIdx_sym,
    uint64_t* remoteTokenMemPoolSlotStatus_sym, int* doneSendingFlag_sym,
    int* sendTokens_global, int* numOutputTokens_global,
    int* reserverdSlotIdx_global, int numTokens, int my_pe, int npes) {
  // 1. Send input to destination PEs based on routing table.
  for (int thisBlockTokenIdx = blockIdx.x; thisBlockTokenIdx < numTokens;
       thisBlockTokenIdx += gridDim.x) {
    const int dst_pe = routingTable[thisBlockTokenIdx];

    const float* thisBlockInput = input + thisBlockTokenIdx * TOKEN_SIZE;

    sendMsg(dst_pe, my_pe, thisBlockTokenIdx, thisBlockInput,
            remoteTokenMemPool_sym, remoteTokenMemPoolFirstFreeIdx_sym,
            remoteTokenMemPoolSlotStatus_sym);

    if (threadIdx.x == 0) {
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

  // 2. Each block processes received msgs.
  if (threadIdx.x == 0) {
    printf("PE %d block %d started to receive msgs.\n", my_pe, blockIdx.x);

    while (true) {
      const int slot =
          __hip_atomic_fetch_add(reserverdSlotIdx_global, 1, __ATOMIC_SEQ_CST,
                                 __HIP_MEMORY_SCOPE_AGENT);

      int slotStatus = rocshmem_uint64_atomic_fetch(
          remoteTokenMemPoolSlotStatus_sym + slot, my_pe);

      printf("PE %d block %d waiting for slot %d to be filled.\n", my_pe,
             blockIdx.x, slot);

      bool shouldStop = false;

      while (slotStatus == 0 && !shouldStop) {
        slotStatus = rocshmem_uint64_atomic_fetch(
            remoteTokenMemPoolSlotStatus_sym + slot, my_pe);

        const bool allDoneSending =
            rocshmem_int_test(doneSendingFlag_sym, ROCSHMEM_CMP_EQ, npes - 1) ==
            1;

        const bool allTokensSaved =
            __hip_atomic_load(numOutputTokens_global, __ATOMIC_RELAXED,
                              __HIP_MEMORY_SCOPE_AGENT) == TOKENS_PER_GPU;

        shouldStop = allDoneSending && allTokensSaved;
      }

      if (shouldStop) {
        printf("PE %d block %d received stop signal.\n", my_pe, blockIdx.x);
        break;
      }

      float* remotePoolPtr =
          remoteTokenMemPool_sym + slot * (TOKEN_SIZE + TOKEN_HEADER_SIZE);
      float pe;
      float tokenIdx;
      rocshmem_float_get(&pe, remotePoolPtr, 1, my_pe);
      rocshmem_float_get(&tokenIdx, remotePoolPtr + 1, 1, my_pe);
      int peInt = (int)pe;
      int tokenIdxInt = (int)tokenIdx;

      printf(
          "PE %d block %d received msg in slot %d: from PE %d, tokenIdx %d\n",
          my_pe, blockIdx.x, slot, peInt, tokenIdxInt);

      processMsg(peInt, tokenIdxInt, remotePoolPtr + TOKEN_HEADER_SIZE, output,
                 remoteTokenMemPool_sym, remoteTokenMemPoolFirstFreeIdx_sym,
                 remoteTokenMemPoolSlotStatus_sym, numOutputTokens_global);
    }
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

  int* numOutputTokens_global;
  CHECK_HIP(hipMalloc(&numOutputTokens_global, sizeof(int)));
  CHECK_HIP(hipMemset(numOutputTokens_global, 0, sizeof(int)));

  float* remoteTokenMemPool_sym = (float*)rocshmem_malloc(
      ALLOC_SLOT_SIZE * (TOKEN_SIZE + TOKEN_HEADER_SIZE) * sizeof(float));

  int* remoteTokenMemPoolFirstFreeIdx_sym = (int*)rocshmem_malloc(sizeof(int));
  CHECK_HIP(hipMemset(remoteTokenMemPoolFirstFreeIdx_sym, 0, sizeof(int)));

  uint64_t* remoteTokenMemPoolSlotStatus_sym =
      (uint64_t*)rocshmem_malloc(ALLOC_SLOT_SIZE * sizeof(uint64_t));
  CHECK_HIP(hipMemset(remoteTokenMemPoolSlotStatus_sym, 0,
                      ALLOC_SLOT_SIZE * sizeof(uint64_t)));

  int* reserverdSlotIdx_global;
  CHECK_HIP(hipMalloc(&reserverdSlotIdx_global, sizeof(int)));
  CHECK_HIP(hipMemset(reserverdSlotIdx_global, 0, sizeof(int)));

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
      doneSendingFlag_sym, sendTokens_global, numOutputTokens_global,
      reserverdSlotIdx_global, TOKENS_PER_GPU, rank, worldSize);

  rocshmem_barrier_all();
  CHECK_HIP(hipDeviceSynchronize());

  std::vector<float> output_host(INPUT_SIZE, 0);
  CHECK_HIP(hipMemcpy(output_host.data(), output_device,
                      INPUT_SIZE * sizeof(float), hipMemcpyDeviceToHost));

  CHECK_HIP(hipFree(output_device));
  rocshmem_free(remoteTokenMemPool_sym);
  CHECK_HIP(hipFree(input_device));
  CHECK_HIP(hipFree(routingTable_device));

  rocshmem_finalize();

  // PrintOutput(output_host, rank, TOKEN_SIZE);
  VerifyOutput(input_host, routingTable, output_host, rank);
  return 0;
}