#include <chrono>
#include <iostream>
#include <thread>

#include "test/multiGPU/cpp/rocshmem/rocshmemCommon.h"

namespace {
const int TOKENS_PER_GPU = 128;
const int TOKEN_SIZE = 2048;
const int THREADS = 1024;
const int BLOCKS = 304;
const int ALLOC_SLOT_SIZE = 1024;
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
      ASSERT_EQ(expectedValue, output_host[globalIdx])
          << "Rank " << rank << " found incorrect value at token " << tokenIdx
          << " index " << i << ": " << output_host[globalIdx]
          << " != " << expectedValue << "\n";
    }
  }
  if (allCorrect) {
    std::cout << "Rank " << rank << " output is correct.\n";
  }
}

#ifdef NDEBUG
#define HIP_DEVICE_LOG(text, ...) (void)0
#else
#define HIP_DEVICE_LOG(STR, ARGS...) \
  printf("[BIdx:%d TIdX:%d]: " STR, blockIdx.x, threadIdx.x, ##ARGS);
#endif

using namespace rocshmem;

__device__ void sendMsg_block(int dst_pe, int msg_pe, int tokenIdx,
                              const float* tokenData,
                              float* remoteTokenMemPool_sym,
                              int* remoteTokenMemPoolFirstFreeIdx_sym,
                              uint64_t* remoteTokenMemPoolSlotStatus_sym) {
  __shared__ int idx_shared;
  if (threadIdx.x == 0) {
    // "Allocate" a slot in the remote PE's token memory pool.
    const int idx = rocshmem_int_atomic_fetch_inc(
        remoteTokenMemPoolFirstFreeIdx_sym, dst_pe);

    HIP_DEVICE_LOG("PE %d sending token %d to PE %d into slot %d\n",
                   rocshmem_my_pe(), tokenIdx, dst_pe, idx);

    idx_shared = idx;
  }
  __syncthreads();

  const int idx = idx_shared;

  float* remotePoolPtr =
      remoteTokenMemPool_sym + idx * (TOKEN_SIZE + TOKEN_HEADER_SIZE);

  if (threadIdx.x == 0) {
    const float my_pe_f = (float)msg_pe;
    const float tokenIdx_f = (float)tokenIdx;
    rocshmem_float_put_nbi(remotePoolPtr, &my_pe_f, 1, dst_pe);
    rocshmem_float_put_nbi(remotePoolPtr + 1, &tokenIdx_f, 1, dst_pe);
  }

  rocshmem_float_put_signal_nbi_wg(
      remotePoolPtr + TOKEN_HEADER_SIZE, tokenData, TOKEN_SIZE,
      remoteTokenMemPoolSlotStatus_sym + idx, 1, ROCSHMEM_SIGNAL_SET, dst_pe);
}

__device__ void processMsg_block(int src_pe, int tokenIdx, float* token,
                                 float* output, float* remoteTokenMemPool_sym,
                                 int* remoteTokenMemPoolFirstFreeIdx_sym,
                                 uint64_t* remoteTokenMemPoolSlotStatus_sym,
                                 int* numOutputTokens_global) {
  if (src_pe == -1) {
    if (threadIdx.x == 0) {
      HIP_DEVICE_LOG("PE %d: Received processed token, %d, saving to output.\n",
                     rocshmem_my_pe(), tokenIdx);
    }
    float* thisBlockOutput = output + tokenIdx * TOKEN_SIZE;

    // Save the processed token to the output buffer.
    for (int i = threadIdx.x; i < TOKEN_SIZE; i += blockDim.x) {
      thisBlockOutput[i] = token[i];
    }

    if (threadIdx.x == 0) {
      __hip_atomic_fetch_add(numOutputTokens_global, 1, __ATOMIC_RELAXED,
                             __HIP_MEMORY_SCOPE_AGENT);
    }
  } else {
    if (threadIdx.x == 0) {
      HIP_DEVICE_LOG("PE %d: Received token to process form %d.\n",
                     rocshmem_my_pe(), src_pe);
    }

    for (int i = threadIdx.x; i < TOKEN_SIZE; i += blockDim.x) {
      token[i] += (float)rocshmem_my_pe();
    }
    if (threadIdx.x == 0) {
      HIP_DEVICE_LOG(
          "PE %d: DONE processing token, %d, sending back to PE %d.\n",
          rocshmem_my_pe(), tokenIdx, src_pe);
    }
    sendMsg_block(src_pe, -1, tokenIdx, token, remoteTokenMemPool_sym,
                  remoteTokenMemPoolFirstFreeIdx_sym,
                  remoteTokenMemPoolSlotStatus_sym);
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

    sendMsg_block(dst_pe, my_pe, thisBlockTokenIdx, thisBlockInput,
                  remoteTokenMemPool_sym, remoteTokenMemPoolFirstFreeIdx_sym,
                  remoteTokenMemPoolSlotStatus_sym);

    if (threadIdx.x == 0) {
      const int prev = __hip_atomic_fetch_add(
          sendTokens_global, 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);

      // Last block informs all PEs that sending is done.
      if (prev == (numTokens - 1)) {
        int dst = my_pe;
        for (int i = 0; i < npes - 1; ++i) {
          dst = (dst + 1) % npes;
          rocshmem_int_atomic_inc(doneSendingFlag_sym, dst);
        }
        HIP_DEVICE_LOG("PE %d done sending all tokens. \n", my_pe);
      }
    }
    __syncthreads();
  }

  __shared__ bool stopSignal_shared;
  __shared__ float* remoteSlotPtr_shared;
  __shared__ int receivedTokenIdx_shared;
  __shared__ int receivedPE_shared;

  // 2. Each block processes received msgs.
  while (true) {
    if (threadIdx.x == 0) {
      HIP_DEVICE_LOG("PE %d block %d started to receive msgs.\n", my_pe,
                     blockIdx.x);

      const int slot =
          __hip_atomic_fetch_add(reserverdSlotIdx_global, 1, __ATOMIC_RELAXED,
                                 __HIP_MEMORY_SCOPE_AGENT);

      int slotStatus = rocshmem_uint64_atomic_fetch(
          remoteTokenMemPoolSlotStatus_sym + slot, my_pe);

      HIP_DEVICE_LOG("PE %d block %d waiting for slot %d to be filled.\n",
                     my_pe, blockIdx.x, slot);

      bool shouldStop = false;

      while (slotStatus == 0 && !shouldStop) {
        slotStatus = rocshmem_uint64_atomic_fetch(
            remoteTokenMemPoolSlotStatus_sym + slot, my_pe);

        const bool allDoneSending =
            rocshmem_int_test(doneSendingFlag_sym, ROCSHMEM_CMP_EQ, npes - 1) ==
            1;

        const bool allTokensSaved =
            __hip_atomic_load(numOutputTokens_global, __ATOMIC_RELAXED,
                              __HIP_MEMORY_SCOPE_AGENT) == numTokens;

        shouldStop = allDoneSending && allTokensSaved;
      }

      if (!shouldStop) {
        float* remotePoolPtr =
            remoteTokenMemPool_sym + slot * (TOKEN_SIZE + TOKEN_HEADER_SIZE);

        float pe;
        float tokenIdx;
        rocshmem_float_get(&pe, remotePoolPtr, 1, my_pe);
        rocshmem_float_get(&tokenIdx, remotePoolPtr + 1, 1, my_pe);
        int peInt = (int)pe;
        int tokenIdxInt = (int)tokenIdx;

        remoteSlotPtr_shared = remotePoolPtr;
        receivedPE_shared = peInt;
        receivedTokenIdx_shared = tokenIdxInt;

        HIP_DEVICE_LOG(
            "PE %d block %d received msg in slot %d: from PE %d, tokenIdx %d\n",
            my_pe, blockIdx.x, slot, peInt, tokenIdxInt);
      }

      stopSignal_shared = shouldStop;
    }

    __syncthreads();

    if (stopSignal_shared) {
      if (threadIdx.x == 0) {
        HIP_DEVICE_LOG("PE %d block %d received stop signal.\n", my_pe,
                       blockIdx.x);
      }
      break;
    }

    float* remotePoolPtr = remoteSlotPtr_shared;
    int peInt = receivedPE_shared;
    int tokenIdxInt = receivedTokenIdx_shared;

    processMsg_block(peInt, tokenIdxInt, remotePoolPtr + TOKEN_HEADER_SIZE,
                     output, remoteTokenMemPool_sym,
                     remoteTokenMemPoolFirstFreeIdx_sym,
                     remoteTokenMemPoolSlotStatus_sym, numOutputTokens_global);

    __syncthreads();
  }

  __syncthreads();
}

TEST_F(RocshmemForkBasedTest, MoeLikeCommPattern) {
  ExecuteInSeparateProcesses(4, [](int _rank, int _size) {
    const int rank = rocshmem_my_pe();
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
    HIP_ERROR_ASSERT(
        hipMalloc(&routingTable_device, TOKENS_PER_GPU * sizeof(int)));
    HIP_ERROR_ASSERT(hipMemcpy(routingTable_device, routingTable.data(),
                               TOKENS_PER_GPU * sizeof(int),
                               hipMemcpyHostToDevice));

    float* input_device;
    HIP_ERROR_ASSERT(hipMalloc(&input_device, INPUT_SIZE * sizeof(float)));
    HIP_ERROR_ASSERT(hipMemcpy(input_device, input_host.data(),
                               INPUT_SIZE * sizeof(float),
                               hipMemcpyHostToDevice));

    int* sendTokens_global;
    HIP_ERROR_ASSERT(hipMalloc(&sendTokens_global, sizeof(int)));
    HIP_ERROR_ASSERT(hipMemset(sendTokens_global, 0, sizeof(int)));

    int* numOutputTokens_global;
    HIP_ERROR_ASSERT(hipMalloc(&numOutputTokens_global, sizeof(int)));
    HIP_ERROR_ASSERT(hipMemset(numOutputTokens_global, 0, sizeof(int)));

    float* remoteTokenMemPool_sym = (float*)rocshmem_malloc(
        ALLOC_SLOT_SIZE * (TOKEN_SIZE + TOKEN_HEADER_SIZE) * sizeof(float));

    int* remoteTokenMemPoolFirstFreeIdx_sym =
        (int*)rocshmem_malloc(sizeof(int));
    HIP_ERROR_ASSERT(
        hipMemset(remoteTokenMemPoolFirstFreeIdx_sym, 0, sizeof(int)));

    uint64_t* remoteTokenMemPoolSlotStatus_sym =
        (uint64_t*)rocshmem_malloc(ALLOC_SLOT_SIZE * sizeof(uint64_t));
    HIP_ERROR_ASSERT(hipMemset(remoteTokenMemPoolSlotStatus_sym, 0,
                               ALLOC_SLOT_SIZE * sizeof(uint64_t)));

    int* reserverdSlotIdx_global;
    HIP_ERROR_ASSERT(hipMalloc(&reserverdSlotIdx_global, sizeof(int)));
    HIP_ERROR_ASSERT(hipMemset(reserverdSlotIdx_global, 0, sizeof(int)));

    int* doneSendingFlag_sym = (int*)rocshmem_malloc(sizeof(int));
    HIP_ERROR_ASSERT(hipMemset(doneSendingFlag_sym, 0, sizeof(int)));

    float* output_device;
    HIP_ERROR_ASSERT(hipMalloc(&output_device, INPUT_SIZE * sizeof(float)));

    if (rank == 0) {
      int delay = 2000;
      std::cout << "Rank " << rank << " will delay kernel launch by " << delay
                << " ms\n";
      std::this_thread::sleep_for(std::chrono::milliseconds(delay));
      std::cout << "Rank " << rank << " launching kernel on device " << rank
                << "\n";
    }

    moeLikeCommKernel<<<BLOCKS, THREADS>>>(
        input_device, output_device, routingTable_device,
        remoteTokenMemPool_sym, remoteTokenMemPoolFirstFreeIdx_sym,
        remoteTokenMemPoolSlotStatus_sym, doneSendingFlag_sym,
        sendTokens_global, numOutputTokens_global, reserverdSlotIdx_global,
        TOKENS_PER_GPU, rank, worldSize);

    rocshmem_barrier_all();
    HIP_ERROR_ASSERT(hipDeviceSynchronize());

    std::vector<float> output_host(INPUT_SIZE, 0);
    HIP_ERROR_ASSERT(hipMemcpy(output_host.data(), output_device,
                               INPUT_SIZE * sizeof(float),
                               hipMemcpyDeviceToHost));

    HIP_ERROR_ASSERT(hipFree(output_device));
    rocshmem_free(remoteTokenMemPool_sym);
    HIP_ERROR_ASSERT(hipFree(input_device));
    HIP_ERROR_ASSERT(hipFree(routingTable_device));

    // PrintOutput(output_host, rank, TOKEN_SIZE);
    VerifyOutput(input_host, routingTable, output_host, rank);
  });
}
}  // namespace
