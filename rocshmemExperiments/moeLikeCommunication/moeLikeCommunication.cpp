#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#include <iostream>
#include <rocshmem/rocshmem.hpp>

const int TOKENS_PER_GPU = 4;
const int TOKEN_SIZE = 2048;
const int THREADS = 1024;
const int BLOCKS = 4;
const int ALLOC_SLOT_SIZE = 16;

void PrintOutput(const std::vector<float>& output_host, int rank) {
  std::string outputString;
  outputString += "\nRank " + std::to_string(rank) + ":\n";
  for (int tokenIdx = 0; tokenIdx < TOKENS_PER_GPU; tokenIdx++) {
    outputString += "Token " + std::to_string(tokenIdx) + ":\n";
    for (int i = 0; i < 5; i++) {
      int globalIdx = tokenIdx * TOKEN_SIZE + i;
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
                                  float* remoteTokenMemPool,
                                  int* remoteTokenMemPoolIdx, int numTokens,
                                  int my_pe, int npes) {
  // 1. Send input to destination PEs based on routing table.
  for (int thisBlockTokenIdx = blockIdx.x; thisBlockTokenIdx < numTokens;
       thisBlockTokenIdx += gridDim.x) {
    const int dst_pe = routingTable[thisBlockTokenIdx];

    if (threadIdx.x == 0) {
      const int idx =
          rocshmem_int_atomic_fetch_inc(remoteTokenMemPoolIdx, dst_pe);
      const float* thisBlockInput = input + thisBlockTokenIdx * TOKEN_SIZE;
      printf("PE %d sending token %d to PE %d into slot %d\n", my_pe,
             thisBlockTokenIdx, dst_pe, idx);

      rocshmem_putmem(remoteTokenMemPool + idx * TOKEN_SIZE, thisBlockInput,
                         TOKEN_SIZE * sizeof(float), dst_pe);
    }

    // float dstForToken = (float)routingTable[thisBlockTokenIdx];

    // float* thisBlockOutput = output + thisBlockTokenIdx * TOKEN_SIZE;

    // for (int i = threadIdx.x; i < TOKEN_SIZE; i += blockDim.x) {
    //   thisBlockOutput[i] = dstForToken + thisBlockInput[i];
    // }
  }
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

  float* remoteTokenMemPool =
      (float*)rocshmem_malloc(ALLOC_SLOT_SIZE * TOKEN_SIZE * sizeof(float));

  int* remoteTokenMemPoolIdx = (int*)rocshmem_malloc(sizeof(int));
  CHECK_HIP(hipMemset(remoteTokenMemPoolIdx, 0, sizeof(int)));

  float* output_device;
  CHECK_HIP(hipMalloc(&output_device, INPUT_SIZE * sizeof(float)));

  moeLikeCommKernel<<<BLOCKS, THREADS>>>(
      input_device, output_device, routingTable_device, remoteTokenMemPool,
      remoteTokenMemPoolIdx, TOKENS_PER_GPU, rank, worldSize);

  rocshmem_barrier_all();
  CHECK_HIP(hipDeviceSynchronize());

  std::vector<float> output_host(INPUT_SIZE, 0);
  CHECK_HIP(hipMemcpy(output_host.data(), remoteTokenMemPool,
                      INPUT_SIZE * sizeof(float), hipMemcpyDeviceToHost));

  CHECK_HIP(hipFree(output_device));
  rocshmem_free(remoteTokenMemPool);
  CHECK_HIP(hipFree(input_device));
  CHECK_HIP(hipFree(routingTable_device));

  rocshmem_finalize();

  PrintOutput(output_host, rank);
  // VerifyOutput(input_host, routingTable, output_host, rank);
  return 0;
}