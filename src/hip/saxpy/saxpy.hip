#include <hip/hip_runtime.h>

#include "saxpy.h"

/// \brief Calculates \p d_y[i]=a*d_x[i]+d_y[i] where \p i stands for the
/// thread's index in the grid.
__global__ void saxpy_kernel(const float a, const float* __restrict__ d_x,
                             float* __restrict__ d_y, const unsigned int size) {
  // Compute the current thread's index in the grid.
  const int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int grid_stride = gridDim.x * blockDim.x;

  for (int i = global_idx; i < size; i += grid_stride) {
    d_y[i] = a * d_x[i] + d_y[i];
  }
}

void saxpy(const float a, const float* d_x, float* d_y,
           const unsigned int size) {
  // Number of threads per kernel block.
  constexpr unsigned int block_size = 512;
  constexpr unsigned int items_per_thread = 4;
  constexpr unsigned int items_per_block = block_size * items_per_thread;

  // Number of blocks per kernel grid. The expression below calculates
  // ceil(size/block_size).
  const unsigned int grid_size = (size + items_per_block - 1) / items_per_block;

  // Launch the kernel on the default stream.
  saxpy_kernel<<<dim3(grid_size), dim3(block_size), 0, hipStreamDefault>>>(
      a, d_x, d_y, size);
}