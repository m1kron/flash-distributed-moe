#pragma once

// Initializes task system gemm.
hipError_t initTaskSystemGemm(void** state);

// Deinitializes task system gemm.
hipError_t deinitTaskSystemGemm(void* state);

// Gemm using persistent kernel and task system
hipError_t taskSystemGemm(void* state, const float* A, const float* B, float* C,
                          int M, int N, int K);