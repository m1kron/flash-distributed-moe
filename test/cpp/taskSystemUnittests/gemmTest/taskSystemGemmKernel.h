#pragma once

// Gemm using persistent kernel and task system
hipError_t taskSystemGemm(const float* A, const float* B, float* C, int M, int N, int K);