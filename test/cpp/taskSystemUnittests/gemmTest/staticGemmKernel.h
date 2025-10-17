#pragma once

// Simple, statnadard gemm with hw scheduling.
hipError_t staticGemm(const float* A, const float* B, float* C, int M, int N,
                      int K);