#pragma once

// Simple, statnadard gemm with hw scheduling.
void staticGemm(const float* A, const float* B, float* C, int M, int N, int K);