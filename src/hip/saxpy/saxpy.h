#pragma once

extern "C" void saxpy(const float a, const float* d_x, float* d_y,
           const unsigned int size);