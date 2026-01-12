#pragma once

#if defined(__GFX8__) || defined(__GFX9__)
#define WARP_SIZE 64
#else
#define WARP_SIZE 32
#endif