#pragma once

#include <hip/hip_fp16.h>

#include "./bits/gpu_infinities.h"
#include "./bits/gpu_atomics.h"

#ifdef __HIPCC_RTC__
typedef __hip_uint8_t uint8_t;
typedef __hip_int8_t int8_t;
typedef __hip_uint16_t uint16_t;
typedef __hip_int16_t int16_t;
#endif
