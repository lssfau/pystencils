#pragma once

#include "./short_arrays.hpp"
#include "./bits/philox_rand.h"

#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)

#define QUALIFIERS __device__ __forceinline__

#else

#define QUALIFIERS inline

#endif

namespace pystencils::runtime::random
{

    QUALIFIERS Fp32x4 philox_fp32x4(uint32_t ctr0, uint32_t ctr1, uint32_t ctr2, uint32_t ctr3, uint32_t key0, uint32_t key1) {
        Fp32x4 result;
        detail::philox_float4(ctr0, ctr1, ctr2, ctr3, key0, key1, result[0], result[1], result[2], result[3]);
        return result;
    }

    QUALIFIERS Fp64x2 philox_fp64x2(uint32_t ctr0, uint32_t ctr1, uint32_t ctr2, uint32_t ctr3, uint32_t key0, uint32_t key1) {
        Fp64x2 result;
        detail::philox_double2(ctr0, ctr1, ctr2, ctr3, key0, key1, result[0], result[1]);
        return result;
    }

} // pystencils::runtime::random

#undef QUALIFIERS
