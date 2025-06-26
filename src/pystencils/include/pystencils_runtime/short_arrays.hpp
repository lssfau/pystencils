#pragma once

#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)

#define QUALIFIERS __device__ __forceinline__

#else

#include <cstdint>

using std::size_t;

#define QUALIFIERS

#endif

namespace pystencils::runtime {

    struct Fp32x4 {
        float v[4];

        QUALIFIERS float & operator[] (size_t i) {
            return v[i];
        }

        QUALIFIERS const float & operator[] (size_t i) const {
            return v[i];
        }
    };

    struct Fp64x2 {
        double v[2];

        QUALIFIERS double & operator[] (size_t i) {
            return v[i];
        }

        QUALIFIERS const double & operator[] (size_t i) const {
            return v[i];
        }
    };

}  // namespace pystencils::runtime

#undef QUALIFIERS
