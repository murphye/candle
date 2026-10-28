// Force-included before every kernel source (hipcc -include). Maps the CUDA
// toolkit names used by candle's kernels onto their HIP equivalents.
#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp8.h>
#include <limits>

typedef __hip_bfloat16 __nv_bfloat16;
typedef __hip_bfloat162 __nv_bfloat162;
// candle's F8E4M3 is the OCP e4m3fn format, which is HIP's __hip_fp8_e4m3
// (not the *_fnuz variant native to gfx942).
typedef __hip_fp8_e4m3 __nv_fp8_e4m3;
#define __NV_E4M3 __HIP_E4M3
#define __nv_cvt_fp8_to_halfraw __hip_cvt_fp8_to_halfraw

// HIP rounds to nearest even by default.
#define __float2bfloat16_rn __float2bfloat16

// NaN-propagating min/max: HIP only provides the __half overloads.
__device__ __forceinline__ __hip_bfloat16 __hmax_nan(__hip_bfloat16 a, __hip_bfloat16 b) {
    float fa = __bfloat162float(a), fb = __bfloat162float(b);
    return isnan(fa) ? a : (isnan(fb) ? b : (fa > fb ? a : b));
}
__device__ __forceinline__ __hip_bfloat16 __hmin_nan(__hip_bfloat16 a, __hip_bfloat16 b) {
    float fa = __bfloat162float(a), fb = __bfloat162float(b);
    return isnan(fa) ? a : (isnan(fb) ? b : (fa < fb ? a : b));
}

// Warp shuffles: the kernels pass 32-bit full masks, which HIP's *_sync
// builtins reject. Every reduction uses xor offsets below 32, so it stays inside
// 32-lane groups and is also correct on wave64 targets.
#define __shfl_xor_sync(mask, ...) __shfl_xor(__VA_ARGS__)
#define __shfl_sync(mask, ...) __shfl(__VA_ARGS__)
#define __shfl_down_sync(mask, ...) __shfl_down(__VA_ARGS__)
#define __shfl_up_sync(mask, ...) __shfl_up(__VA_ARGS__)

// The kernels gate their f16/bf16/fp8 instantiations on the CUDA compute
// capability. All of these types are available on the supported AMD targets.
// Defined last so that the HIP and libhipcxx headers above never see it.
#define __CUDA_ARCH__ 890
#define __CUDACC_VER_MAJOR__ 12
#define __CUDACC_VER_MINOR__ 8
