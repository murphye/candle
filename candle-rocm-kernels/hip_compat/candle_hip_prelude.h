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

// SIMD-in-register integer ops used by the quantized kernels, mapped onto the
// AMD dot product instructions as in llama.cpp's HIP backend.
typedef int8_t candle_int8x4_t __attribute__((ext_vector_type(4)));

__device__ __forceinline__ int __dp4a(const int a, const int b, int c) {
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) \
    || defined(__gfx1150__) || defined(__gfx1151__) || defined(__gfx1152__) || defined(__gfx1153__) \
    || defined(__gfx1200__) || defined(__gfx1201__)
    return __builtin_amdgcn_sudot4(true, a, true, b, c, false);
#elif defined(__gfx906__) || defined(__gfx908__) || defined(__gfx90a__) || defined(__gfx942__) \
    || defined(__gfx950__) || defined(__gfx1030__) || defined(__gfx1031__) || defined(__gfx1032__) \
    || defined(__gfx1034__) || defined(__gfx1035__) || defined(__gfx1036__)
    return __builtin_amdgcn_sdot4(a, b, c, false);
#else
    const candle_int8x4_t va = __builtin_bit_cast(candle_int8x4_t, a);
    const candle_int8x4_t vb = __builtin_bit_cast(candle_int8x4_t, b);
    return c + va[0] * vb[0] + va[1] * vb[1] + va[2] * vb[2] + va[3] * vb[3];
#endif
}

// Per-byte saturating subtraction of packed signed 8-bit values.
__device__ __forceinline__ int __vsubss4(const int a, const int b) {
    const candle_int8x4_t va = __builtin_bit_cast(candle_int8x4_t, a);
    const candle_int8x4_t vb = __builtin_bit_cast(candle_int8x4_t, b);
    const candle_int8x4_t c = __builtin_elementwise_sub_sat(va, vb);
    return __builtin_bit_cast(int, c);
}

// Warp shuffles: the kernels pass 32-bit full masks, which HIP's *_sync
// builtins reject. Every reduction uses xor offsets below 32, so it stays inside
// 32-lane groups and is also correct on wave64 targets.
#define __shfl_xor_sync(mask, ...) __shfl_xor(__VA_ARGS__)
#define __shfl_sync(mask, ...) __shfl(__VA_ARGS__)
#define __shfl_down_sync(mask, ...) __shfl_down(__VA_ARGS__)
#define __shfl_up_sync(mask, ...) __shfl_up(__VA_ARGS__)

__device__ __forceinline__ __hip_bfloat162 __floats2bfloat162_rn(float a, float b) {
    return __halves2bfloat162(__float2bfloat16(a), __float2bfloat16(b));
}

// cudaFuncSetAttribute accepts any kernel pointer; HIP's takes const void*.
template <typename Kernel>
static inline hipError_t candle_func_set_attribute(Kernel kernel, hipFuncAttribute attr,
                                                   int value) {
    return hipFuncSetAttribute(reinterpret_cast<const void *>(kernel), attr, value);
}

// CUDA runtime API names used by host-side kernel launchers.
#define cudaStream_t hipStream_t
#define cudaError_t hipError_t
#define cudaSuccess hipSuccess
#define cudaErrorInvalidValue hipErrorInvalidValue
#define cudaGetLastError hipGetLastError
#define cudaPeekAtLastError hipPeekAtLastError
#define cudaGetErrorString hipGetErrorString
#define cudaGetDevice hipGetDevice
#define cudaDeviceGetAttribute hipDeviceGetAttribute
#define cudaDevAttrMaxSharedMemoryPerBlockOptin hipDeviceAttributeSharedMemPerBlockOptin
#define cudaDevAttrMultiProcessorCount hipDeviceAttributeMultiprocessorCount
#define cudaFuncAttributes hipFuncAttributes
#define cudaFuncGetAttributes hipFuncGetAttributes
#define cudaFuncSetAttribute candle_func_set_attribute
#define cudaFuncAttributeMaxDynamicSharedMemorySize hipFuncAttributeMaxDynamicSharedMemorySize
#define cudaMemsetAsync hipMemsetAsync
#define cudaMemcpyAsync hipMemcpyAsync

// The kernels gate their f16/bf16/fp8 instantiations on the CUDA compute
// capability. All of these types are available on the supported AMD targets.
// Defined last so that the HIP and libhipcxx headers above never see it.
#define __CUDA_ARCH__ 890
#define __CUDACC_VER_MAJOR__ 12
#define __CUDACC_VER_MINOR__ 8
