// Matrix-vector products for decode-shaped matmuls: `y[m, n] = x[m, k] . w[n, k]^T`
// with a handful of rows `m`. One 32-lane warp computes one output column for every
// row, streaming its weight row once with 16-byte loads. The BLAS libraries pick
// tiles that are far too large for these shapes and cost several microseconds of
// host time per call.
#include "cuda_bf16.h"
#include "cuda_fp16.h"
#include <stdint.h>

#define GEMV_MAX_ROWS 8
#define GEMV_LANES 32

template <typename T> __device__ __forceinline__ float gemv_to_float(T x) { return static_cast<float>(x); }
template <> __device__ __forceinline__ float gemv_to_float<__half>(__half x) { return __half2float(x); }
template <> __device__ __forceinline__ float gemv_to_float<__nv_bfloat16>(__nv_bfloat16 x) {
  return __bfloat162float(x);
}

template <typename T> __device__ __forceinline__ T gemv_from_float(float x) { return static_cast<T>(x); }
template <> __device__ __forceinline__ __half gemv_from_float<__half>(float x) { return __float2half(x); }
template <> __device__ __forceinline__ __nv_bfloat16 gemv_from_float<__nv_bfloat16>(float x) {
  return __float2bfloat16(x);
}

// `VEC` elements make one 16-byte load; the host only picks this kernel when `k` is a
// multiple of `VEC` and every row starts 16-byte aligned.
template <typename T>
__device__ __forceinline__ void gemv_vec_kernel(const T *__restrict__ w, const T *__restrict__ x, T *__restrict__ y,
                                const int n, const int k, const int m) {
  constexpr int VEC = 16 / sizeof(T);
  const int col = blockIdx.x * blockDim.y + threadIdx.y;
  if (col >= n) {
    return;
  }
  const int lane = threadIdx.x;
  const uint4 *wrow = reinterpret_cast<const uint4 *>(w + (int64_t)col * k);
  const int kv = k / VEC;

  float acc[GEMV_MAX_ROWS];
#pragma unroll
  for (int r = 0; r < GEMV_MAX_ROWS; ++r) {
    acc[r] = 0.0f;
  }
  for (int i = lane; i < kv; i += GEMV_LANES) {
    const uint4 wv = wrow[i];
    const T *wp = reinterpret_cast<const T *>(&wv);
    float wf[VEC];
#pragma unroll
    for (int j = 0; j < VEC; ++j) {
      wf[j] = gemv_to_float(wp[j]);
    }
#pragma unroll
    for (int r = 0; r < GEMV_MAX_ROWS; ++r) {
      if (r < m) {
        const uint4 xv = reinterpret_cast<const uint4 *>(x + (int64_t)r * k)[i];
        const T *xp = reinterpret_cast<const T *>(&xv);
#pragma unroll
        for (int j = 0; j < VEC; ++j) {
          acc[r] += wf[j] * gemv_to_float(xp[j]);
        }
      }
    }
  }
#pragma unroll
  for (int r = 0; r < GEMV_MAX_ROWS; ++r) {
    if (r < m) {
      float v = acc[r];
#pragma unroll
      for (int offset = GEMV_LANES / 2; offset > 0; offset >>= 1) {
        v += __shfl_xor_sync(0xffffffff, v, offset, GEMV_LANES);
      }
      if (lane == 0) {
        y[(int64_t)r * n + col] = gemv_from_float<T>(v);
      }
    }
  }
}

#define GEMV_OP(T, SUFFIX)                                                                       \
  extern "C" __global__ void gemv_##SUFFIX(const T *w, const T *x, T *y, const int n, const int k, \
                                           const int m) {                                        \
    gemv_vec_kernel<T>(w, x, y, n, k, m);                                                        \
  }

GEMV_OP(float, f32)
GEMV_OP(__half, f16)
GEMV_OP(__nv_bfloat16, bf16)
