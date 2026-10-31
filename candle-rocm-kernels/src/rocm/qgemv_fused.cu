// Fused matrix-vector products on GGUF weights for decode-sized batches (up to 8 rows),
// dequantizing weights on the fly against bf16 activations. They replace short chains of
// tiny launches (quantize the activations, a small matmul, an activation, another matmul,
// an elementwise mix), where launch latency, not bandwidth, sets the cost.
#include "quantized.cu"
#include "dequant8.cuh"

#define QGEMV_MAX_ROWS 8
#define QGEMV_THREADS 256

// Weight formats, by the code the host passes.
#define QG_Q4_0 0
#define QG_Q5_0 1
#define QG_Q8_0 2
#define QG_Q4_K 3
#define QG_Q6_K 4

static __device__ __forceinline__ int qg_block_weights(int t) { return t >= QG_Q4_K ? 256 : 32; }

static __device__ __forceinline__ int64_t qg_block_bytes(int t) {
  switch (t) {
  case QG_Q4_0: return sizeof(block_q4_0);
  case QG_Q5_0: return sizeof(block_q5_0);
  case QG_Q8_0: return sizeof(block_q8_0);
  case QG_Q4_K: return sizeof(block_q4_K);
  default: return sizeof(block_q6_K);
  }
}

// Weights `8 * c .. 8 * c + 7` of row `row` (`blocks_per_row` quant blocks each).
static __device__ __forceinline__ void qg_dequant8(int t, const void *w, int64_t blocks_per_row,
                                                   int64_t row, int c, float *out) {
  const int qk = qg_block_weights(t);
  const int64_t ib = row * blocks_per_row + (int64_t)(c * 8) / qk;
  const int part = (c * 8 % qk) / 8;
  switch (t) {
  case QG_Q4_0: moe_dequant8_q4_0(w, ib, part, out); break;
  case QG_Q5_0: moe_dequant8_q5_0(w, ib, part, out); break;
  case QG_Q8_0: moe_dequant8_q8_0(w, ib, part, out); break;
  case QG_Q4_K: moe_dequant8_q4_K(w, ib, part, out); break;
  default: moe_dequant8_q6_K(w, ib, part, out); break;
  }
}

static __device__ __forceinline__ void qg_load8(const __nv_bfloat16 *x, float *out) {
  const uint4 v = *reinterpret_cast<const uint4 *>(x);
  const __nv_bfloat16 *h = reinterpret_cast<const __nv_bfloat16 *>(&v);
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    out[i] = __bfloat162float(h[i]);
  }
}

// y1[b, r] = act(x[b] . w1[r]) for r < r1 and y2[b, r] = x[b] . w2[r] for r < r2: two
// matrices over the same activations in one launch. A workgroup owns QG_PAIR_ROWS output rows;
// each lane loads one activation chunk for all rows of the batch once and reuses it for every
// row of the tile.
// act: 0 identity, 1 silu(v * scale1). k is a multiple of both formats' block sizes.
template <int QG_PAIR_ROWS>
__device__ __forceinline__ void qgemv_pair_impl(const __nv_bfloat16 *__restrict__ x, const int m, const int k,
                    const void *__restrict__ w1, const int t1, const int r1, const int act1,
                    const float scale1, __nv_bfloat16 *__restrict__ y1,
                    const void *__restrict__ w2, const int t2, const int r2,
                    __nv_bfloat16 *__restrict__ y2) {
  const int row0 = blockIdx.x * QG_PAIR_ROWS;
  float acc[QG_PAIR_ROWS][QGEMV_MAX_ROWS] = {};
  for (int c = threadIdx.x; c < k / 8; c += QGEMV_THREADS) {
    float xv[QGEMV_MAX_ROWS][8];
#pragma unroll
    for (int b = 0; b < QGEMV_MAX_ROWS; ++b) {
      if (b < m) {
        qg_load8(x + (int64_t)b * k + c * 8, xv[b]);
      }
    }
#pragma unroll
    for (int r = 0; r < QG_PAIR_ROWS; ++r) {
      const int row = row0 + r;
      if (row >= r1 + r2) {
        break;
      }
      const bool first = row < r1;
      const int t = first ? t1 : t2;
      float wv[8];
      qg_dequant8(t, first ? w1 : w2, k / qg_block_weights(t), first ? row : row - r1, c, wv);
#pragma unroll
      for (int b = 0; b < QGEMV_MAX_ROWS; ++b) {
        if (b < m) {
          float d = 0.f;
#pragma unroll
          for (int i = 0; i < 8; ++i) {
            d += wv[i] * xv[b][i];
          }
          acc[r][b] += d;
        }
      }
    }
  }
  __shared__ float partial[QG_PAIR_ROWS * QGEMV_MAX_ROWS][QGEMV_THREADS / 32];
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
#pragma unroll
  for (int r = 0; r < QG_PAIR_ROWS; ++r) {
#pragma unroll
    for (int b = 0; b < QGEMV_MAX_ROWS; ++b) {
      if (b < m) {
        float v = acc[r][b];
        for (int off = 16; off > 0; off >>= 1) {
          v += __shfl_xor_sync(0xffffffff, v, off, 32);
        }
        if (lane == 0) {
          partial[r * QGEMV_MAX_ROWS + b][warp] = v;
        }
      }
    }
  }
  __syncthreads();
  if (threadIdx.x < QG_PAIR_ROWS * m) {
    const int r = threadIdx.x / m, b = threadIdx.x % m;
    const int row = row0 + r;
    if (row < r1 + r2) {
      float v = 0.f;
#pragma unroll
      for (int i = 0; i < QGEMV_THREADS / 32; ++i) {
        v += partial[r * QGEMV_MAX_ROWS + b][i];
      }
      if (row < r1) {
        if (act1 == 1) {
          v *= scale1;
          v = v / (1.f + __expf(-v));
        }
        y1[(int64_t)b * r1 + row] = __float2bfloat16(v);
      } else {
        y2[(int64_t)b * r2 + row - r1] = __float2bfloat16(v);
      }
    }
  }
}

#define QG_PAIR(ROWS)                                                                              \
  extern "C" __global__ void __launch_bounds__(QGEMV_THREADS) qgemv_pair_bf16_r##ROWS(            \
      const __nv_bfloat16 *__restrict__ x, const int m, const int k, const void *__restrict__ w1,   \
      const int t1, const int r1, const int act1, const float scale1, __nv_bfloat16 *__restrict__ y1, \
      const void *__restrict__ w2, const int t2, const int r2, __nv_bfloat16 *__restrict__ y2) {     \
    qgemv_pair_impl<ROWS>(x, m, k, w1, t1, r1, act1, scale1, y1, w2, t2, r2, y2);                 \
  }
// One row per workgroup measured fastest on gfx1201 for these shapes (more workgroups beat
// fewer activation loads); the tile stays a parameter for other shapes.
QG_PAIR(1)

// out[b, h] = mean_s xn[b, s * hidden + h] * sigmoid(lo[b] . w[s * hidden + h]) for s < groups:
// the gate projection and the gated mean over residual streams in one launch. One wave per
// output column h: each group of 32 / groups lanes takes one stream's row, so a lane's chunks
// all belong to one row; lo ([m, k], at most 8 rows) is read straight from global memory.
#define QG_GATE_WARPS 8
#define QG_MAX_GROUPS 4
extern "C" __global__ void __launch_bounds__(QG_GATE_WARPS * 32)
    qgemv_gate_mean_bf16(const __nv_bfloat16 *__restrict__ lo, const int m, const int k,
                         const void *__restrict__ w, const int t, const __nv_bfloat16 *__restrict__ xn,
                         const int groups, const int hidden, __nv_bfloat16 *__restrict__ out) {
  const int lane = threadIdx.x % 32;
  const int h = blockIdx.x * QG_GATE_WARPS + threadIdx.x / 32;
  if (h >= hidden) {
    return;
  }
  const int per = 32 / groups;          // lanes per stream
  const int s = lane / per;             // this lane's stream
  const int sub = lane % per;
  const int64_t blocks_per_row = k / qg_block_weights(t);
  const int64_t row = (int64_t)s * hidden + h;
  float acc[QGEMV_MAX_ROWS] = {};
  if (s < groups) {
    for (int c = sub; c < k / 8; c += per) {
      float wv[8];
      qg_dequant8(t, w, blocks_per_row, row, c, wv);
#pragma unroll
      for (int b = 0; b < QGEMV_MAX_ROWS; ++b) {
        if (b < m) {
          float xv[8];
          qg_load8(lo + (int64_t)b * k + c * 8, xv);
          float d = 0.f;
#pragma unroll
          for (int i = 0; i < 8; ++i) {
            d += wv[i] * xv[i];
          }
          acc[b] += d;
        }
      }
    }
  }
  // Sum each stream's partial dots over its lanes, gate, then sum the streams.
  float mix[QGEMV_MAX_ROWS] = {};
#pragma unroll
  for (int b = 0; b < QGEMV_MAX_ROWS; ++b) {
    if (b < m) {
      float v = acc[b];
      for (int off = per / 2; off > 0; off >>= 1) {
        v += __shfl_xor_sync(0xffffffff, v, off, 32);
      }
      float g = 0.f;
      if (sub == 0 && s < groups) {
        g = __bfloat162float(xn[(int64_t)b * groups * hidden + row]) / (1.f + __expf(-v));
      }
      for (int off = 16; off > 0; off >>= 1) {
        g += __shfl_xor_sync(0xffffffff, g, off, 32);
      }
      mix[b] = g;
    }
  }
#pragma unroll
  for (int b = 0; b < QGEMV_MAX_ROWS; ++b) {
    if (b < m && lane == b) {
      out[(int64_t)b * hidden + h] = __float2bfloat16(mix[b] / groups);
    }
  }
}

// ---- Shape-specialized variants ----------------------------------------------------------
// With the weight format and the chunk count per lane known at compile time, every chunk's
// loads go out before the first dot product: the generic kernels above walk their chunks one
// dependent round trip at a time, which at decode shapes costs more than the bytes do.

template <int T>
static __device__ __forceinline__ void qg_dequant8_t(const void *w, int64_t blocks_per_row, int64_t row,
                                                     int c, float *out) {
  constexpr int qk = T >= QG_Q4_K ? 256 : 32;
  const int64_t ib = row * blocks_per_row + (int64_t)(c * 8) / qk;
  const int part = (c * 8 % qk) / 8;
  if constexpr (T == QG_Q4_0) {
    moe_dequant8_q4_0(w, ib, part, out);
  } else if constexpr (T == QG_Q5_0) {
    moe_dequant8_q5_0(w, ib, part, out);
  } else if constexpr (T == QG_Q8_0) {
    moe_dequant8_q8_0(w, ib, part, out);
  } else if constexpr (T == QG_Q4_K) {
    moe_dequant8_q4_K(w, ib, part, out);
  } else {
    moe_dequant8_q6_K(w, ib, part, out);
  }
}

// One output row per workgroup over k = ITERS * QGEMV_THREADS * 8 activations.
template <int T1, int T2, int ITERS>
__device__ __forceinline__ void qgemv_pair_fixed(const __nv_bfloat16 *__restrict__ x, const int m,
                                                 const void *__restrict__ w1, const int r1, const int act1,
                                                 const float scale1, __nv_bfloat16 *__restrict__ y1,
                                                 const void *__restrict__ w2, const int r2,
                                                 __nv_bfloat16 *__restrict__ y2) {
  constexpr int k = ITERS * QGEMV_THREADS * 8;
  const int row = blockIdx.x;
  if (row >= r1 + r2) {
    return;
  }
  const bool first = row < r1;
  float wv[ITERS][8];
  if (first) {
#pragma unroll
    for (int i = 0; i < ITERS; ++i) {
      qg_dequant8_t<T1>(w1, k / (T1 >= QG_Q4_K ? 256 : 32), row, threadIdx.x + i * QGEMV_THREADS, wv[i]);
    }
  } else {
#pragma unroll
    for (int i = 0; i < ITERS; ++i) {
      qg_dequant8_t<T2>(w2, k / (T2 >= QG_Q4_K ? 256 : 32), row - r1, threadIdx.x + i * QGEMV_THREADS,
                        wv[i]);
    }
  }
  float acc[QGEMV_MAX_ROWS] = {};
#pragma unroll
  for (int b = 0; b < QGEMV_MAX_ROWS; ++b) {
    if (b < m) {
      float xv[ITERS][8];
#pragma unroll
      for (int i = 0; i < ITERS; ++i) {
        qg_load8(x + (int64_t)b * k + (threadIdx.x + i * QGEMV_THREADS) * 8, xv[i]);
      }
      // Per-chunk sums, then the running total: the generic kernel's order, so results match.
#pragma unroll
      for (int i = 0; i < ITERS; ++i) {
        float d = 0.f;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          d += wv[i][j] * xv[i][j];
        }
        acc[b] += d;
      }
    }
  }
  __shared__ float partial[QGEMV_MAX_ROWS][QGEMV_THREADS / 32];
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
#pragma unroll
  for (int b = 0; b < QGEMV_MAX_ROWS; ++b) {
    if (b < m) {
      float v = acc[b];
      for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_xor_sync(0xffffffff, v, off, 32);
      }
      if (lane == 0) {
        partial[b][warp] = v;
      }
    }
  }
  __syncthreads();
  if (threadIdx.x < m) {
    const int b = threadIdx.x;
    float v = 0.f;
#pragma unroll
    for (int i = 0; i < QGEMV_THREADS / 32; ++i) {
      v += partial[b][i];
    }
    if (first) {
      if (act1 == 1) {
        v *= scale1;
        v = v / (1.f + __expf(-v));
      }
      y1[(int64_t)b * r1 + row] = __float2bfloat16(v);
    } else {
      y2[(int64_t)b * r2 + row - r1] = __float2bfloat16(v);
    }
  }
}

// Same signature as qgemv_pair_bf16_r1; `k`, `t1` and `t2` are fixed by the name.
#define QG_PAIR_FIXED(NAME, T1, T2, ITERS)                                                         \
  extern "C" __global__ void __launch_bounds__(QGEMV_THREADS) NAME(                                \
      const __nv_bfloat16 *__restrict__ x, const int m, const int k, const void *__restrict__ w1,   \
      const int t1, const int r1, const int act1, const float scale1, __nv_bfloat16 *__restrict__ y1, \
      const void *__restrict__ w2, const int t2, const int r2, __nv_bfloat16 *__restrict__ y2) {     \
    qgemv_pair_fixed<T1, T2, ITERS>(x, m, w1, r1, act1, scale1, y1, w2, r2, y2);                   \
  }
QG_PAIR_FIXED(qgemv_pair_bf16_k10240_q4k_q4k, QG_Q4_K, QG_Q4_K, 5)
QG_PAIR_FIXED(qgemv_pair_bf16_k10240_q6k_q4k, QG_Q6_K, QG_Q4_K, 5)

// qgemv_gate_mean_bf16 for GROUPS streams over k = ITERS * (32 / GROUPS) * 8, WARPS columns
// per workgroup. With Q8 the workgroup's 32 columns form one q8_1 block per batch row, written
// to `q8` in the layout of mmvq_gguf_quantize_q8_1 (`nblocks = hidden / 32` blocks per row),
// so the matmuls that consume the mixed activations skip their quantize launch.
template <int T, int GROUPS, int ITERS, int WARPS, bool Q8, int MAX_ROWS = QGEMV_MAX_ROWS>
__device__ __forceinline__ void qgemv_gate_mean_fixed(const __nv_bfloat16 *__restrict__ lo, const int m,
                                                      const void *__restrict__ w,
                                                      const __nv_bfloat16 *__restrict__ xn, const int hidden,
                                                      __nv_bfloat16 *__restrict__ out,
                                                      block_q8_1 *__restrict__ q8) {
  constexpr int per = 32 / GROUPS;
  constexpr int k = ITERS * per * 8;
  constexpr int qk = T >= QG_Q4_K ? 256 : 32;
  const int lane = threadIdx.x % 32;
  const int warp = threadIdx.x / 32;
  const int h = blockIdx.x * WARPS + warp;
  // Every wave, and each of its GROUPS lane groups, reads all of lo; past 8 rows those reads
  // from cache cost more than the weights, so the wide variants stage lo in LDS once.
  constexpr bool STAGE = MAX_ROWS > QGEMV_MAX_ROWS;
  __shared__ __attribute__((aligned(16))) __nv_bfloat16 lo_s[STAGE ? MAX_ROWS * k : 8];
  if constexpr (STAGE) {
    for (int i = threadIdx.x * 8; i < m * k; i += blockDim.x * 8) {
      *reinterpret_cast<uint4 *>(lo_s + i) = *reinterpret_cast<const uint4 *>(lo + i);
    }
    __syncthreads();
  }
  if constexpr (!Q8) {
    if (h >= hidden) {
      return;
    }
  }
  const int s = lane / per;
  const int sub = lane % per;
  const int64_t row = (int64_t)s * hidden + h;
  float wv[ITERS][8];
#pragma unroll
  for (int i = 0; i < ITERS; ++i) {
    qg_dequant8_t<T>(w, k / qk, row, sub + i * per, wv[i]);
  }
  float acc[MAX_ROWS] = {};
#pragma unroll
  for (int b = 0; b < MAX_ROWS; ++b) {
    if (b < m) {
      float xv[ITERS][8];
#pragma unroll
      for (int i = 0; i < ITERS; ++i) {
        if constexpr (STAGE) {
          qg_load8(lo_s + b * k + (sub + i * per) * 8, xv[i]);
        } else {
          qg_load8(lo + (int64_t)b * k + (sub + i * per) * 8, xv[i]);
        }
      }
#pragma unroll
      for (int i = 0; i < ITERS; ++i) {
        float d = 0.f;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          d += wv[i][j] * xv[i][j];
        }
        acc[b] += d;
      }
    }
  }
  float mix[MAX_ROWS] = {};
#pragma unroll
  for (int b = 0; b < MAX_ROWS; ++b) {
    if (b < m) {
      float v = acc[b];
      for (int off = per / 2; off > 0; off >>= 1) {
        v += __shfl_xor_sync(0xffffffff, v, off, 32);
      }
      float g = 0.f;
      if (sub == 0) {
        g = __bfloat162float(xn[(int64_t)b * GROUPS * hidden + row]) / (1.f + __expf(-v));
      }
      for (int off = 16; off > 0; off >>= 1) {
        g += __shfl_xor_sync(0xffffffff, g, off, 32);
      }
      mix[b] = g;
    }
  }
#pragma unroll
  for (int b = 0; b < MAX_ROWS; ++b) {
    if (b < m && lane == b) {
      out[(int64_t)b * hidden + h] = __float2bfloat16(mix[b] / GROUPS);
    }
  }
  if constexpr (Q8) {
    // Quantize this workgroup's 32 columns (one block per batch row) from the stored bf16
    // values, so the result equals a quantize launch over `out`.
    static_assert(WARPS == 32, "the q8 variant quantizes one 32-column block per workgroup");
    __shared__ float cols[MAX_ROWS][32];
#pragma unroll
    for (int b = 0; b < MAX_ROWS; ++b) {
      if (b < m && lane == 0) {
        cols[b][warp] = __bfloat162float(__float2bfloat16(mix[b] / GROUPS));
      }
    }
    __syncthreads();
    if (warp < m) {
      const int b = warp;
      const float xi = cols[b][lane];
      float amax = fabsf(xi);
      float sum = xi;
      for (int off = 16; off > 0; off >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, off, 32));
        sum += __shfl_xor_sync(0xffffffff, sum, off, 32);
      }
      const float d = amax / 127.0f;
      block_q8_1 *blk = q8 + (size_t)b * (hidden / 32) + blockIdx.x;
      blk->qs[lane] = amax == 0.0f ? 0 : (int8_t)roundf(xi / d);
      if (lane == 0) {
        reinterpret_cast<half &>(blk->ds.x) = (half)d;
        reinterpret_cast<half &>(blk->ds.y) = (half)sum;
      }
    }
  }
}

// Same signature as qgemv_gate_mean_bf16; `k`, `t` and `groups` are fixed by the name. The _r16
// variants take up to 16 batch rows (a speculative verify step over three or four sequences).
#define QG_GATE_MEAN_FIXED(NAME, T, GROUPS, ITERS)                                                 \
  extern "C" __global__ void __launch_bounds__(QG_GATE_WARPS * 32) NAME(                           \
      const __nv_bfloat16 *__restrict__ lo, const int m, const int k, const void *__restrict__ w,   \
      const int t, const __nv_bfloat16 *__restrict__ xn, const int groups, const int hidden,       \
      __nv_bfloat16 *__restrict__ out) {                                                           \
    qgemv_gate_mean_fixed<T, GROUPS, ITERS, QG_GATE_WARPS, false>(lo, m, w, xn, hidden, out,       \
                                                                   nullptr);                       \
  }                                                                                                \
  extern "C" __global__ void __launch_bounds__(1024) NAME##_q8(                                    \
      const __nv_bfloat16 *__restrict__ lo, const int m, const int k, const void *__restrict__ w,   \
      const int t, const __nv_bfloat16 *__restrict__ xn, const int groups, const int hidden,       \
      __nv_bfloat16 *__restrict__ out, void *__restrict__ q8) {                                    \
    qgemv_gate_mean_fixed<T, GROUPS, ITERS, 32, true>(lo, m, w, xn, hidden, out,                   \
                                                       (block_q8_1 *)q8);                          \
  }                                                                                                \
  extern "C" __global__ void __launch_bounds__(QG_GATE_WARPS * 32) NAME##_r16(                     \
      const __nv_bfloat16 *__restrict__ lo, const int m, const int k, const void *__restrict__ w,   \
      const int t, const __nv_bfloat16 *__restrict__ xn, const int groups, const int hidden,       \
      __nv_bfloat16 *__restrict__ out) {                                                           \
    qgemv_gate_mean_fixed<T, GROUPS, ITERS, QG_GATE_WARPS, false, 16>(lo, m, w, xn, hidden, out,   \
                                                                       nullptr);                   \
  }                                                                                                \
  extern "C" __global__ void __launch_bounds__(1024) NAME##_q8_r16(                                \
      const __nv_bfloat16 *__restrict__ lo, const int m, const int k, const void *__restrict__ w,   \
      const int t, const __nv_bfloat16 *__restrict__ xn, const int groups, const int hidden,       \
      __nv_bfloat16 *__restrict__ out, void *__restrict__ q8) {                                    \
    qgemv_gate_mean_fixed<T, GROUPS, ITERS, 32, true, 16>(lo, m, w, xn, hidden, out,               \
                                                           (block_q8_1 *)q8);                      \
  }
QG_GATE_MEAN_FIXED(qgemv_gate_mean_bf16_k320_g4_q5_0, QG_Q5_0, 4, 5)

// qgemv_gate_mean_bf16_k320_g4_q5_0 for 9-16 batch rows on gfx12 WMMA, where the per-row dot
// products of the scalar kernels dominate: the gate projection is a [16, 320] x [320, 4 * hidden]
// product. A workgroup of eight waves owns 32 output columns; wave w computes stream w / 2's
// 16 x 16 gate tile at columns 16 (w % 2) of them, with lo staged in LDS as f16 and the Q5_0
// weights dequantized to f16 per 16-wide step (A lane l: row l % 16, k 8h .. 8h + 7; B lane l:
// column l % 16, the same k; D lane l: column l % 16, rows 8h .. 8h + 7). The streams' gated
// values meet in LDS for the mean. With Q8 the 32 columns form one q8_1 block per batch row.
#if defined(__gfx1200__) || defined(__gfx1201__)
#define QG_WMMA_GFX12 1
typedef _Float16 qg_h8 __attribute__((ext_vector_type(8)));
typedef float qg_f8 __attribute__((ext_vector_type(8)));
#endif
#define QGW_K 320
#define QGW_GROUPS 4

template <bool Q8>
__device__ __forceinline__ void qgemv_gate_mean_wmma_q5_0(const __nv_bfloat16 *__restrict__ lo, const int m,
                                                          const void *__restrict__ w,
                                                          const __nv_bfloat16 *__restrict__ xn,
                                                          const int hidden, __nv_bfloat16 *__restrict__ out,
                                                          block_q8_1 *__restrict__ q8) {
#ifdef QG_WMMA_GFX12
  __shared__ __attribute__((aligned(16))) _Float16 lo_s[16 * QGW_K];
  __shared__ float gated[QGW_GROUPS][16][32];
  const int lane = threadIdx.x % 32, wave = threadIdx.x / 32;
  const int h = lane >> 4, col = lane & 15;
  for (int i = threadIdx.x; i < 16 * QGW_K; i += blockDim.x) {
    lo_s[i] = i < m * QGW_K ? (_Float16)__bfloat162float(lo[i]) : (_Float16)0.f;
  }
  __syncthreads();
  const int s = wave >> 1, ct = wave & 1;
  const int c0 = blockIdx.x * 32 + ct * 16;
  const block_q5_0 *wr = (const block_q5_0 *)w + ((int64_t)s * hidden + c0 + col) * (QGW_K / 32);
  qg_f8 d = {};
#pragma unroll
  for (int st = 0; st < QGW_K / 16; ++st) {
    const qg_h8 a = *reinterpret_cast<const qg_h8 *>(lo_s + col * QGW_K + st * 16 + 8 * h);
    // Element t of a block is nibble t / 16 of qs[t % 16] with bit t of qh above it, minus 16:
    // this step's elements are 16 (st % 2) + 8h + e.
    const block_q5_0 *bq = wr + st / 2;
    const int hi = st % 2;
    const uint16_t *qh16 = (const uint16_t *)bq->qh;
    const uint32_t qh = ((uint32_t)qh16[0] | ((uint32_t)qh16[1] << 16)) >> (16 * hi + 8 * h);
    const float dd = __half2float(*(const half *)&bq->d);
    qg_h8 b;
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const int q = ((bq->qs[8 * h + e] >> (4 * hi)) & 0xF) | (((qh >> e) & 1) << 4);
      b[e] = (_Float16)(dd * (float)(q - 16));
    }
    d = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, d);
  }
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int b = 8 * h + i;
    gated[s][b][ct * 16 + col] =
        b < m ? __bfloat162float(xn[(int64_t)b * QGW_GROUPS * hidden + (int64_t)s * hidden + c0 + col]) /
                    (1.f + __expf(-d[i]))
              : 0.f;
  }
  __syncthreads();
  for (int o = threadIdx.x; o < 16 * 32; o += blockDim.x) {
    const int b = o / 32, c = o % 32;
    if (b < m) {
      const __nv_bfloat16 v = __float2bfloat16(
          (gated[0][b][c] + gated[1][b][c] + gated[2][b][c] + gated[3][b][c]) / QGW_GROUPS);
      out[(int64_t)b * hidden + blockIdx.x * 32 + c] = v;
      // Only this thread reads (b, c): it can hold the stored value for the quantization.
      gated[0][b][c] = __bfloat162float(v);
    }
  }
  if constexpr (Q8) {
    // As qgemv_gate_mean_fixed: quantize the stored bf16 values, one block per batch row.
    __syncthreads();
    for (int b = wave; b < m; b += 8) {
      const float xi = gated[0][b][lane];
      float amax = fabsf(xi);
      float sum = xi;
      for (int off = 16; off > 0; off >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, off, 32));
        sum += __shfl_xor_sync(0xffffffff, sum, off, 32);
      }
      const float dq = amax / 127.0f;
      block_q8_1 *blk = q8 + (size_t)b * (hidden / 32) + blockIdx.x;
      blk->qs[lane] = amax == 0.0f ? 0 : (int8_t)roundf(xi / dq);
      if (lane == 0) {
        reinterpret_cast<half &>(blk->ds.x) = (half)dq;
        reinterpret_cast<half &>(blk->ds.y) = (half)sum;
      }
    }
  }
#else
  (void)lo, (void)m, (void)w, (void)xn, (void)hidden, (void)out, (void)q8;
#endif
}

// Same signatures as qgemv_gate_mean_bf16_k320_g4_q5_0(_q8); 256 threads, hidden / 32 workgroups.
extern "C" __global__ void __launch_bounds__(256) qgemv_gate_mean_bf16_k320_g4_q5_0_wmma(
    const __nv_bfloat16 *__restrict__ lo, const int m, const int k, const void *__restrict__ w,
    const int t, const __nv_bfloat16 *__restrict__ xn, const int groups, const int hidden,
    __nv_bfloat16 *__restrict__ out) {
  qgemv_gate_mean_wmma_q5_0<false>(lo, m, w, xn, hidden, out, nullptr);
}
extern "C" __global__ void __launch_bounds__(256) qgemv_gate_mean_bf16_k320_g4_q5_0_q8_wmma(
    const __nv_bfloat16 *__restrict__ lo, const int m, const int k, const void *__restrict__ w,
    const int t, const __nv_bfloat16 *__restrict__ xn, const int groups, const int hidden,
    __nv_bfloat16 *__restrict__ out, void *__restrict__ q8) {
  qgemv_gate_mean_wmma_q5_0<true>(lo, m, w, xn, hidden, out, (block_q8_1 *)q8);
}

