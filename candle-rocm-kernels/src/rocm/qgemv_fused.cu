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
