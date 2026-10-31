// The MMVQ kernels from candle-kernels, plus a single-token variant that gives
// each output row to one warp. The shared kernel splits a row across four
// warps, which leaves most lanes idle on the short rows (a few thousand
// columns) common in decode on wave32 RDNA GPUs.
#include "mmvq_gguf.cu"

template <typename dst_t, int qk, int qi, typename block_q_t, int vdr,
          vec_dot_q_cuda_t vec_dot_q_cuda>
static __device__ void mmvq_warp_row_impl(const void *__restrict__ vx,
                                          const block_q8_1 *__restrict__ y,
                                          dst_t *__restrict__ dst, const int ncols_x,
                                          const int nrows_x, const int block) {
  const int row = block * blockDim.y + threadIdx.y;
  if (row >= nrows_x) {
    return;
  }
  const int blocks_per_row_x = ncols_x / qk;
  constexpr int blocks_per_iter = vdr * WARP_SIZE / qi;

  float tmp = 0.0f;
  for (int kbx = threadIdx.x / (qi / vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
    const int kby = kbx * (qk / QK8_1);
    const int kqs = vdr * (threadIdx.x % (qi / vdr));
    tmp += vec_dot_q_cuda(vx, &y[kby], row * blocks_per_row_x + kbx, kqs);
  }
  tmp = warp_reduce_sum_f32(tmp);
  if (threadIdx.x == 0) {
    dst[row] = (dst_t)tmp;
  }
}

// The same for 2..8 activation columns (decode batches): each warp still owns one
// weight row and reads it once, keeping one running sum per column.
template <typename dst_t, int ncols_y, int qk, int qi, typename block_q_t, int vdr,
          vec_dot_q_cuda_t vec_dot_q_cuda>
static __device__ void mmvq_warp_row_multi_impl(const void *__restrict__ vx,
                                                const block_q8_1 *__restrict__ y,
                                                dst_t *__restrict__ dst, const int ncols_x,
                                                const int nrows_x, const int stride_col_y,
                                                const int block) {
  const int row = block * blockDim.y + threadIdx.y;
  if (row >= nrows_x) {
    return;
  }
  const int blocks_per_row_x = ncols_x / qk;
  constexpr int blocks_per_iter = vdr * WARP_SIZE / qi;

  float tmp[ncols_y];
#pragma unroll
  for (int j = 0; j < ncols_y; ++j) {
    tmp[j] = 0.0f;
  }
  for (int kbx = threadIdx.x / (qi / vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
    const int kby = kbx * (qk / QK8_1);
    const int kqs = vdr * (threadIdx.x % (qi / vdr));
#pragma unroll
    for (int j = 0; j < ncols_y; ++j) {
      tmp[j] += vec_dot_q_cuda(vx, &y[j * stride_col_y + kby], row * blocks_per_row_x + kbx, kqs);
    }
  }
#pragma unroll
  for (int j = 0; j < ncols_y; ++j) {
    const float v = warp_reduce_sum_f32(tmp[j]);
    if (threadIdx.x == 0) {
      dst[j * nrows_x + row] = (dst_t)v;
    }
  }
}

// Weights with few rows (k > 8 * rows: routers, gates, small projections) against the 12..32
// activation columns of a speculative verify step. One row per wave leaves most of the GPU
// idle on a few dozen rows, so a block's KSPLIT waves share one row and split its K blocks;
// lane 0 of each wave parks its column sums in LDS and the first wave adds them. Without it
// these weights were dequantized every step for a dense GEMM, 12-22 us a call at twelve rows.
template <typename dst_t, int ncols_y, int qk, int qi, typename block_q_t, int vdr,
          vec_dot_q_cuda_t vec_dot_q_cuda, int ksplit>
static __device__ void mmvq_splitk_multi_impl(const void *__restrict__ vx,
                                              const block_q8_1 *__restrict__ y,
                                              dst_t *__restrict__ dst, const int ncols_x,
                                              const int nrows_x, const int stride_col_y) {
  const int row = blockIdx.x;
  const int blocks_per_row_x = ncols_x / qk;
  constexpr int lanes_per_block = qi / vdr;
  constexpr int blocks_per_iter = WARP_SIZE * ksplit / lanes_per_block;
  const int tid = threadIdx.y * WARP_SIZE + threadIdx.x;

  float tmp[ncols_y];
#pragma unroll
  for (int j = 0; j < ncols_y; ++j) {
    tmp[j] = 0.0f;
  }
  const int kqs = vdr * (tid % lanes_per_block);
  for (int kbx = tid / lanes_per_block; kbx < blocks_per_row_x; kbx += blocks_per_iter) {
    const int kby = kbx * (qk / QK8_1);
#pragma unroll
    for (int j = 0; j < ncols_y; ++j) {
      tmp[j] += vec_dot_q_cuda(vx, &y[j * stride_col_y + kby], row * blocks_per_row_x + kbx, kqs);
    }
  }
  __shared__ float part[ksplit][ncols_y];
#pragma unroll
  for (int j = 0; j < ncols_y; ++j) {
    const float v = warp_reduce_sum_f32(tmp[j]);
    if (threadIdx.x == 0) {
      part[threadIdx.y][j] = v;
    }
  }
  __syncthreads();
  if (threadIdx.y == 0 && threadIdx.x < ncols_y) {
    float v = 0.0f;
#pragma unroll
    for (int w = 0; w < ksplit; ++w) {
      v += part[w][threadIdx.x];
    }
    dst[threadIdx.x * nrows_x + row] = (dst_t)v;
  }
}

// Q4_K rows against several activation columns: the weight block's nibbles and scales are
// unpacked once per lane and reused for every column, where the generic kernel redoes them
// per column and ran at a third of memory bandwidth on three-row decode batches.
// `nrows_w` weight rows per wave share each activation load.
template <typename dst_t, int ncols_y, int nrows_w>
static __device__ void mmvq_warp_row_multi_q4k_impl(const void *__restrict__ vx,
                                                    const block_q8_1 *__restrict__ y,
                                                    dst_t *__restrict__ dst, const int ncols_x,
                                                    const int nrows_x, const int stride_col_y,
                                                    const int block) {
  const int row0 = (block * blockDim.y + threadIdx.y) * nrows_w;
  if (row0 >= nrows_x) {
    return;
  }
  constexpr int qk = QK_K, qi = QI4_K, vdr = VDR_Q4_K_Q8_1_MMVQ;
  const int blocks_per_row_x = ncols_x / qk;
  constexpr int blocks_per_iter = vdr * WARP_SIZE / qi;
  const int iqs = vdr * (threadIdx.x % (qi / vdr));
  // Lane-constant offsets into a block (see vec_dot_q4_K_q8_1).
  const int bq8_offset = QR4_K * ((iqs / 2) / (QI8_1 / 2));
  const int q4_off = 16 * bq8_offset + 4 * ((iqs / 2) % 4);
  const int q8_off = (iqs / 2) % 4;
  const int j = bq8_offset / 2;

  float tmp[nrows_w][ncols_y];
#pragma unroll
  for (int r = 0; r < nrows_w; ++r) {
#pragma unroll
    for (int c = 0; c < ncols_y; ++c) {
      tmp[r][c] = 0.0f;
    }
  }
  for (int kbx = threadIdx.x / (qi / vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
    int vi[nrows_w][QR4_K][2];
    float scm[nrows_w][QR4_K][2];
    float2 dm[nrows_w];
#pragma unroll
    for (int r = 0; r < nrows_w; ++r) {
      const block_q4_K *bq4 = (const block_q4_K *)vx + (row0 + r) * blocks_per_row_x + kbx;
      const int *q4 = (const int *)(bq4->qs + q4_off);
      const int v0 = q4[0], v1 = q4[4];
      const uint16_t *scales = (const uint16_t *)bq4->scales;
      uint16_t aux[2];
      if (j < 2) {
        aux[0] = scales[j + 0] & 0x3f3f;
        aux[1] = scales[j + 2] & 0x3f3f;
      } else {
        aux[0] = ((scales[j + 2] >> 0) & 0x0f0f) | ((scales[j - 2] & 0xc0c0) >> 2);
        aux[1] = ((scales[j + 2] >> 4) & 0x0f0f) | ((scales[j - 0] & 0xc0c0) >> 2);
      }
      const uint8_t *sc = (const uint8_t *)aux;
      dm[r] = __half22float2(bq4->dm);
#pragma unroll
      for (int i = 0; i < QR4_K; ++i) {
        vi[r][i][0] = (v0 >> (4 * i)) & 0x0F0F0F0F;
        vi[r][i][1] = (v1 >> (4 * i)) & 0x0F0F0F0F;
        scm[r][i][0] = sc[i];
        scm[r][i][1] = sc[2 + i];
      }
    }
    const int kby = kbx * (qk / QK8_1);
#pragma unroll
    for (int c = 0; c < ncols_y; ++c) {
      const block_q8_1 *bq8 = y + c * stride_col_y + kby + bq8_offset;
      float sumf_d[nrows_w] = {}, sumf_m[nrows_w] = {};
#pragma unroll
      for (int i = 0; i < QR4_K; ++i) {
        const int *q8 = (const int *)bq8[i].qs + q8_off;
        const int u0 = q8[0], u1 = q8[4];
        const float d8 = __low2float(bq8[i].ds);
        const int dot2 = ggml_cuda_dp4a(0x01010101, u1, ggml_cuda_dp4a(0x01010101, u0, 0));
#pragma unroll
        for (int r = 0; r < nrows_w; ++r) {
          const int dot1 = ggml_cuda_dp4a(vi[r][i][1], u1, ggml_cuda_dp4a(vi[r][i][0], u0, 0));
          sumf_d[r] += d8 * (dot1 * scm[r][i][0]);
          sumf_m[r] += d8 * (dot2 * scm[r][i][1]);
        }
      }
#pragma unroll
      for (int r = 0; r < nrows_w; ++r) {
        tmp[r][c] += dm[r].x * sumf_d[r] - dm[r].y * sumf_m[r];
      }
    }
  }
#pragma unroll
  for (int r = 0; r < nrows_w; ++r) {
#pragma unroll
    for (int c = 0; c < ncols_y; ++c) {
      const float v = warp_reduce_sum_f32(tmp[r][c]);
      if (threadIdx.x == 0 && row0 + r < nrows_x) {
        dst[c * nrows_x + row0 + r] = (dst_t)v;
      }
    }
  }
}

#define MMVQ_WARP_ROW_MULTI_Q4K_ENTRY(dst_tag, dst_c_type, n)                                             \
  extern "C" __global__ void mmvq_gguf_q4_k_##dst_tag##_warp_row##n(                                        \
      const void *__restrict__ vx, const void *__restrict__ vy, dst_c_type *__restrict__ dst,               \
      const int ncols_x, const int nrows_x, const int stride_col_y) {                                       \
    mmvq_warp_row_multi_q4k_impl<dst_c_type, n, 1>(vx, (const block_q8_1 *)vy, dst, ncols_x, nrows_x,      \
                                                   stride_col_y, blockIdx.x);                               \
  }                                                                                                         \
  extern "C" __global__ void mmvq_gguf_q4_k_##dst_tag##_warp_row##n##x2(                                    \
      const void *__restrict__ vx, const void *__restrict__ vy, dst_c_type *__restrict__ dst,               \
      const int ncols_x, const int nrows_x, const int stride_col_y) {                                       \
    mmvq_warp_row_multi_q4k_impl<dst_c_type, n, 2>(vx, (const block_q8_1 *)vy, dst, ncols_x, nrows_x,      \
                                                   stride_col_y, blockIdx.x);                               \
  }
#define MMVQ_WARP_ROW_MULTI_Q4K_DST(dst_tag, dst_c_type)                                                  \
  MMVQ_WARP_ROW_MULTI_Q4K_ENTRY(dst_tag, dst_c_type, 2)                                                   \
  MMVQ_WARP_ROW_MULTI_Q4K_ENTRY(dst_tag, dst_c_type, 3)                                                   \
  MMVQ_WARP_ROW_MULTI_Q4K_ENTRY(dst_tag, dst_c_type, 4)                                                   \
  MMVQ_WARP_ROW_MULTI_Q4K_ENTRY(dst_tag, dst_c_type, 5)                                                   \
  MMVQ_WARP_ROW_MULTI_Q4K_ENTRY(dst_tag, dst_c_type, 6)                                                   \
  MMVQ_WARP_ROW_MULTI_Q4K_ENTRY(dst_tag, dst_c_type, 7)                                                   \
  MMVQ_WARP_ROW_MULTI_Q4K_ENTRY(dst_tag, dst_c_type, 8)                                                   \
  MMVQ_WARP_ROW_MULTI_Q4K_ENTRY(dst_tag, dst_c_type, 12)                                                  \
  MMVQ_WARP_ROW_MULTI_Q4K_ENTRY(dst_tag, dst_c_type, 16)                                                  \
  MMVQ_WARP_ROW_MULTI_Q4K_ENTRY(dst_tag, dst_c_type, 24)                                                  \
  MMVQ_WARP_ROW_MULTI_Q4K_ENTRY(dst_tag, dst_c_type, 32)

#define MMVQ_WARP_ROW_MULTI_ENTRY(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot, dst_tag, dst_c_type, n) \
  extern "C" __global__ void mmvq_gguf_##tag##_##dst_tag##_warp_row##n(                                     \
      const void *__restrict__ vx, const void *__restrict__ vy, dst_c_type *__restrict__ dst,               \
      const int ncols_x, const int nrows_x, const int stride_col_y) {                                       \
    mmvq_warp_row_multi_impl<dst_c_type, n, qk_val, qi_val, block_q_t, vdr_val, vec_dot>(                   \
        vx, (const block_q8_1 *)vy, dst, ncols_x, nrows_x, stride_col_y, blockIdx.x);                       \
  }

#define MMVQ_WARP_ROW_MULTI_DST(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot, dst_tag, dst_c_type)    \
  MMVQ_WARP_ROW_MULTI_ENTRY(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot, dst_tag, dst_c_type, 2)     \
  MMVQ_WARP_ROW_MULTI_ENTRY(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot, dst_tag, dst_c_type, 3)     \
  MMVQ_WARP_ROW_MULTI_ENTRY(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot, dst_tag, dst_c_type, 4)     \
  MMVQ_WARP_ROW_MULTI_ENTRY(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot, dst_tag, dst_c_type, 5)     \
  MMVQ_WARP_ROW_MULTI_ENTRY(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot, dst_tag, dst_c_type, 6)     \
  MMVQ_WARP_ROW_MULTI_ENTRY(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot, dst_tag, dst_c_type, 7)     \
  MMVQ_WARP_ROW_MULTI_ENTRY(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot, dst_tag, dst_c_type, 8)     \
  MMVQ_WARP_ROW_MULTI_ENTRY(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot, dst_tag, dst_c_type, 12)    \
  MMVQ_WARP_ROW_MULTI_ENTRY(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot, dst_tag, dst_c_type, 16)    \
  MMVQ_WARP_ROW_MULTI_ENTRY(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot, dst_tag, dst_c_type, 24)    \
  MMVQ_WARP_ROW_MULTI_ENTRY(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot, dst_tag, dst_c_type, 32)

#define MMVQ_SPLITK_WAVES 4
#define MMVQ_SPLITK_ENTRY(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot, dst_tag, dst_c_type, n)     \
  extern "C" __global__ void __launch_bounds__(WARP_SIZE * MMVQ_SPLITK_WAVES)                          \
      mmvq_gguf_##tag##_##dst_tag##_splitk##n(const void *__restrict__ vx, const void *__restrict__ vy, \
                                              dst_c_type *__restrict__ dst, const int ncols_x,          \
                                              const int nrows_x, const int stride_col_y) {              \
    mmvq_splitk_multi_impl<dst_c_type, n, qk_val, qi_val, block_q_t, vdr_val, vec_dot,                  \
                           MMVQ_SPLITK_WAVES>(vx, (const block_q8_1 *)vy, dst, ncols_x, nrows_x,       \
                                              stride_col_y);                                            \
  }

#define MMVQ_SPLITK_DST(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot, dst_tag, dst_c_type)     \
  MMVQ_SPLITK_ENTRY(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot, dst_tag, dst_c_type, 12)     \
  MMVQ_SPLITK_ENTRY(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot, dst_tag, dst_c_type, 16)     \
  MMVQ_SPLITK_ENTRY(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot, dst_tag, dst_c_type, 24)     \
  MMVQ_SPLITK_ENTRY(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot, dst_tag, dst_c_type, 32)

#define MMVQ_SPLITK_SET(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot)                       \
  MMVQ_SPLITK_DST(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot, bf16, __nv_bfloat16)        \
  MMVQ_SPLITK_DST(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot, f16, half)                  \
  MMVQ_SPLITK_DST(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot, f32, float)

#define MMVQ_WARP_ROW_ENTRY(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot, dst_tag, dst_c_type) \
  extern "C" __global__ void mmvq_gguf_##tag##_##dst_tag##_warp_row(                              \
      const void *__restrict__ vx, const void *__restrict__ vy, dst_c_type *__restrict__ dst,     \
      const int ncols_x, const int nrows_x) {                                                     \
    mmvq_warp_row_impl<dst_c_type, qk_val, qi_val, block_q_t, vdr_val, vec_dot>(                  \
        vx, (const block_q8_1 *)vy, dst, ncols_x, nrows_x, blockIdx.x);                           \
  }

#define MMVQ_WARP_ROW_SET(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot)                       \
  MMVQ_WARP_ROW_ENTRY(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot, bf16, __nv_bfloat16)      \
  MMVQ_WARP_ROW_ENTRY(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot, f16, half)                \
  MMVQ_WARP_ROW_ENTRY(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot, f32, float)               \
  MMVQ_WARP_ROW_MULTI_DST(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot, bf16, __nv_bfloat16)  \
  MMVQ_WARP_ROW_MULTI_DST(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot, f16, half)            \
  MMVQ_WARP_ROW_MULTI_DST(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot, f32, float)            \
  MMVQ_SPLITK_SET(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot)

MMVQ_WARP_ROW_SET(q4_0, block_q4_0, QK4_0, QI4_0, VDR_Q4_0_Q8_1_MMVQ, vec_dot_q4_0_q8_1)
MMVQ_WARP_ROW_SET(q4_1, block_q4_1, QK4_1, QI4_1, VDR_Q4_1_Q8_1_MMVQ, vec_dot_q4_1_q8_1)
MMVQ_WARP_ROW_SET(q5_0, block_q5_0, QK5_0, QI5_0, VDR_Q5_0_Q8_1_MMVQ, vec_dot_q5_0_q8_1)
MMVQ_WARP_ROW_SET(q5_1, block_q5_1, QK5_1, QI5_1, VDR_Q5_1_Q8_1_MMVQ, vec_dot_q5_1_q8_1)
MMVQ_WARP_ROW_SET(q8_0, block_q8_0, QK8_0, QI8_0, VDR_Q8_0_Q8_1_MMVQ, vec_dot_q8_0_q8_1)
MMVQ_WARP_ROW_SET(q2_k, block_q2_K, QK_K, QI2_K, VDR_Q2_K_Q8_1_MMVQ, vec_dot_q2_K_q8_1)
MMVQ_WARP_ROW_SET(q3_k, block_q3_K, QK_K, QI3_K, VDR_Q3_K_Q8_1_MMVQ, vec_dot_q3_K_q8_1)
MMVQ_WARP_ROW_ENTRY(q4_k, block_q4_K, QK_K, QI4_K, VDR_Q4_K_Q8_1_MMVQ, vec_dot_q4_K_q8_1, bf16, __nv_bfloat16)
MMVQ_WARP_ROW_ENTRY(q4_k, block_q4_K, QK_K, QI4_K, VDR_Q4_K_Q8_1_MMVQ, vec_dot_q4_K_q8_1, f16, half)
MMVQ_WARP_ROW_ENTRY(q4_k, block_q4_K, QK_K, QI4_K, VDR_Q4_K_Q8_1_MMVQ, vec_dot_q4_K_q8_1, f32, float)
MMVQ_WARP_ROW_MULTI_Q4K_DST(bf16, __nv_bfloat16)
MMVQ_WARP_ROW_MULTI_Q4K_DST(f16, half)
MMVQ_WARP_ROW_MULTI_Q4K_DST(f32, float)
MMVQ_SPLITK_SET(q4_k, block_q4_K, QK_K, QI4_K, VDR_Q4_K_Q8_1_MMVQ, vec_dot_q4_K_q8_1)
MMVQ_WARP_ROW_SET(q5_k, block_q5_K, QK_K, QI5_K, VDR_Q5_K_Q8_1_MMVQ, vec_dot_q5_K_q8_1)
MMVQ_WARP_ROW_SET(q6_k, block_q6_K, QK_K, QI6_K, VDR_Q6_K_Q8_1_MMVQ, vec_dot_q6_K_q8_1)

// ---- Several weight tables against one quantized activation in a single launch -------------
// Decode steps run groups of projections over the same activations (a GDN layer's qkv, z, a
// and b; a QSA layer's q, kv and indexer): each was its own launch, and in a HIP graph every
// node costs a few microseconds of dispatch. One grid covers all tables, a contiguous range of
// blocks per table, each block dispatching on its table's format.
#define MMVQ_MULTI_MAX_SEGMENTS 4
#define MMVQ_MULTI_TAG_Q4_0 0
#define MMVQ_MULTI_TAG_Q5_0 1
#define MMVQ_MULTI_TAG_Q8_0 2
#define MMVQ_MULTI_TAG_Q4_K 3
#define MMVQ_MULTI_TAG_Q5_K 4
#define MMVQ_MULTI_TAG_Q6_K 5

struct MmvqMultiSegments {
  const void *vx[MMVQ_MULTI_MAX_SEGMENTS];
  void *dst[MMVQ_MULTI_MAX_SEGMENTS];
  int nrows[MMVQ_MULTI_MAX_SEGMENTS];
  int block_start[MMVQ_MULTI_MAX_SEGMENTS + 1];
  int tag[MMVQ_MULTI_MAX_SEGMENTS];
  int x2[MMVQ_MULTI_MAX_SEGMENTS];
  int count;
};

template <typename dst_t, int ncols_y, int qk, int qi, typename block_q_t, int vdr,
          vec_dot_q_cuda_t vec_dot>
static __device__ __forceinline__ void mmvq_multi_dispatch(const void *vx, const block_q8_1 *y,
                                                           dst_t *dst, const int ncols_x,
                                                           const int nrows_x, const int stride_col_y,
                                                           const int block) {
  if constexpr (ncols_y == 1) {
    mmvq_warp_row_impl<dst_t, qk, qi, block_q_t, vdr, vec_dot>(vx, y, dst, ncols_x, nrows_x, block);
  } else {
    mmvq_warp_row_multi_impl<dst_t, ncols_y, qk, qi, block_q_t, vdr, vec_dot>(
        vx, y, dst, ncols_x, nrows_x, stride_col_y, block);
  }
}

template <typename dst_t, int ncols_y>
static __device__ void mmvq_multi_impl(const MmvqMultiSegments s, const block_q8_1 *__restrict__ y,
                                       const int ncols_x, const int stride_col_y) {
  int seg = 0;
  while (seg + 1 < s.count && (int)blockIdx.x >= s.block_start[seg + 1]) {
    ++seg;
  }
  const int block = blockIdx.x - s.block_start[seg];
  const void *vx = s.vx[seg];
  dst_t *dst = reinterpret_cast<dst_t *>(s.dst[seg]);
  const int nrows = s.nrows[seg];
  switch (s.tag[seg]) {
  case MMVQ_MULTI_TAG_Q4_0:
    mmvq_multi_dispatch<dst_t, ncols_y, QK4_0, QI4_0, block_q4_0, VDR_Q4_0_Q8_1_MMVQ, vec_dot_q4_0_q8_1>(
        vx, y, dst, ncols_x, nrows, stride_col_y, block);
    break;
  case MMVQ_MULTI_TAG_Q5_0:
    mmvq_multi_dispatch<dst_t, ncols_y, QK5_0, QI5_0, block_q5_0, VDR_Q5_0_Q8_1_MMVQ, vec_dot_q5_0_q8_1>(
        vx, y, dst, ncols_x, nrows, stride_col_y, block);
    break;
  case MMVQ_MULTI_TAG_Q8_0:
    mmvq_multi_dispatch<dst_t, ncols_y, QK8_0, QI8_0, block_q8_0, VDR_Q8_0_Q8_1_MMVQ, vec_dot_q8_0_q8_1>(
        vx, y, dst, ncols_x, nrows, stride_col_y, block);
    break;
  case MMVQ_MULTI_TAG_Q4_K:
    if constexpr (ncols_y == 1) {
      mmvq_warp_row_impl<dst_t, QK_K, QI4_K, block_q4_K, VDR_Q4_K_Q8_1_MMVQ, vec_dot_q4_K_q8_1>(
          vx, y, dst, ncols_x, nrows, block);
    } else if (s.x2[seg]) {
      mmvq_warp_row_multi_q4k_impl<dst_t, ncols_y, 2>(vx, y, dst, ncols_x, nrows, stride_col_y, block);
    } else {
      mmvq_warp_row_multi_q4k_impl<dst_t, ncols_y, 1>(vx, y, dst, ncols_x, nrows, stride_col_y, block);
    }
    break;
  case MMVQ_MULTI_TAG_Q5_K:
    mmvq_multi_dispatch<dst_t, ncols_y, QK_K, QI5_K, block_q5_K, VDR_Q5_K_Q8_1_MMVQ, vec_dot_q5_K_q8_1>(
        vx, y, dst, ncols_x, nrows, stride_col_y, block);
    break;
  default:
    mmvq_multi_dispatch<dst_t, ncols_y, QK_K, QI6_K, block_q6_K, VDR_Q6_K_Q8_1_MMVQ, vec_dot_q6_K_q8_1>(
        vx, y, dst, ncols_x, nrows, stride_col_y, block);
    break;
  }
}

#define MMVQ_MULTI_ENTRY(dst_tag, dst_c_type, n)                                                   \
  extern "C" __global__ void mmvq_gguf_multi_##dst_tag##_seg##n(                                   \
      const MmvqMultiSegments segments, const void *__restrict__ vy, const int ncols_x,            \
      const int stride_col_y) {                                                                    \
    mmvq_multi_impl<dst_c_type, n>(segments, (const block_q8_1 *)vy, ncols_x, stride_col_y);       \
  }
#define MMVQ_MULTI_DST(dst_tag, dst_c_type)                                                        \
  MMVQ_MULTI_ENTRY(dst_tag, dst_c_type, 1)                                                         \
  MMVQ_MULTI_ENTRY(dst_tag, dst_c_type, 2)                                                         \
  MMVQ_MULTI_ENTRY(dst_tag, dst_c_type, 3)                                                         \
  MMVQ_MULTI_ENTRY(dst_tag, dst_c_type, 4)                                                         \
  MMVQ_MULTI_ENTRY(dst_tag, dst_c_type, 5)                                                         \
  MMVQ_MULTI_ENTRY(dst_tag, dst_c_type, 6)                                                         \
  MMVQ_MULTI_ENTRY(dst_tag, dst_c_type, 7)                                                         \
  MMVQ_MULTI_ENTRY(dst_tag, dst_c_type, 8)
MMVQ_MULTI_DST(bf16, __nv_bfloat16)
MMVQ_MULTI_DST(f16, half)
MMVQ_MULTI_DST(f32, float)

// Q4_K weights times q8_1 activations for up to 16 activation rows on gfx12's int8 WMMA
// (v_wmma_i32_16x16x16_iu8). Activation row m is A's row m and weight row n0 + n is B's
// column n: each 32-element sub-block is two 16x16x16 steps whose int32 product is scaled into
// f32 by the weight's sub-block scale and the activation block's scale, and the min term comes
// from the activation block's sum. The dp4a kernels redo the scale arithmetic once per
// activation row and are VALU-bound past a few rows (a third of memory bandwidth at 12); here
// it is once per 16. A workgroup of blockDim.y waves shares one 16-row weight tile, each wave
// taking every blockDim.y-th super-block of K, and the partial sums meet in LDS.
//
// gfx12 wave32 operand layouts (lane l, h = l / 16): A holds row l % 16, k = 8h .. 8h + 7;
// B holds column l % 16, the same k; D holds column l % 16, rows 8h .. 8h + 7.
#define MMQ_WMMA_MAX_SPLIT 8
#if defined(__gfx1200__) || defined(__gfx1201__)
#define MMQ_WMMA_GFX12 1
#endif

extern "C" __global__ void mmq_wmma_available(int *flag) {
#ifdef MMQ_WMMA_GFX12
  *flag = 1;
#else
  (void)flag;
#endif
}

#ifdef MMQ_WMMA_GFX12
typedef int mmq_v2i __attribute__((ext_vector_type(2)));
typedef int mmq_v8i __attribute__((ext_vector_type(8)));

static __device__ __forceinline__ void mmq_q4k_scale_min(const uint8_t *q, const int j, int &sc,
                                                         int &m) {
  if (j < 4) {
    sc = q[j] & 63;
    m = q[j + 4] & 63;
  } else {
    sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
    m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
  }
}

// The workgroup's waves add their partial sums in LDS and wave 0 stores the tile: lane l holds
// rows 8 * (l / 16) + i of weight row n0 + l % 16.
template <typename dst_t>
static __device__ __forceinline__ void mmq_wmma_finish(float (&acc)[8], dst_t *__restrict__ dst,
                                                       const int n0, const int nrows_x,
                                                       const int ncols_y) {
  __shared__ float part[MMQ_WMMA_MAX_SPLIT - 1][8][WARP_SIZE];
  const int lane = threadIdx.x, wave = threadIdx.y;
  if (wave > 0) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      part[wave - 1][i][lane] = acc[i];
    }
  }
  __syncthreads();
  if (wave > 0) {
    return;
  }
  for (int w = 1; w < (int)blockDim.y; ++w) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      acc[i] += part[w - 1][i][lane];
    }
  }
  const int nl = lane & 15, h = lane >> 4;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int m = 8 * h + i;
    if (m < ncols_y && n0 + nl < nrows_x) {
      dst[(size_t)m * nrows_x + n0 + nl] = (dst_t)acc[i];
    }
  }
}

// A 32-bit word of a 2-byte-aligned field (Q6_K's 210-byte and Q5_0's 22-byte blocks).
static __device__ __forceinline__ uint32_t mmq_ld32(const uint8_t *p) {
  const uint16_t *q = (const uint16_t *)p;
  return (uint32_t)q[0] | ((uint32_t)q[1] << 16);
}

// Bytes x - c for bytes x >= c and c < 128: the high bit set first keeps borrows inside each byte.
static __device__ __forceinline__ uint32_t mmq_sub_bytes(const uint32_t x, const uint32_t c) {
  return ((x | 0x80808080u) - c) ^ 0x80808080u;
}

// Loads row m's 8 q8_1 values for this lane's half (h) of each 16-wide step of block `bq`, and
// the block's (scale, sum).
static __device__ __forceinline__ void mmq_load_q8(const block_q8_1 *__restrict__ bq, const int h,
                                                   const bool ok, mmq_v2i &a0, mmq_v2i &a1,
                                                   uint32_t &ds) {
  a0 = (mmq_v2i){0, 0};
  a1 = (mmq_v2i){0, 0};
  ds = 0;
  if (ok) {
    const int *qa = (const int *)(bq->qs + 8 * h);
    a0 = (mmq_v2i){qa[0], qa[1]};
    a1 = (mmq_v2i){qa[4], qa[5]};
    ds = *(const uint32_t *)&bq->ds;
  }
}
#endif

template <typename dst_t>
static __device__ void mmq_wmma_q4k_impl(const void *__restrict__ vx,
                                         const block_q8_1 *__restrict__ y,
                                         dst_t *__restrict__ dst, const int ncols_x,
                                         const int nrows_x, const int stride_col_y,
                                         const int ncols_y) {
#ifdef MMQ_WMMA_GFX12
  const int lane = threadIdx.x, wave = threadIdx.y, split = blockDim.y;
  const int n0 = blockIdx.x * 16;
  const int nl = lane & 15, h = lane >> 4;
  // Rows past the matrix read the last row; their results are not stored.
  const int n = min(n0 + nl, nrows_x - 1);
  const int nb = ncols_x / QK_K;
  const block_q4_K *wrow = (const block_q4_K *)vx + (size_t)n * nb;
  const bool m_ok = nl < ncols_y;
  const block_q8_1 *yrow = y + (size_t)(m_ok ? nl : 0) * stride_col_y;
  float acc[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    acc[i] = 0.0f;
  }
  for (int kb = wave; kb < nb; kb += split) {
    const block_q4_K *b = wrow + kb;
    const float2 dm = __half22float2(b->dm);
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      // Sub-blocks 2c and 2c + 1 are the low and high nibbles of the same 32 bytes; a WMMA
      // step's 16 elements are 16 consecutive bytes, of which this lane takes 8.
      const uint32_t *q = (const uint32_t *)(b->qs + 32 * c + 8 * h);
      const uint32_t w0 = q[0], w1 = q[1], w2 = q[4], w3 = q[5];
#pragma unroll
      for (int hi = 0; hi < 2; ++hi) {
        const int j = 2 * c + hi;
        const int sh = 4 * hi;
        const mmq_v2i b0 = {(int)((w0 >> sh) & 0x0F0F0F0Fu), (int)((w1 >> sh) & 0x0F0F0F0Fu)};
        const mmq_v2i b1 = {(int)((w2 >> sh) & 0x0F0F0F0Fu), (int)((w3 >> sh) & 0x0F0F0F0Fu)};
        mmq_v2i a0 = {0, 0}, a1 = {0, 0};
        uint32_t ds = 0;
        if (m_ok) {
          const block_q8_1 *bq = yrow + kb * (QK_K / QK8_1) + j;
          const int *qa = (const int *)(bq->qs + 8 * h);
          a0 = (mmq_v2i){qa[0], qa[1]};
          a1 = (mmq_v2i){qa[4], qa[5]};
          ds = *(const uint32_t *)&bq->ds;
        }
        mmq_v8i d = {0, 0, 0, 0, 0, 0, 0, 0};
        d = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a0, false, b0, d, false);
        d = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a1, false, b1, d, false);
        int sc, mn;
        mmq_q4k_scale_min(b->scales, j, sc, mn);
        const float scw = dm.x * (float)sc, mw = dm.y * (float)mn;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          // Row 8h + i's activation scale and sum, from the lane that loaded that row.
          const uint32_t o = __shfl(ds, 8 * h + i, WARP_SIZE);
          const float2 f = __half22float2(*(const half2 *)&o);
          acc[i] = fmaf((float)d[i], f.x * scw, acc[i]);
          acc[i] = fmaf(-f.y, mw, acc[i]);
        }
      }
    }
  }
  mmq_wmma_finish(acc, dst, n0, nrows_x, ncols_y);
#else
  (void)vx, (void)y, (void)dst, (void)ncols_x, (void)nrows_x, (void)stride_col_y, (void)ncols_y;
#endif
}


// Q6_K: a 16-element WMMA step shares one int8 scale, so the two steps of each 32-element
// activation block are combined in int32 (D0 * sc0 + D1 * sc1) before one f32 scaling. Element
// e of a super-block (half p = e / 128, quarter g = e % 128 / 32, l = e % 32) is the nibble g / 2
// of ql[64p + 32(g % 2) + l] with bits 2g of qh[32p + l] above it, minus 32, times scales[8p +
// 2g + l / 16].
template <typename dst_t>
static __device__ void mmq_wmma_q6k_impl(const void *__restrict__ vx,
                                         const block_q8_1 *__restrict__ y,
                                         dst_t *__restrict__ dst, const int ncols_x,
                                         const int nrows_x, const int stride_col_y,
                                         const int ncols_y) {
#ifdef MMQ_WMMA_GFX12
  const int lane = threadIdx.x, wave = threadIdx.y, split = blockDim.y;
  const int n0 = blockIdx.x * 16;
  const int nl = lane & 15, h = lane >> 4;
  const int n = min(n0 + nl, nrows_x - 1);
  const int nb = ncols_x / QK_K;
  const block_q6_K *wrow = (const block_q6_K *)vx + (size_t)n * nb;
  const bool m_ok = nl < ncols_y;
  const block_q8_1 *yrow = y + (size_t)(m_ok ? nl : 0) * stride_col_y;
  float acc[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    acc[i] = 0.0f;
  }
  for (int kb = wave; kb < nb; kb += split) {
    const block_q6_K *b = wrow + kb;
    const float d = __half2float(b->d);
#pragma unroll
    for (int p = 0; p < 2; ++p) {
      // The high bits of all four quarters of this half: steps s = 0, 1 at byte 16s + 8h.
      const uint8_t *qh = b->qh + 32 * p + 8 * h;
      const uint32_t h00 = mmq_ld32(qh), h01 = mmq_ld32(qh + 4);
      const uint32_t h10 = mmq_ld32(qh + 16), h11 = mmq_ld32(qh + 20);
#pragma unroll
      for (int g = 0; g < 4; ++g) {
        const uint8_t *ql = b->ql + 64 * p + 32 * (g % 2) + 8 * h;
        const int lsh = 4 * (g / 2), hsh = 2 * g;
        auto q6 = [&](const uint32_t lo, const uint32_t hi) {
          return (int)mmq_sub_bytes(((lo >> lsh) & 0x0F0F0F0Fu) | (((hi >> hsh) & 0x03030303u) << 4),
                                    0x20202020u);
        };
        const mmq_v2i b0 = {q6(mmq_ld32(ql), h00), q6(mmq_ld32(ql + 4), h01)};
        const mmq_v2i b1 = {q6(mmq_ld32(ql + 16), h10), q6(mmq_ld32(ql + 20), h11)};
        mmq_v2i a0, a1;
        uint32_t ds;
        mmq_load_q8(yrow + kb * (QK_K / QK8_1) + 4 * p + g, h, m_ok, a0, a1, ds);
        const mmq_v8i zero = {0, 0, 0, 0, 0, 0, 0, 0};
        const mmq_v8i d0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a0, true, b0, zero, false);
        const mmq_v8i d1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a1, true, b1, zero, false);
        const int sc0 = b->scales[8 * p + 2 * g], sc1 = b->scales[8 * p + 2 * g + 1];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          const uint32_t o = __shfl(ds, 8 * h + i, WARP_SIZE);
          const float da = __low2float(*(const half2 *)&o);
          acc[i] = fmaf((float)(d0[i] * sc0 + d1[i] * sc1), da * d, acc[i]);
        }
      }
    }
  }
  mmq_wmma_finish(acc, dst, n0, nrows_x, ncols_y);
#else
  (void)vx, (void)y, (void)dst, (void)ncols_x, (void)nrows_x, (void)stride_col_y, (void)ncols_y;
#endif
}

// Q5_0: element t of a 32-element block is nibble t / 16 of qs[t % 16] with bit t of qh above
// it, minus 16, times d. Blocks are 22 bytes, so their fields are read in 16-bit pieces.
template <typename dst_t>
static __device__ void mmq_wmma_q5_0_impl(const void *__restrict__ vx,
                                          const block_q8_1 *__restrict__ y,
                                          dst_t *__restrict__ dst, const int ncols_x,
                                          const int nrows_x, const int stride_col_y,
                                          const int ncols_y) {
#ifdef MMQ_WMMA_GFX12
  const int lane = threadIdx.x, wave = threadIdx.y, split = blockDim.y;
  const int n0 = blockIdx.x * 16;
  const int nl = lane & 15, h = lane >> 4;
  const int n = min(n0 + nl, nrows_x - 1);
  const int nb = ncols_x / QK5_0;
  const block_q5_0 *wrow = (const block_q5_0 *)vx + (size_t)n * nb;
  const bool m_ok = nl < ncols_y;
  const block_q8_1 *yrow = y + (size_t)(m_ok ? nl : 0) * stride_col_y;
  float acc[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    acc[i] = 0.0f;
  }
  // Spreads the low four bits of x into bit 4 of four bytes.
  auto spread = [](const uint32_t x) {
    return ((x & 1u) | ((x & 2u) << 7) | ((x & 4u) << 14) | ((x & 8u) << 21)) << 4;
  };
  for (int kb = wave; kb < nb; kb += split) {
    const block_q5_0 *b = wrow + kb;
    const uint32_t w0 = mmq_ld32(b->qs + 8 * h), w1 = mmq_ld32(b->qs + 8 * h + 4);
    const uint32_t qh = mmq_ld32(b->qh);
    // Step 0 holds elements 8h .. 8h + 7 (low nibbles), step 1 elements 16 + 8h .. (high).
    const uint32_t hb0 = qh >> (8 * h), hb1 = qh >> (16 + 8 * h);
    const mmq_v2i b0 = {(int)mmq_sub_bytes((w0 & 0x0F0F0F0Fu) | spread(hb0), 0x10101010u),
                        (int)mmq_sub_bytes((w1 & 0x0F0F0F0Fu) | spread(hb0 >> 4), 0x10101010u)};
    const mmq_v2i b1 = {(int)mmq_sub_bytes(((w0 >> 4) & 0x0F0F0F0Fu) | spread(hb1), 0x10101010u),
                        (int)mmq_sub_bytes(((w1 >> 4) & 0x0F0F0F0Fu) | spread(hb1 >> 4), 0x10101010u)};
    mmq_v2i a0, a1;
    uint32_t ds;
    mmq_load_q8(yrow + kb, h, m_ok, a0, a1, ds);
    mmq_v8i dd = {0, 0, 0, 0, 0, 0, 0, 0};
    dd = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a0, true, b0, dd, false);
    dd = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a1, true, b1, dd, false);
    const float d = __half2float(b->d);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const uint32_t o = __shfl(ds, 8 * h + i, WARP_SIZE);
      const float da = __low2float(*(const half2 *)&o);
      acc[i] = fmaf((float)dd[i], da * d, acc[i]);
    }
  }
  mmq_wmma_finish(acc, dst, n0, nrows_x, ncols_y);
#else
  (void)vx, (void)y, (void)dst, (void)ncols_x, (void)nrows_x, (void)stride_col_y, (void)ncols_y;
#endif
}

#define MMQ_WMMA_Q4K_ENTRY(dst_tag, dst_c_type)                                                   \
  extern "C" __global__ void __launch_bounds__(WARP_SIZE * MMQ_WMMA_MAX_SPLIT)                    \
      mmq_wmma_q4_k_##dst_tag(const void *__restrict__ vx, const void *__restrict__ vy,         \
                              dst_c_type *__restrict__ dst, const int ncols_x, const int nrows_x, \
                              const int stride_col_y, const int ncols_y) {                       \
    mmq_wmma_q4k_impl<dst_c_type>(vx, (const block_q8_1 *)vy, dst, ncols_x, nrows_x,            \
                                  stride_col_y, ncols_y);                                         \
  }
MMQ_WMMA_Q4K_ENTRY(bf16, __nv_bfloat16)
MMQ_WMMA_Q4K_ENTRY(f16, half)
MMQ_WMMA_Q4K_ENTRY(f32, float)

#define MMQ_WMMA_ENTRY(tag, impl, dst_tag, dst_c_type)                                            \
  extern "C" __global__ void __launch_bounds__(WARP_SIZE * MMQ_WMMA_MAX_SPLIT)                    \
      mmq_wmma_##tag##_##dst_tag(const void *__restrict__ vx, const void *__restrict__ vy,        \
                                 dst_c_type *__restrict__ dst, const int ncols_x,                 \
                                 const int nrows_x, const int stride_col_y, const int ncols_y) {  \
    impl<dst_c_type>(vx, (const block_q8_1 *)vy, dst, ncols_x, nrows_x, stride_col_y, ncols_y);  \
  }
#define MMQ_WMMA_SET(tag, impl)                                                                    \
  MMQ_WMMA_ENTRY(tag, impl, bf16, __nv_bfloat16)                                                   \
  MMQ_WMMA_ENTRY(tag, impl, f16, half)                                                             \
  MMQ_WMMA_ENTRY(tag, impl, f32, float)
MMQ_WMMA_SET(q6_k, mmq_wmma_q6k_impl)
MMQ_WMMA_SET(q5_0, mmq_wmma_q5_0_impl)
