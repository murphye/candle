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
                                          const int nrows_x) {
  const int row = blockIdx.x * blockDim.y + threadIdx.y;
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
                                                const int nrows_x, const int stride_col_y) {
  const int row = blockIdx.x * blockDim.y + threadIdx.y;
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

// Q4_K rows against several activation columns: the weight block's nibbles and scales are
// unpacked once per lane and reused for every column, where the generic kernel redoes them
// per column and ran at a third of memory bandwidth on three-row decode batches.
// `nrows_w` weight rows per wave share each activation load.
template <typename dst_t, int ncols_y, int nrows_w>
static __device__ void mmvq_warp_row_multi_q4k_impl(const void *__restrict__ vx,
                                                    const block_q8_1 *__restrict__ y,
                                                    dst_t *__restrict__ dst, const int ncols_x,
                                                    const int nrows_x, const int stride_col_y) {
  const int row0 = (blockIdx.x * blockDim.y + threadIdx.y) * nrows_w;
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
                                                   stride_col_y);                                           \
  }                                                                                                         \
  extern "C" __global__ void mmvq_gguf_q4_k_##dst_tag##_warp_row##n##x2(                                    \
      const void *__restrict__ vx, const void *__restrict__ vy, dst_c_type *__restrict__ dst,               \
      const int ncols_x, const int nrows_x, const int stride_col_y) {                                       \
    mmvq_warp_row_multi_q4k_impl<dst_c_type, n, 2>(vx, (const block_q8_1 *)vy, dst, ncols_x, nrows_x,      \
                                                   stride_col_y);                                           \
  }
#define MMVQ_WARP_ROW_MULTI_Q4K_DST(dst_tag, dst_c_type)                                                  \
  MMVQ_WARP_ROW_MULTI_Q4K_ENTRY(dst_tag, dst_c_type, 2)                                                   \
  MMVQ_WARP_ROW_MULTI_Q4K_ENTRY(dst_tag, dst_c_type, 3)                                                   \
  MMVQ_WARP_ROW_MULTI_Q4K_ENTRY(dst_tag, dst_c_type, 4)                                                   \
  MMVQ_WARP_ROW_MULTI_Q4K_ENTRY(dst_tag, dst_c_type, 5)                                                   \
  MMVQ_WARP_ROW_MULTI_Q4K_ENTRY(dst_tag, dst_c_type, 6)                                                   \
  MMVQ_WARP_ROW_MULTI_Q4K_ENTRY(dst_tag, dst_c_type, 7)                                                   \
  MMVQ_WARP_ROW_MULTI_Q4K_ENTRY(dst_tag, dst_c_type, 8)

#define MMVQ_WARP_ROW_MULTI_ENTRY(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot, dst_tag, dst_c_type, n) \
  extern "C" __global__ void mmvq_gguf_##tag##_##dst_tag##_warp_row##n(                                     \
      const void *__restrict__ vx, const void *__restrict__ vy, dst_c_type *__restrict__ dst,               \
      const int ncols_x, const int nrows_x, const int stride_col_y) {                                       \
    mmvq_warp_row_multi_impl<dst_c_type, n, qk_val, qi_val, block_q_t, vdr_val, vec_dot>(                   \
        vx, (const block_q8_1 *)vy, dst, ncols_x, nrows_x, stride_col_y);                                   \
  }

#define MMVQ_WARP_ROW_MULTI_DST(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot, dst_tag, dst_c_type)    \
  MMVQ_WARP_ROW_MULTI_ENTRY(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot, dst_tag, dst_c_type, 2)     \
  MMVQ_WARP_ROW_MULTI_ENTRY(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot, dst_tag, dst_c_type, 3)     \
  MMVQ_WARP_ROW_MULTI_ENTRY(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot, dst_tag, dst_c_type, 4)     \
  MMVQ_WARP_ROW_MULTI_ENTRY(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot, dst_tag, dst_c_type, 5)     \
  MMVQ_WARP_ROW_MULTI_ENTRY(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot, dst_tag, dst_c_type, 6)     \
  MMVQ_WARP_ROW_MULTI_ENTRY(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot, dst_tag, dst_c_type, 7)     \
  MMVQ_WARP_ROW_MULTI_ENTRY(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot, dst_tag, dst_c_type, 8)

#define MMVQ_WARP_ROW_ENTRY(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot, dst_tag, dst_c_type) \
  extern "C" __global__ void mmvq_gguf_##tag##_##dst_tag##_warp_row(                              \
      const void *__restrict__ vx, const void *__restrict__ vy, dst_c_type *__restrict__ dst,     \
      const int ncols_x, const int nrows_x) {                                                     \
    mmvq_warp_row_impl<dst_c_type, qk_val, qi_val, block_q_t, vdr_val, vec_dot>(                  \
        vx, (const block_q8_1 *)vy, dst, ncols_x, nrows_x);                                       \
  }

#define MMVQ_WARP_ROW_SET(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot)                       \
  MMVQ_WARP_ROW_ENTRY(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot, bf16, __nv_bfloat16)      \
  MMVQ_WARP_ROW_ENTRY(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot, f16, half)                \
  MMVQ_WARP_ROW_ENTRY(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot, f32, float)               \
  MMVQ_WARP_ROW_MULTI_DST(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot, bf16, __nv_bfloat16)  \
  MMVQ_WARP_ROW_MULTI_DST(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot, f16, half)            \
  MMVQ_WARP_ROW_MULTI_DST(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot, f32, float)

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
MMVQ_WARP_ROW_SET(q5_k, block_q5_K, QK_K, QI5_K, VDR_Q5_K_Q8_1_MMVQ, vec_dot_q5_K_q8_1)
MMVQ_WARP_ROW_SET(q6_k, block_q6_K, QK_K, QI6_K, VDR_Q6_K_Q8_1_MMVQ, vec_dot_q6_K_q8_1)
