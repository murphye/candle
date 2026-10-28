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
  MMVQ_WARP_ROW_ENTRY(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot, f32, float)

MMVQ_WARP_ROW_SET(q4_0, block_q4_0, QK4_0, QI4_0, VDR_Q4_0_Q8_1_MMVQ, vec_dot_q4_0_q8_1)
MMVQ_WARP_ROW_SET(q4_1, block_q4_1, QK4_1, QI4_1, VDR_Q4_1_Q8_1_MMVQ, vec_dot_q4_1_q8_1)
MMVQ_WARP_ROW_SET(q5_0, block_q5_0, QK5_0, QI5_0, VDR_Q5_0_Q8_1_MMVQ, vec_dot_q5_0_q8_1)
MMVQ_WARP_ROW_SET(q5_1, block_q5_1, QK5_1, QI5_1, VDR_Q5_1_Q8_1_MMVQ, vec_dot_q5_1_q8_1)
MMVQ_WARP_ROW_SET(q8_0, block_q8_0, QK8_0, QI8_0, VDR_Q8_0_Q8_1_MMVQ, vec_dot_q8_0_q8_1)
MMVQ_WARP_ROW_SET(q2_k, block_q2_K, QK_K, QI2_K, VDR_Q2_K_Q8_1_MMVQ, vec_dot_q2_K_q8_1)
MMVQ_WARP_ROW_SET(q3_k, block_q3_K, QK_K, QI3_K, VDR_Q3_K_Q8_1_MMVQ, vec_dot_q3_K_q8_1)
MMVQ_WARP_ROW_SET(q4_k, block_q4_K, QK_K, QI4_K, VDR_Q4_K_Q8_1_MMVQ, vec_dot_q4_K_q8_1)
MMVQ_WARP_ROW_SET(q5_k, block_q5_K, QK_K, QI5_K, VDR_Q5_K_Q8_1_MMVQ, vec_dot_q5_K_q8_1)
MMVQ_WARP_ROW_SET(q6_k, block_q6_K, QK_K, QI6_K, VDR_Q6_K_Q8_1_MMVQ, vec_dot_q6_K_q8_1)
