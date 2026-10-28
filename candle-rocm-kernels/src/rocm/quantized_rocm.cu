// The quantized kernels from candle-kernels, plus bf16 outputs for the
// dequantize kernels. The ROCm backend dequantizes weights to bf16 for large
// batches and runs the matmul with rocBLAS, which is much faster there than
// the dp4a based MMQ kernels.
#include "quantized.cu"

#define DEQUANTIZE_K_BF16(QNAME) \
extern "C" __global__ void dequantize_block_##QNAME##_bf16(const void * __restrict__ vx, __nv_bfloat16 * __restrict__ y) { \
  dequantize_block_##QNAME(vx, y); \
}

#define DEQUANTIZE_BF16(QNAME) \
extern "C" __global__ void dequantize_block_##QNAME##_bf16(const void * __restrict__ vx, __nv_bfloat16 * __restrict__ y, const int k) { \
  dequantize_block_##QNAME(vx, y, k); \
}

DEQUANTIZE_K_BF16(q2_K)
DEQUANTIZE_K_BF16(q3_K)
DEQUANTIZE_K_BF16(q4_K)
DEQUANTIZE_K_BF16(q5_K)
DEQUANTIZE_K_BF16(q6_K)
DEQUANTIZE_K_BF16(q8_K)
DEQUANTIZE_BF16(q4_0)
DEQUANTIZE_BF16(q4_1)
DEQUANTIZE_BF16(q5_0)
DEQUANTIZE_BF16(q5_1)
DEQUANTIZE_BF16(q8_0)

// Indexed MoE writing the activation dtype directly. The inputs are q8_1 rows
// from the MMVQ quantizer, so half precision activations need no f32 round
// trip. Each warp computes one output row; a block covers `nwarps` rows of
// one (token, expert) pair, which keeps short rows from idling most threads.
template <int qk, int qi, typename block_q_t, int vdr, vec_dot_q_cuda_t vec_dot_q_cuda, typename dst_t>
__device__ void indexed_moe_dst(
    const void * __restrict__ all_weights,
    const void * __restrict__ all_inputs,
    const unsigned int * __restrict__ indices,
    dst_t * __restrict__ all_outputs,
    const int n,
    const int k,
    const int k_padded,
    const int input_dim1) {
    const int task_id = blockIdx.y * gridDim.z + blockIdx.z;
    const int input_idx = (input_dim1 == 1) ? blockIdx.y : task_id;
    const unsigned int expert_id = indices[task_id];
    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= n) {
        return;
    }

    const size_t weight_expert_stride_bytes = (size_t)n * k / qk * sizeof(block_q_t);
    const size_t input_task_stride_bytes = (size_t)k_padded / QK8_1 * sizeof(block_q8_1);
    const block_q_t * w = (const block_q_t *)((const char *)all_weights + expert_id * weight_expert_stride_bytes);
    const block_q8_1 * x = (const block_q8_1 *)((const char *)all_inputs + input_idx * input_task_stride_bytes);

    const int blocks_per_row_x = k / qk;
    constexpr int blocks_per_iter = vdr * WARP_SIZE / qi;

    float tmp = 0.0f;
    for (int kbx = threadIdx.x / (qi / vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk / QK8_1);
        const int kqs = vdr * (threadIdx.x % (qi / vdr));
        tmp += vec_dot_q_cuda(&w[kbx + row * blocks_per_row_x], &x[kby], kqs);
    }
    tmp = warp_reduce_sum(tmp);
    if (threadIdx.x == 0) {
        all_outputs[(size_t)task_id * n + row] = (dst_t)tmp;
    }
}

#define INDEXED_MOE_DST(QNAME, QK, QI, BLOCK, VDR, VEC_DOT, DST, DST_T) \
extern "C" __global__ void indexed_moe_##QNAME##_##DST( \
    const void * __restrict__ all_weights, \
    const void * __restrict__ all_inputs, \
    const unsigned int * __restrict__ indices, \
    DST_T * __restrict__ all_outputs, \
    const int n, \
    const int k, \
    const int k_padded, \
    const int input_dim1) { \
    indexed_moe_dst<QK, QI, BLOCK, VDR, VEC_DOT, DST_T> \
        (all_weights, all_inputs, indices, all_outputs, n, k, k_padded, input_dim1); \
}

#define INDEXED_MOE_ALL_DST(QNAME, QK, QI, BLOCK, VDR, VEC_DOT) \
INDEXED_MOE_DST(QNAME, QK, QI, BLOCK, VDR, VEC_DOT, f32, float) \
INDEXED_MOE_DST(QNAME, QK, QI, BLOCK, VDR, VEC_DOT, f16, __half) \
INDEXED_MOE_DST(QNAME, QK, QI, BLOCK, VDR, VEC_DOT, bf16, __nv_bfloat16)

INDEXED_MOE_ALL_DST(q4_0, QK4_0, QI4_0, block_q4_0, VDR_Q4_0_Q8_1_MMVQ, vec_dot_q4_0_q8_1)
INDEXED_MOE_ALL_DST(q4_1, QK4_1, QI4_1, block_q4_1, VDR_Q4_1_Q8_1_MMVQ, vec_dot_q4_1_q8_1)
INDEXED_MOE_ALL_DST(q5_0, QK5_0, QI5_0, block_q5_0, VDR_Q5_0_Q8_1_MMVQ, vec_dot_q5_0_q8_1)
INDEXED_MOE_ALL_DST(q5_1, QK5_1, QI5_1, block_q5_1, VDR_Q5_1_Q8_1_MMVQ, vec_dot_q5_1_q8_1)
INDEXED_MOE_ALL_DST(q8_0, QK8_0, QI8_0, block_q8_0, VDR_Q8_0_Q8_1_MMVQ, vec_dot_q8_0_q8_1)
INDEXED_MOE_ALL_DST(q2k, QK_K, QI2_K, block_q2_K, VDR_Q2_K_Q8_1_MMVQ, vec_dot_q2_K_q8_1)
INDEXED_MOE_ALL_DST(q3k, QK_K, QI3_K, block_q3_K, VDR_Q3_K_Q8_1_MMVQ, vec_dot_q3_K_q8_1)
INDEXED_MOE_ALL_DST(q4k, QK_K, QI4_K, block_q4_K, VDR_Q4_K_Q8_1_MMVQ, vec_dot_q4_K_q8_1)
INDEXED_MOE_ALL_DST(q5k, QK_K, QI5_K, block_q5_K, VDR_Q5_K_Q8_1_MMVQ, vec_dot_q5_K_q8_1)
INDEXED_MOE_ALL_DST(q6k, QK_K, QI6_K, block_q6_K, VDR_Q6_K_Q8_1_MMVQ, vec_dot_q6_K_q8_1)

// Dequantization of the 32-element block types to bf16 with one thread per
// eight outputs and a single 16-byte store, for the grouped MoE prefill path.
// The generic kernels write two scattered values per thread.
static __device__ __forceinline__ void dequant8_q8_0(const void *vx, int64_t ib, int part, float *out) {
    const block_q8_0 *x = (const block_q8_0 *)vx + ib;
    const float d = __half2float(x->d);
#pragma unroll
    for (int l = 0; l < 8; ++l) {
        out[l] = d * x->qs[8 * part + l];
    }
}

static __device__ __forceinline__ void dequant8_q5_0(const void *vx, int64_t ib, int part, float *out) {
    const block_q5_0 *x = (const block_q5_0 *)vx + ib;
    const float d = __half2float(x->d);
    uint32_t qh;
    memcpy(&qh, x->qh, sizeof(qh));
#pragma unroll
    for (int l = 0; l < 8; ++l) {
        const int j = 8 * part + l;
        const int nibble = j < 16 ? (x->qs[j] & 0x0F) : (x->qs[j - 16] >> 4);
        const int q = (nibble | (((qh >> j) & 1) << 4)) - 16;
        out[l] = d * q;
    }
}

template <void (*dequant8)(const void *, int64_t, int, float *)>
static __device__ void dequantize_v8_bf16(const void *__restrict__ vx, __nv_bfloat16 *__restrict__ y,
                                          const int64_t n_groups) {
    const int64_t g = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= n_groups) {
        return;
    }
    float v[8];
    dequant8(vx, g / 4, g % 4, v);
    __align__(16) __nv_bfloat16 packed[8];
#pragma unroll
    for (int l = 0; l < 8; ++l) {
        packed[l] = __float2bfloat16(v[l]);
    }
    *(uint4 *)(y + 8 * g) = *(const uint4 *)packed;
}

extern "C" __global__ void dequantize_v8_q8_0_bf16(const void *__restrict__ vx, __nv_bfloat16 *__restrict__ y,
                                                   const int64_t n_groups) {
    dequantize_v8_bf16<dequant8_q8_0>(vx, y, n_groups);
}

extern "C" __global__ void dequantize_v8_q5_0_bf16(const void *__restrict__ vx, __nv_bfloat16 *__restrict__ y,
                                                   const int64_t n_groups) {
    dequantize_v8_bf16<dequant8_q5_0>(vx, y, n_groups);
}
