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
// trip. Same work split as `indexed_moe_forward`: one output row per block.
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
    const int row0 = blockIdx.x;
    if (row0 >= n) {
        return;
    }

    constexpr int nwarps = 4;
    const size_t weight_expert_stride_bytes = (size_t)n * k / qk * sizeof(block_q_t);
    const size_t input_task_stride_bytes = (size_t)k_padded / QK8_1 * sizeof(block_q8_1);
    const block_q_t * w = (const block_q_t *)((const char *)all_weights + expert_id * weight_expert_stride_bytes);
    const block_q8_1 * x = (const block_q8_1 *)((const char *)all_inputs + input_idx * input_task_stride_bytes);

    const int tid = WARP_SIZE * threadIdx.y + threadIdx.x;
    const int blocks_per_row_x = k / qk;
    constexpr int blocks_per_iter = vdr * nwarps * WARP_SIZE / qi;

    float tmp = 0.0f;
    for (int kbx = tid / (qi / vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk / QK8_1);
        const int kqs = vdr * (tid % (qi / vdr));
        tmp += vec_dot_q_cuda(&w[kbx + row0 * blocks_per_row_x], &x[kby], kqs);
    }

    __shared__ float tmp_shared[nwarps - 1][WARP_SIZE];
    if (threadIdx.y > 0) {
        tmp_shared[threadIdx.y - 1][threadIdx.x] = tmp;
    }
    __syncthreads();
    if (threadIdx.y == 0) {
        for (int l = 0; l < nwarps - 1; ++l) {
            tmp += tmp_shared[l][threadIdx.x];
        }
        tmp = warp_reduce_sum(tmp);
        if (threadIdx.x == 0) {
            all_outputs[(size_t)task_id * n + row0] = (dst_t)tmp;
        }
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
