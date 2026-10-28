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

// Indexed MoE for the legacy 32-element block types, which some GGUF MoE files
// use for their down projections.
#define INDEXED_MOE_LEGACY(QNAME, QK, QI, VDR) \
extern "C" __global__ void indexed_moe_forward_##QNAME##_q8_1( \
    const void * __restrict__ all_weights, \
    const void * __restrict__ all_inputs, \
    const unsigned int * __restrict__ indices, \
    float * __restrict__ all_outputs, \
    const int n, \
    const int k, \
    const int batch, \
    const int topk, \
    const int k_padded, \
    const int input_dim1) { \
    indexed_moe_forward<QK, QI, block_##QNAME, VDR, vec_dot_##QNAME##_q8_1> \
        (all_weights, all_inputs, indices, all_outputs, n, k, batch, topk, k_padded, input_dim1); \
}

INDEXED_MOE_LEGACY(q4_0, QK4_0, QI4_0, VDR_Q4_0_Q8_1_MMVQ)
INDEXED_MOE_LEGACY(q4_1, QK4_1, QI4_1, VDR_Q4_1_Q8_1_MMVQ)
INDEXED_MOE_LEGACY(q5_0, QK5_0, QI5_0, VDR_Q5_0_Q8_1_MMVQ)
INDEXED_MOE_LEGACY(q5_1, QK5_1, QI5_1, VDR_Q5_1_Q8_1_MMVQ)
