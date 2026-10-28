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
