// Eight-weight dequantizers for GGUF quant blocks, shared by the kernels that dequantize
// weights on the fly. Needs the block definitions of quantized.cu.
#pragma once

// Eight consecutive weights `8 * part .. 8 * part + 7` of quant block `ib`.
static __device__ __forceinline__ void moe_dequant8_q8_0(const void *vx, int64_t ib, int part, float *out) {
    // 34-byte blocks are only 2-byte aligned, so the quants are read in 16-bit words.
    const uint16_t *b = (const uint16_t *)((const uint8_t *)vx + ib * (int64_t)sizeof(block_q8_0));
    const float d = __half2float(__ushort_as_half(b[0]));
    const uint16_t *qs = b + 1 + 4 * part;
#pragma unroll
    for (int w = 0; w < 4; ++w) {
        const uint32_t pair = qs[w];
        out[2 * w] = d * (int8_t)(pair & 0xFF);
        out[2 * w + 1] = d * (int8_t)(pair >> 8);
    }
}

static __device__ __forceinline__ void moe_dequant8_q5_0(const void *vx, int64_t ib, int part, float *out) {
    // 22-byte blocks are only 2-byte aligned, so fields are read in 16-bit words.
    const uint16_t *b = (const uint16_t *)((const uint8_t *)vx + ib * (int64_t)sizeof(block_q5_0));
    const float d = __half2float(__ushort_as_half(b[0]));
    const uint32_t qh = (uint32_t)b[1] | ((uint32_t)b[2] << 16);
    // Weights 0-15 are the low nibbles of qs[0..16], 16-31 the high nibbles.
    const uint16_t *qs = b + 3 + 4 * (part & 1);
    const int shift = part >= 2 ? 4 : 0;
    const uint32_t hbits = qh >> (8 * part);
#pragma unroll
    for (int w = 0; w < 4; ++w) {
        const uint32_t pair = qs[w];
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int l = 2 * w + h;
            const int nibble = (pair >> (8 * h + shift)) & 0x0F;
            const int q = (int)(nibble | (((hbits >> l) & 1) << 4)) - 16;
            out[l] = d * q;
        }
    }
}

static __device__ __forceinline__ void moe_dequant8_q4_0(const void *vx, int64_t ib, int part, float *out) {
    const block_q4_0 *x = (const block_q4_0 *)vx + ib;
    const float d = __half2float(x->d);
#pragma unroll
    for (int l = 0; l < 8; ++l) {
        const int j = 8 * part + l;
        const int nibble = j < 16 ? (x->qs[j] & 0x0F) : (x->qs[j - 16] >> 4);
        out[l] = d * (nibble - 8);
    }
}

// Q4_K super-blocks hold 256 weights in eight 32-weight sub-blocks; `part` is 0..31.
static __device__ __forceinline__ void moe_dequant8_q4_K(const void *vx, int64_t ib, int part, float *out) {
    const block_q4_K *x = (const block_q4_K *)vx + ib;
    const int sub = part / 4;
    uint8_t sc, m;
    get_scale_min_k4(sub, x->scales, sc, m);
    const float d = __low2float(x->dm) * sc;
    const float dmin = __high2float(x->dm) * m;
    // Eight quant bytes in one 8-byte load: super-blocks are 144 bytes, so this is aligned.
    const uint2 qq = *(const uint2 *)(x->qs + 32 * (sub / 2) + 8 * (part % 4));
    const int shift = 4 * (sub & 1);
#pragma unroll
    for (int l = 0; l < 8; ++l) {
        const uint32_t word = l < 4 ? qq.x : qq.y;
        out[l] = d * ((word >> (8 * (l % 4) + shift)) & 0x0F) - dmin;
    }
}

// Q6_K: 256 weights; ql holds the low four bits, qh the high two, 16 scales of 16 weights.
// The eight weights of a part share one ql run, one qh run and one scale, read as 16-bit words
// (210-byte blocks are only 2-byte aligned).
static __device__ __forceinline__ void moe_dequant8_q6_K(const void *vx, int64_t ib, int part, float *out) {
    const block_q6_K *x = (const block_q6_K *)vx + ib;
    const int j = 8 * part;          // 0..248, a multiple of 8
    const int half = j / 128;
    const int jj = j % 128;
    const int quarter = jj / 32;
    const int lane0 = jj % 32;
    const uint16_t *ql = (const uint16_t *)(x->ql + 64 * half + lane0 + 32 * (quarter & 1));
    const uint16_t *qh = (const uint16_t *)(x->qh + 32 * half + lane0);
    const float d = __half2float(x->d) * x->scales[8 * half + 2 * quarter + lane0 / 16];
    const int lshift = quarter >= 2 ? 4 : 0;
    const int hshift = 2 * quarter;
#pragma unroll
    for (int w = 0; w < 4; ++w) {
        const uint32_t l2 = ql[w], h2 = qh[w];
#pragma unroll
        for (int b = 0; b < 2; ++b) {
            const int low = (l2 >> (8 * b + lshift)) & 0x0F;
            const int high = (h2 >> (8 * b + hshift)) & 3;
            out[2 * w + b] = d * ((low | (high << 4)) - 32);
        }
    }
}

