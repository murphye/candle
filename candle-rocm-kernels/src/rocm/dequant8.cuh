// Eight-weight dequantizers for GGUF quant blocks, shared by the kernels that dequantize
// weights on the fly. Needs the block definitions of quantized.cu.
//
// Two forms: `moe_dequant8_*` loads and converts in one go; `moe_load8_*` + `moe_conv8_*`
// split the global loads from the arithmetic, so a kernel can issue the loads of its next
// tile before it computes on the current one.
#pragma once

// The raw words eight weights of one quant block need, in the type's own packing.
struct moe_raw8 {
    uint32_t w[6];
};

static __device__ __forceinline__ uint32_t moe_pack16(const uint16_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 16);
}

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

static __device__ __forceinline__ void moe_load8_q8_0(const void *vx, int64_t ib, int part, moe_raw8 &r) {
    const uint16_t *b = (const uint16_t *)((const uint8_t *)vx + ib * (int64_t)sizeof(block_q8_0));
    const uint16_t *qs = b + 1 + 4 * part;
    r.w[0] = b[0];
    r.w[1] = moe_pack16(qs);
    r.w[2] = moe_pack16(qs + 2);
}

static __device__ __forceinline__ void moe_conv8_q8_0(const moe_raw8 &r, int part, float *out) {
    const float d = __half2float(__ushort_as_half((uint16_t)r.w[0]));
#pragma unroll
    for (int l = 0; l < 8; ++l) {
        const uint32_t word = l < 4 ? r.w[1] : r.w[2];
        out[l] = d * (int8_t)((word >> (8 * (l % 4))) & 0xFF);
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

static __device__ __forceinline__ void moe_load8_q5_0(const void *vx, int64_t ib, int part, moe_raw8 &r) {
    const uint16_t *b = (const uint16_t *)((const uint8_t *)vx + ib * (int64_t)sizeof(block_q5_0));
    const uint16_t *qs = b + 3 + 4 * (part & 1);
    r.w[0] = b[0];
    r.w[1] = moe_pack16(b + 1);
    r.w[2] = moe_pack16(qs);
    r.w[3] = moe_pack16(qs + 2);
}

static __device__ __forceinline__ void moe_conv8_q5_0(const moe_raw8 &r, int part, float *out) {
    const float d = __half2float(__ushort_as_half((uint16_t)r.w[0]));
    const int shift = part >= 2 ? 4 : 0;
    const uint32_t hbits = r.w[1] >> (8 * part);
#pragma unroll
    for (int l = 0; l < 8; ++l) {
        const uint32_t word = l < 4 ? r.w[2] : r.w[3];
        const int nibble = (word >> (8 * (l % 4) + shift)) & 0x0F;
        const int q = (int)(nibble | (((hbits >> l) & 1) << 4)) - 16;
        out[l] = d * q;
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

static __device__ __forceinline__ void moe_load8_q4_0(const void *vx, int64_t ib, int part, moe_raw8 &r) {
    // 18-byte blocks are 2-byte aligned; parts 0/2 and 1/3 share the same eight quant bytes.
    const uint16_t *b = (const uint16_t *)((const uint8_t *)vx + ib * (int64_t)sizeof(block_q4_0));
    const uint16_t *qs = b + 1 + 4 * (part & 1);
    r.w[0] = b[0];
    r.w[1] = moe_pack16(qs);
    r.w[2] = moe_pack16(qs + 2);
}

static __device__ __forceinline__ void moe_conv8_q4_0(const moe_raw8 &r, int part, float *out) {
    const float d = __half2float(__ushort_as_half((uint16_t)r.w[0]));
    const int shift = part >= 2 ? 4 : 0;
#pragma unroll
    for (int l = 0; l < 8; ++l) {
        const uint32_t word = l < 4 ? r.w[1] : r.w[2];
        const int nibble = (word >> (8 * (l % 4) + shift)) & 0x0F;
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

static __device__ __forceinline__ void moe_load8_q4_K(const void *vx, int64_t ib, int part, moe_raw8 &r) {
    // The 16-byte head (dm and the twelve scale bytes) and the eight quant bytes, two loads.
    // The conversion picks its sub-block's scale bytes out of the head words by a shift, so
    // the per-lane sub-block index never selects a word.
    const uint8_t *blk = (const uint8_t *)vx + ib * (int64_t)sizeof(block_q4_K);
    const int sub = part / 4;
    const uint4 head = *(const uint4 *)blk;
    const uint2 qq = *(const uint2 *)(blk + 16 + 32 * (sub / 2) + 8 * (part % 4));
    r.w[0] = head.x;
    r.w[1] = head.y;
    r.w[2] = head.z;
    r.w[3] = head.w;
    r.w[4] = qq.x;
    r.w[5] = qq.y;
}

// The sub-block's 6-bit scale and min from the three scale words of a Q4_K head
// (get_scale_min_k4). Byte `sub & 3` of each word is shifted into the low byte, so the
// per-lane sub-block index never selects a word.
static __device__ __forceinline__ void moe_q4_K_scale_min(const moe_raw8 &r, int sub, uint32_t &sc, uint32_t &m) {
    const int bsh = 8 * (sub & 3);
    const uint32_t s0 = r.w[1] >> bsh, s1 = r.w[2] >> bsh, s2 = r.w[3] >> bsh;
    sc = sub < 4 ? (s0 & 63) : ((s2 & 0xF) | (((s0 >> 6) & 3) << 4));
    m = sub < 4 ? (s1 & 63) : (((s2 >> 4) & 0xF) | (((s1 >> 6) & 3) << 4));
}

static __device__ __forceinline__ void moe_conv8_q4_K(const moe_raw8 &r, int part, float *out) {
    const int sub = part / 4;
    uint32_t sc, m;
    moe_q4_K_scale_min(r, sub, sc, m);
    const float d = __half2float(__ushort_as_half((uint16_t)r.w[0])) * sc;
    const float dmin = __half2float(__ushort_as_half((uint16_t)(r.w[0] >> 16))) * m;
    // One shift and mask per quant word leaves each weight in its own byte, which the
    // byte-to-float conversions take directly.
    const int shift = 4 * (sub & 1);
    const uint32_t q0 = (r.w[4] >> shift) & 0x0F0F0F0Fu;
    const uint32_t q1 = (r.w[5] >> shift) & 0x0F0F0F0Fu;
#pragma unroll
    for (int l = 0; l < 8; ++l) {
        const uint32_t word = l < 4 ? q0 : q1;
        out[l] = d * (float)((word >> (8 * (l % 4))) & 0xFFu) - dmin;
    }
}

// Q3_K: 256 weights in 110-byte blocks (2-byte aligned): hmask[32] (high bits), qs[64] (two
// low bits), scales[12] (sixteen 6-bit scales), d. Weight w: n = w / 128, j = (w % 128) / 32,
// l = w % 32; q = (qs[32 n + l] >> 2 j) & 3 minus 4 unless hmask[l] bit (4 n + j); scale
// index 8 n + 2 j + (l >= 16). The eight weights of a part share n, j and the scale.
static __device__ __forceinline__ void moe_load8_q3_K(const void *vx, int64_t ib, int part, moe_raw8 &r) {
    const uint8_t *blk = (const uint8_t *)vx + ib * (int64_t)sizeof(block_q3_K);
    const int n = part / 16;
    const int j = (part % 16) / 4;
    const int l0 = 8 * (part % 4);
    const int is = 8 * n + 2 * j + (l0 >= 16 ? 1 : 0);
    const uint16_t *hm = (const uint16_t *)(blk + l0);
    const uint16_t *qs = (const uint16_t *)(blk + 32 + 32 * n + l0);
    const uint8_t *scales = blk + 96;
    // The two scale bytes the 6-bit scale `is` is spread over (dequantize_block_q3_K).
    const int a = is < 8 ? is : is - 8;
    const int b = is < 4 ? is + 8 : is < 8 ? is + 4 : is < 12 ? is : is - 4;
    r.w[0] = *(const uint16_t *)(blk + 108);
    r.w[1] = moe_pack16(qs);
    r.w[2] = moe_pack16(qs + 2);
    r.w[3] = moe_pack16(hm);
    r.w[4] = moe_pack16(hm + 2);
    r.w[5] = (uint32_t)scales[a] | ((uint32_t)scales[b] << 8);
}

static __device__ __forceinline__ void moe_conv8_q3_K(const moe_raw8 &r, int part, float *out) {
    const int n = part / 16;
    const int j = (part % 16) / 4;
    const int l0 = 8 * (part % 4);
    const int is = 8 * n + 2 * j + (l0 >= 16 ? 1 : 0);
    const uint32_t sa = r.w[5] & 0xFF, sb = r.w[5] >> 8;
    const uint32_t us = is < 4    ? (sa & 0xF) | (((sb >> 0) & 3) << 4)
                        : is < 8  ? (sa & 0xF) | (((sb >> 2) & 3) << 4)
                        : is < 12 ? (sa >> 4) | (((sb >> 4) & 3) << 4)
                                  : (sa >> 4) | (((sb >> 6) & 3) << 4);
    const float dl = __half2float(__ushort_as_half((uint16_t)r.w[0])) * ((float)us - 32.0f);
    const int shift = 2 * j;
    const uint32_t m = 1u << (4 * n + j);
#pragma unroll
    for (int l = 0; l < 8; ++l) {
        const uint32_t qword = l < 4 ? r.w[1] : r.w[2];
        const uint32_t hword = l < 4 ? r.w[3] : r.w[4];
        const int q = (int)((qword >> (8 * (l % 4) + shift)) & 3);
        const int h = (int)((hword >> (8 * (l % 4))) & 0xFF);
        out[l] = dl * (float)(q - ((h & m) ? 0 : 4));
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

static __device__ __forceinline__ void moe_load8_q6_K(const void *vx, int64_t ib, int part, moe_raw8 &r) {
    const block_q6_K *x = (const block_q6_K *)vx + ib;
    const int j = 8 * part;
    const int half = j / 128;
    const int jj = j % 128;
    const int quarter = jj / 32;
    const int lane0 = jj % 32;
    const uint16_t *ql = (const uint16_t *)(x->ql + 64 * half + lane0 + 32 * (quarter & 1));
    const uint16_t *qh = (const uint16_t *)(x->qh + 32 * half + lane0);
    const uint32_t scale = (uint32_t)(uint8_t)x->scales[8 * half + 2 * quarter + lane0 / 16];
    r.w[0] = (uint32_t)__half_as_ushort(x->d) | (scale << 16);
    r.w[1] = moe_pack16(ql);
    r.w[2] = moe_pack16(ql + 2);
    r.w[3] = moe_pack16(qh);
    r.w[4] = moe_pack16(qh + 2);
}

static __device__ __forceinline__ void moe_conv8_q6_K(const moe_raw8 &r, int part, float *out) {
    const int quarter = (part % 16) / 4;
    const float d = __half2float(__ushort_as_half((uint16_t)r.w[0])) * (float)(int8_t)(r.w[0] >> 16);
    const int lshift = quarter >= 2 ? 4 : 0;
    const int hshift = 2 * quarter;
#pragma unroll
    for (int l = 0; l < 8; ++l) {
        const uint32_t l2 = (l < 4 ? r.w[1] : r.w[2]) >> (16 * ((l / 2) % 2));
        const uint32_t h2 = (l < 4 ? r.w[3] : r.w[4]) >> (16 * ((l / 2) % 2));
        const int b = l % 2;
        const int low = (l2 >> (8 * b + lshift)) & 0x0F;
        const int high = (h2 >> (8 * b + hshift)) & 3;
        out[l] = d * ((low | (high << 4)) - 32);
    }
}
