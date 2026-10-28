// Grouped MoE GEMM on quantized expert weights for prefill-sized batches.
//
// `moe_route` sorts the (token, slot) pairs by expert on the device and cuts each
// expert's rows into BM-row tiles; `moe_gemm_*` then computes one BM x BN output
// tile per workgroup: it dequantizes the expert's weight slice into LDS as bf16 and
// multiplies with WMMA (gfx11 / gfx12), accumulating in f32. Nothing is padded,
// no dequantized copy of the experts is written to memory and the host never waits.
#include "quantized.cu"

#include <rocwmma/rocwmma.hpp>


#define MOE_BM 32
#define MOE_BN 128
#define MOE_BK 64
#define MOE_THREADS 256
// Rows of the LDS tiles are padded by 8 bf16 to spread banks.
#define MOE_LDS_K (MOE_BK + 8)
#define MOE_ROUTE_THREADS 1024
#define MOE_MAX_EXPERTS 1024

#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || \
    defined(__gfx1150__) || defined(__gfx1151__) || defined(__gfx1200__) || defined(__gfx1201__)
#define MOE_HAS_WMMA 1
#else
#define MOE_HAS_WMMA 0
#endif

// Sorts `n_pairs` expert ids by expert. Writes, for every expert, the start of its
// rows in `sorted` (`offsets[n_experts + 1]`), the pair index of every sorted row,
// and one (expert, first row) entry per BM-row tile in `tiles`, with the tile count
// in `n_tiles`. One workgroup; the order of pairs within an expert is unspecified.
extern "C" __global__ void moe_route(const uint32_t *__restrict__ ids, const int n_pairs, const int n_experts,
                                     uint32_t *__restrict__ sorted, uint32_t *__restrict__ offsets,
                                     int2 *__restrict__ tiles, uint32_t *__restrict__ n_tiles) {
    __shared__ uint32_t counts[MOE_MAX_EXPERTS];
    __shared__ uint32_t starts[MOE_MAX_EXPERTS];
    __shared__ uint32_t tile_starts[MOE_MAX_EXPERTS];
    const int tid = threadIdx.x;
    for (int e = tid; e < n_experts; e += blockDim.x) {
        counts[e] = 0;
    }
    __syncthreads();
    for (int p = tid; p < n_pairs; p += blockDim.x) {
        atomicAdd(&counts[ids[p]], 1u);
    }
    __syncthreads();
    if (tid == 0) {
        uint32_t row = 0, tile = 0;
        for (int e = 0; e < n_experts; ++e) {
            starts[e] = row;
            tile_starts[e] = tile;
            offsets[e] = row;
            row += counts[e];
            tile += (counts[e] + MOE_BM - 1) / MOE_BM;
        }
        offsets[n_experts] = row;
        *n_tiles = tile;
    }
    __syncthreads();
    for (int e = tid; e < n_experts; e += blockDim.x) {
        const uint32_t ntiles = (counts[e] + MOE_BM - 1) / MOE_BM;
        for (uint32_t t = 0; t < ntiles; ++t) {
            tiles[tile_starts[e] + t] = make_int2(e, starts[e] + t * MOE_BM);
        }
    }
    __syncthreads();
    for (int p = tid; p < n_pairs; p += blockDim.x) {
        const uint32_t at = atomicAdd(&starts[ids[p]], 1u);
        sorted[at] = p;
    }
}

// Eight consecutive weights `8 * part .. 8 * part + 7` of quant block `ib`.
static __device__ __forceinline__ void moe_dequant8_q8_0(const void *vx, int64_t ib, int part, float *out) {
    const block_q8_0 *x = (const block_q8_0 *)vx + ib;
    const float d = __half2float(x->d);
#pragma unroll
    for (int l = 0; l < 8; ++l) {
        out[l] = d * x->qs[8 * part + l];
    }
}

static __device__ __forceinline__ void moe_dequant8_q5_0(const void *vx, int64_t ib, int part, float *out) {
    const block_q5_0 *x = (const block_q5_0 *)vx + ib;
    const float d = __half2float(x->d);
    uint32_t qh;
    memcpy(&qh, x->qh, sizeof(qh));
#pragma unroll
    for (int l = 0; l < 8; ++l) {
        const int j = 8 * part + l;
        const int nibble = j < 16 ? (x->qs[j] & 0x0F) : (x->qs[j - 16] >> 4);
        // In int: with qh unsigned the subtraction would wrap.
        const int q = (nibble | (((qh >> j) & 1) << 4)) - 16;
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
static __device__ __forceinline__ void moe_dequant8_q6_K(const void *vx, int64_t ib, int part, float *out) {
    const block_q6_K *x = (const block_q6_K *)vx + ib;
    const float d = __half2float(x->d);
#pragma unroll
    for (int l = 0; l < 8; ++l) {
        const int j = 8 * part + l;      // 0..255
        const int half = j / 128;        // which 128-weight half
        const int jj = j % 128;
        const int quarter = jj / 32;     // ql / qh selector within the half
        const int lane = jj % 32;
        const uint8_t *ql = x->ql + 64 * half;
        const uint8_t *qh = x->qh + 32 * half;
        // Quarters 0 and 1 take the low nibbles of ql[lane] and ql[lane + 32], 2 and 3 the high ones.
        const uint8_t packed = ql[lane + 32 * (quarter & 1)];
        const int low = quarter >= 2 ? (packed >> 4) : (packed & 0x0F);
        const int high = (qh[lane] >> (2 * quarter)) & 3;
        const int sc = x->scales[8 * half + 2 * quarter + lane / 16];
        out[l] = d * sc * ((low | (high << 4)) - 32);
    }
}

template <typename T> static __device__ __forceinline__ float moe_to_float(T v) { return static_cast<float>(v); }
template <> __device__ __forceinline__ float moe_to_float<__half>(__half v) { return __half2float(v); }
template <> __device__ __forceinline__ float moe_to_float<__nv_bfloat16>(__nv_bfloat16 v) {
    return __bfloat162float(v);
}
template <typename T> static __device__ __forceinline__ T moe_from_float(float v) { return static_cast<T>(v); }
template <> __device__ __forceinline__ __half moe_from_float<__half>(float v) { return __float2half(v); }
template <> __device__ __forceinline__ __nv_bfloat16 moe_from_float<__nv_bfloat16>(float v) {
    return __float2bfloat16(v);
}

// y[pair, n] = x[row(pair), :] . w[expert(pair), n, :] for every routed pair.
// `x_per_pair` selects x's row: the pair itself ([pairs, k], down projections) or its
// token (pair / topk, [tokens, k], gate and up). `qk` weights per quant block,
// `parts` = qk / 8.
template <typename T, int QK, void (*dequant8)(const void *, int64_t, int, float *)>
static __device__ void moe_gemm(const void *__restrict__ w, const T *__restrict__ x, T *__restrict__ y,
                                const uint32_t *__restrict__ sorted, const uint32_t *__restrict__ offsets,
                                const int2 *__restrict__ tiles, const uint32_t *__restrict__ n_tiles,
                                const int n, const int k, const int topk, const int x_per_pair,
                                const int64_t block_bytes) {
#if MOE_HAS_WMMA
    using namespace rocwmma;
    const int tile = blockIdx.y;
    if (tile >= (int)*n_tiles) {
        return;
    }
    const int2 info = tiles[tile];
    const int expert = info.x;
    const int row0 = info.y;
    const int rows = min(MOE_BM, (int)offsets[expert + 1] - row0);
    const int n0 = blockIdx.x * MOE_BN;
    const int tid = threadIdx.x;
    const int wave = tid / 32;

    // The output tile reuses the operand tiles' LDS once the K loop is done.
    constexpr int OPERAND_BYTES = (MOE_BM + MOE_BN) * MOE_LDS_K * sizeof(bfloat16_t);
    constexpr int OUTPUT_BYTES = MOE_BM * MOE_BN * sizeof(float);
    __shared__ __align__(16) char lds[OPERAND_BYTES > OUTPUT_BYTES ? OPERAND_BYTES : OUTPUT_BYTES];
    bfloat16_t *a_lds = (bfloat16_t *)lds;
    bfloat16_t *b_lds = a_lds + MOE_BM * MOE_LDS_K;
    float *c_lds = (float *)lds;
    __shared__ uint32_t pair_of[MOE_BM];

    if (tid < MOE_BM) {
        pair_of[tid] = tid < rows ? sorted[row0 + tid] : 0;
    }
    __syncthreads();

    // Each wave owns a 32 x 16 column strip: two 16 x 16 accumulators.
    fragment<accumulator, 16, 16, 16, float32_t> acc[2];
    fill_fragment(acc[0], 0.0f);
    fill_fragment(acc[1], 0.0f);

    const int blocks_per_row = k / QK;
    const char *wexp = (const char *)w + (int64_t)expert * n * blocks_per_row * block_bytes;

    for (int k0 = 0; k0 < k; k0 += MOE_BK) {
        // A: BM rows x BK in groups of eight; bf16 rows are copied with one 16-byte load.
        for (int g = tid; g < MOE_BM * (MOE_BK / 8); g += MOE_THREADS) {
            const int r = g / (MOE_BK / 8);
            const int c = (g % (MOE_BK / 8)) * 8;
            uint4 packed = make_uint4(0, 0, 0, 0);
            if (r < rows) {
                const uint32_t pair = pair_of[r];
                const int64_t xrow = x_per_pair ? pair : pair / topk;
                const T *src = x + xrow * k + k0 + c;
                if constexpr (sizeof(T) == 2 && std::is_same<T, __nv_bfloat16>::value) {
                    packed = *(const uint4 *)src;
                } else {
                    __align__(16) bfloat16_t v[8];
#pragma unroll
                    for (int l = 0; l < 8; ++l) {
                        v[l] = static_cast<bfloat16_t>(moe_to_float(src[l]));
                    }
                    packed = *(const uint4 *)v;
                }
            }
            *(uint4 *)(a_lds + r * MOE_LDS_K + c) = packed;
        }
        // B: BN weight rows x BK, two 8-weight groups per thread.
#pragma unroll
        for (int g = tid; g < MOE_BN * (MOE_BK / 8); g += MOE_THREADS) {
            const int nn = g / (MOE_BK / 8);
            const int c = (g % (MOE_BK / 8)) * 8;
            __align__(16) bfloat16_t v[8];
            if (n0 + nn < n) {
                const int kk = k0 + c;
                const int64_t ib = (int64_t)(n0 + nn) * blocks_per_row + kk / QK;
                float f[8];
                dequant8(wexp, ib, (kk % QK) / 8, f);
#pragma unroll
                for (int l = 0; l < 8; ++l) {
                    v[l] = static_cast<bfloat16_t>(f[l]);
                }
            } else {
#pragma unroll
                for (int l = 0; l < 8; ++l) {
                    v[l] = static_cast<bfloat16_t>(0.0f);
                }
            }
            *(uint4 *)(b_lds + nn * MOE_LDS_K + c) = *(const uint4 *)v;
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < MOE_BK; kk += 16) {
            fragment<matrix_b, 16, 16, 16, bfloat16_t, col_major> b;
            load_matrix_sync(b, b_lds + (wave * 16) * MOE_LDS_K + kk, MOE_LDS_K);
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                fragment<matrix_a, 16, 16, 16, bfloat16_t, row_major> a;
                load_matrix_sync(a, a_lds + (i * 16) * MOE_LDS_K + kk, MOE_LDS_K);
                mma_sync(acc[i], a, b, acc[i]);
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        store_matrix_sync(c_lds + (i * 16) * MOE_BN + wave * 16, acc[i], MOE_BN, mem_row_major);
    }
    __syncthreads();
    for (int e = tid; e < MOE_BM * MOE_BN; e += MOE_THREADS) {
        const int r = e / MOE_BN;
        const int c = e % MOE_BN;
        if (r < rows && n0 + c < n) {
            y[(int64_t)pair_of[r] * n + n0 + c] = moe_from_float<T>(c_lds[e]);
        }
    }
#endif
}

#define MOE_GEMM(TAG, QK, BLOCK, DEQ, T, DST)                                                              \
    extern "C" __global__ void moe_gemm_##TAG##_##DST(                                                     \
        const void *w, const T *x, T *y, const uint32_t *sorted, const uint32_t *offsets,                 \
        const int2 *tiles, const uint32_t *n_tiles, const int n, const int k, const int topk,             \
        const int x_per_pair) {                                                                            \
        moe_gemm<T, QK, DEQ>(w, x, y, sorted, offsets, tiles, n_tiles, n, k, topk, x_per_pair,             \
                             sizeof(BLOCK));                                                               \
    }

#define MOE_GEMM_ALL_DST(TAG, QK, BLOCK, DEQ)        \
    MOE_GEMM(TAG, QK, BLOCK, DEQ, float, f32)        \
    MOE_GEMM(TAG, QK, BLOCK, DEQ, __half, f16)       \
    MOE_GEMM(TAG, QK, BLOCK, DEQ, __nv_bfloat16, bf16)

MOE_GEMM_ALL_DST(q4_0, QK4_0, block_q4_0, moe_dequant8_q4_0)
MOE_GEMM_ALL_DST(q5_0, QK5_0, block_q5_0, moe_dequant8_q5_0)
MOE_GEMM_ALL_DST(q8_0, QK8_0, block_q8_0, moe_dequant8_q8_0)
MOE_GEMM_ALL_DST(q4k, QK_K, block_q4_K, moe_dequant8_q4_K)
MOE_GEMM_ALL_DST(q6k, QK_K, block_q6_K, moe_dequant8_q6_K)

// Writes 1 when this code object has the WMMA kernels above, 0 otherwise.
extern "C" __global__ void moe_gemm_available(int *out) { *out = MOE_HAS_WMMA; }
