// Grouped MoE GEMM on quantized expert weights for prefill-sized batches.
//
// `moe_route` sorts the (token, slot) pairs by expert on the device and cuts each
// expert's rows into BM-row tiles; `moe_gemm_*` then computes one BM x BN output
// tile per workgroup: it dequantizes the expert's weight slice into LDS as bf16 and
// multiplies with WMMA (gfx11 / gfx12), accumulating in f32. Nothing is padded,
// no dequantized copy of the experts is written to memory and the host never waits.
#include "quantized.cu"

#include <rocwmma/rocwmma.hpp>


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
                                     const int bm, uint32_t *__restrict__ sorted, uint32_t *__restrict__ offsets,
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
            tile += (counts[e] + bm - 1) / bm;
        }
        offsets[n_experts] = row;
        *n_tiles = tile;
    }
    __syncthreads();
    for (int e = tid; e < n_experts; e += blockDim.x) {
        const uint32_t ntiles = (counts[e] + bm - 1) / bm;
        for (uint32_t t = 0; t < ntiles; ++t) {
            tiles[tile_starts[e] + t] = make_int2(e, starts[e] + t * bm);
        }
    }
    __syncthreads();
    for (int p = tid; p < n_pairs; p += blockDim.x) {
        const uint32_t at = atomicAdd(&starts[ids[p]], 1u);
        sorted[at] = p;
    }
}

#include "dequant8.cuh"

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
// f32 -> bf16 bits, round to nearest even. gfx12 has no conversion instruction and
// __float2bfloat16 compiles to a NaN check under exec-mask juggling plus six rounding ops per
// value; dequantized weights are finite, so the rounding alone will do (three VALU ops).
static __device__ __forceinline__ uint32_t moe_bf16_bits(float f) {
    const uint32_t u = __float_as_uint(f);
    return (u + 0x7fffu + ((u >> 16) & 1u)) >> 16;
}
static __device__ __forceinline__ uint32_t moe_bf16x2_bits(float lo, float hi) {
    return moe_bf16_bits(lo) | (moe_bf16_bits(hi) << 16);
}

// A workgroup barrier that publishes LDS writes only. `__syncthreads()` also waits for every
// outstanding global load, which would make the K loop's prefetch of the next step wait at
// each barrier; this leaves those loads in flight.
static __device__ __forceinline__ void moe_lds_barrier() {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}

// `x_per_pair` selects x's row: the pair itself ([pairs, k], down projections) or its
// token (pair / topk, [tokens, k], gate and up). `qk` weights per quant block,
// `parts` = qk / 8.
// BM x BN output tiles; eight waves in a (BM / 32) x (8 / (BM / 32)) grid, each owning 32 rows.
// The K loop is a two-stage pipeline: a step's x rows and raw weight words are loaded into
// registers while the previous step's tile is written to LDS and multiplied, so the global
// load latency hides behind the MMAs instead of stalling every step.
template <typename T, int QK, void (*load8)(const void *, int64_t, int, moe_raw8 &),
          void (*conv8)(const moe_raw8 &, int, float *), int MOE_BM, int MOE_BN>
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

    // The output tile reuses the operand tiles' LDS once the K loop is done, in as many
    // passes over the rows as it takes to fit, so the operands alone size the LDS footprint
    // (and with it the workgroups per WGP).
    constexpr int OPERAND_BYTES = (MOE_BM + MOE_BN) * MOE_LDS_K * sizeof(bfloat16_t);
    constexpr int OUTPUT_BYTES = MOE_BM * MOE_BN * sizeof(float);
    constexpr int OUTPUT_PASSES = (OUTPUT_BYTES + OPERAND_BYTES - 1) / OPERAND_BYTES;
    constexpr int PASS_ROWS = MOE_BM / OUTPUT_PASSES;
    static_assert(PASS_ROWS % 32 == 0 && PASS_ROWS * OUTPUT_PASSES == MOE_BM, "output passes cover whole waves");
    static_assert(PASS_ROWS * MOE_BN * (int)sizeof(float) <= OPERAND_BYTES, "an output pass fits the operand LDS");
    __shared__ __align__(16) char lds[OPERAND_BYTES];
    bfloat16_t *a_lds = (bfloat16_t *)lds;
    bfloat16_t *b_lds = a_lds + MOE_BM * MOE_LDS_K;
    float *c_lds = (float *)lds;
    __shared__ uint32_t pair_of[MOE_BM];
    // x's row for every tile row, resolved once: `pair / topk` in the K loop was a software
    // division per load.
    __shared__ uint32_t xrow_of[MOE_BM];

    if (tid < MOE_BM) {
        const uint32_t pair = tid < rows ? sorted[row0 + tid] : 0;
        pair_of[tid] = pair;
        xrow_of[tid] = x_per_pair ? pair : pair / topk;
    }
    __syncthreads();

    // Tall tiles let an expert's rows share one dequantization of its weights; waves
    // whose rows are all padding skip the MMAs.
    constexpr int WAVES_M = MOE_BM / 32;
    constexpr int WAVES_N = (MOE_THREADS / 32) / WAVES_M;
    constexpr int FN = MOE_BN / WAVES_N / 16;
    static_assert(FN * WAVES_N * 16 == MOE_BN, "the N tile divides over the waves");
    const int wave_m = wave / WAVES_N;
    const int wave_n = wave % WAVES_N;
    const bool wave_active = wave_m * 32 < rows;
    fragment<accumulator, 16, 16, 16, float32_t> acc[2][FN];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
#pragma unroll
        for (int j = 0; j < FN; ++j) {
            fill_fragment(acc[i][j], 0.0f);
        }
    }

    const int blocks_per_row = k / QK;
    const char *wexp = (const char *)w + (int64_t)expert * n * blocks_per_row * block_bytes;

    // Per thread and K step: A_GROUPS 16-byte pieces of x rows, B_GROUPS eight-weight pieces.
    constexpr int A_GROUPS = MOE_BM * (MOE_BK / 8) / MOE_THREADS;
    constexpr int B_GROUPS = MOE_BN * (MOE_BK / 8) / MOE_THREADS;
    static_assert(A_GROUPS * MOE_THREADS == MOE_BM * (MOE_BK / 8), "the A tile divides over the threads");
    static_assert(B_GROUPS * MOE_THREADS == MOE_BN * (MOE_BK / 8), "the B tile divides over the threads");

    // Global loads of one K step into registers.
    auto load_step = [&](const int k0, uint4 (&a_reg)[A_GROUPS], moe_raw8 (&b_reg)[B_GROUPS]) {
#pragma unroll
        for (int i = 0; i < A_GROUPS; ++i) {
            const int g = tid + i * MOE_THREADS;
            const int r = g / (MOE_BK / 8);
            const int c = (g % (MOE_BK / 8)) * 8;
            uint4 packed = make_uint4(0, 0, 0, 0);
            if (r < rows) {
                const T *src = x + (int64_t)xrow_of[r] * k + k0 + c;
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
            a_reg[i] = packed;
        }
#pragma unroll
        for (int i = 0; i < B_GROUPS; ++i) {
            const int g = tid + i * MOE_THREADS;
            const int nn = g / (MOE_BK / 8);
            const int c = (g % (MOE_BK / 8)) * 8;
            if (n0 + nn < n) {
                const int kk = k0 + c;
                const int64_t ib = (int64_t)(n0 + nn) * blocks_per_row + kk / QK;
                load8(wexp, ib, (kk % QK) / 8, b_reg[i]);
            }
        }
    };
    // The registers of K step `k0` into the LDS tiles, weights converted to bf16.
    auto store_step = [&](const int k0, const uint4 (&a_reg)[A_GROUPS], const moe_raw8 (&b_reg)[B_GROUPS]) {
#pragma unroll
        for (int i = 0; i < A_GROUPS; ++i) {
            const int g = tid + i * MOE_THREADS;
            const int r = g / (MOE_BK / 8);
            const int c = (g % (MOE_BK / 8)) * 8;
            *(uint4 *)(a_lds + r * MOE_LDS_K + c) = a_reg[i];
        }
#pragma unroll
        for (int i = 0; i < B_GROUPS; ++i) {
            const int g = tid + i * MOE_THREADS;
            const int nn = g / (MOE_BK / 8);
            const int c = (g % (MOE_BK / 8)) * 8;
            uint4 packed = make_uint4(0u, 0u, 0u, 0u);
            if (n0 + nn < n) {
                float f[8];
                conv8(b_reg[i], ((k0 + c) % QK) / 8, f);
                packed = make_uint4(moe_bf16x2_bits(f[0], f[1]), moe_bf16x2_bits(f[2], f[3]),
                                    moe_bf16x2_bits(f[4], f[5]), moe_bf16x2_bits(f[6], f[7]));
            }
            *(uint4 *)(b_lds + nn * MOE_LDS_K + c) = packed;
        }
    };

    uint4 a_cur[A_GROUPS];
    moe_raw8 b_cur[B_GROUPS];
    load_step(0, a_cur, b_cur);
    for (int k0 = 0; k0 < k; k0 += MOE_BK) {
        store_step(k0, a_cur, b_cur);
        moe_lds_barrier();
        // The next step's loads are issued after the LDS write, so the wait for this step's
        // registers above does not also wait for them; they land during the MMAs below.
        const bool more = k0 + MOE_BK < k;
        uint4 a_nxt[A_GROUPS];
        moe_raw8 b_nxt[B_GROUPS];
        if (more) {
            load_step(k0 + MOE_BK, a_nxt, b_nxt);
        }
        if (wave_active) {
#pragma unroll
            for (int kk = 0; kk < MOE_BK; kk += 16) {
                fragment<matrix_a, 16, 16, 16, bfloat16_t, row_major> a[2];
                fragment<matrix_b, 16, 16, 16, bfloat16_t, col_major> b[FN];
#pragma unroll
                for (int i = 0; i < 2; ++i) {
                    load_matrix_sync(a[i], a_lds + (wave_m * 32 + i * 16) * MOE_LDS_K + kk, MOE_LDS_K);
                }
#pragma unroll
                for (int j = 0; j < FN; ++j) {
                    load_matrix_sync(b[j], b_lds + (wave_n * FN * 16 + j * 16) * MOE_LDS_K + kk, MOE_LDS_K);
                }
#pragma unroll
                for (int i = 0; i < 2; ++i) {
#pragma unroll
                    for (int j = 0; j < FN; ++j) {
                        mma_sync(acc[i][j], a[i], b[j], acc[i][j]);
                    }
                }
            }
        }
        moe_lds_barrier();
        if (more) {
#pragma unroll
            for (int i = 0; i < A_GROUPS; ++i) {
                a_cur[i] = a_nxt[i];
            }
#pragma unroll
            for (int i = 0; i < B_GROUPS; ++i) {
                b_cur[i] = b_nxt[i];
            }
        }
    }
#pragma unroll
    for (int pass = 0; pass < OUTPUT_PASSES; ++pass) {
        const int pass_row0 = pass * PASS_ROWS;
        if (wave_m * 32 >= pass_row0 && wave_m * 32 < pass_row0 + PASS_ROWS) {
#pragma unroll
            for (int i = 0; i < 2; ++i) {
#pragma unroll
                for (int j = 0; j < FN; ++j) {
                    store_matrix_sync(c_lds + (wave_m * 32 - pass_row0 + i * 16) * MOE_BN + (wave_n * FN + j) * 16,
                                      acc[i][j], MOE_BN, mem_row_major);
                }
            }
        }
        __syncthreads();
        for (int e = tid; e < PASS_ROWS * MOE_BN; e += MOE_THREADS) {
            const int r = pass_row0 + e / MOE_BN;
            const int c = e % MOE_BN;
            if (r < rows && n0 + c < n) {
                y[(int64_t)pair_of[r] * n + n0 + c] = moe_from_float<T>(c_lds[e]);
            }
        }
        __syncthreads();
    }
#endif
}

// `wide` (32 x 128 tiles) suits few rows per expert, `tall` (128 x 64) many. (A 128 x 160
// tile, reading x 2.5x less often with five MMAs per A fragment, measured no faster: the
// kernel is not bound by LDS reads once the per-load division and the scale-word selects
// are gone, and the wider tile costs occupancy.)
#define MOE_GEMM(TAG, QK, BLOCK, LOAD, CONV, T, DST)                                                              \
    extern "C" __global__ void moe_gemm_##TAG##_##DST##_wide(                                              \
        const void *w, const T *x, T *y, const uint32_t *sorted, const uint32_t *offsets,                 \
        const int2 *tiles, const uint32_t *n_tiles, const int n, const int k, const int topk,             \
        const int x_per_pair) {                                                                            \
        moe_gemm<T, QK, LOAD, CONV, 32, 128>(w, x, y, sorted, offsets, tiles, n_tiles, n, k, topk, x_per_pair,    \
                                      sizeof(BLOCK));                                                      \
    }                                                                                                      \
    extern "C" __global__ void moe_gemm_##TAG##_##DST##_tall(                                              \
        const void *w, const T *x, T *y, const uint32_t *sorted, const uint32_t *offsets,                 \
        const int2 *tiles, const uint32_t *n_tiles, const int n, const int k, const int topk,             \
        const int x_per_pair) {                                                                            \
        moe_gemm<T, QK, LOAD, CONV, 128, 64>(w, x, y, sorted, offsets, tiles, n_tiles, n, k, topk, x_per_pair,    \
                                      sizeof(BLOCK));                                                      \
    }

#define MOE_GEMM_ALL_DST(TAG, QK, BLOCK, LOAD, CONV)        \
    MOE_GEMM(TAG, QK, BLOCK, LOAD, CONV, float, f32)        \
    MOE_GEMM(TAG, QK, BLOCK, LOAD, CONV, __half, f16)       \
    MOE_GEMM(TAG, QK, BLOCK, LOAD, CONV, __nv_bfloat16, bf16)

MOE_GEMM_ALL_DST(q4_0, QK4_0, block_q4_0, moe_load8_q4_0, moe_conv8_q4_0)
MOE_GEMM_ALL_DST(q5_0, QK5_0, block_q5_0, moe_load8_q5_0, moe_conv8_q5_0)
MOE_GEMM_ALL_DST(q8_0, QK8_0, block_q8_0, moe_load8_q8_0, moe_conv8_q8_0)
MOE_GEMM_ALL_DST(q4k, QK_K, block_q4_K, moe_load8_q4_K, moe_conv8_q4_K)
MOE_GEMM_ALL_DST(q6k, QK_K, block_q6_K, moe_load8_q6_K, moe_conv8_q6_K)

// Writes 1 when this code object has the WMMA kernels above, 0 otherwise.
extern "C" __global__ void moe_gemm_available(int *out) { *out = MOE_HAS_WMMA; }

// ---------------------------------------------------------------------------------------
// int8 path: the activations are quantized to int8 per 32-element block once per call, the
// weights' quant bytes go to LDS as int8 (Q8_0) or in-order unsigned nibbles (Q4_K: one AND
// per dword), and the tiles multiply on the int8 WMMA. gfx12 has no f32 -> bf16 instruction,
// so the bf16 path above spends five VALU ops (and a select) per dequantized weight; here the
// per-32-block scales are applied once per 16 x 16 tile instead.
// ---------------------------------------------------------------------------------------

// `xq[rows][k]` int8 in blocks of 32 and `xm[rows][k / 32]` = {d, d * sum(q)}: one wave per
// block, one lane per element.
template <typename T>
static __device__ void moe_quant_x_q8(const T *__restrict__ x, int8_t *__restrict__ xq,
                                      float2 *__restrict__ xm, const int k) {
    const int row = blockIdx.x;
    const int lane = threadIdx.x % 32;
    const int wave = threadIdx.x / 32;
    const int nblocks = k / 32;
    for (int kb = wave; kb < nblocks; kb += blockDim.x / 32) {
        const size_t at = (size_t)row * k + kb * 32 + lane;
        const float v = moe_to_float(x[at]);
        float amax = fabsf(v);
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o, 32));
        }
        const float d = amax / 127.0f;
        const float id = amax > 0.0f ? 127.0f / amax : 0.0f;
        const int q = __float2int_rn(v * id);
        int sum = q;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            sum += __shfl_xor_sync(0xffffffffu, sum, o, 32);
        }
        xq[at] = (int8_t)q;
        if (lane == 0) {
            xm[(size_t)row * nblocks + kb] = make_float2(d, d * (float)sum);
        }
    }
}

#define MOE_QUANT_X(T, TAG)                                                                              \
    extern "C" __global__ void moe_quant_x_q8_##TAG(const T *x, int8_t *xq, float2 *xm, const int k) { \
        moe_quant_x_q8<T>(x, xq, xm, k);                                                                 \
    }
MOE_QUANT_X(float, f32)
MOE_QUANT_X(__half, f16)
MOE_QUANT_X(__nv_bfloat16, bf16)

// Eight weights of a part as two dwords of int8 (Q8_0) or in-order unsigned nibbles (Q4_K),
// plus the sub-block's (scale, min) in output units: w = scale * q - min.
static __device__ __forceinline__ void moe_i8_q4_K(const moe_raw8 &r, int part, uint32_t &lo, uint32_t &hi,
                                                   float &scale, float &min) {
    const int sub = part / 4;
    const int shift = 4 * (sub & 1);
    lo = (r.w[4] >> shift) & 0x0F0F0F0Fu;
    hi = (r.w[5] >> shift) & 0x0F0F0F0Fu;
    const uint32_t b0 = r.w[1], b1 = r.w[2], b2 = r.w[3];
    const uint32_t sc = sub < 4 ? (b0 & 63) : ((b0 & 0xF) | ((b1 >> 6) << 4));
    const uint32_t m = sub < 4 ? (b1 & 63) : ((b0 >> 4) | ((b2 >> 6) << 4));
    scale = __half2float(__ushort_as_half((uint16_t)r.w[0])) * sc;
    min = __half2float(__ushort_as_half((uint16_t)(r.w[0] >> 16))) * m;
}

static __device__ __forceinline__ void moe_i8_q8_0(const moe_raw8 &r, int part, uint32_t &lo, uint32_t &hi,
                                                   float &scale, float &min) {
    lo = r.w[1];
    hi = r.w[2];
    scale = __half2float(__ushort_as_half((uint16_t)r.w[0]));
    min = 0.0f;
}

typedef int moe_v2i __attribute__((ext_vector_type(2)));
typedef int moe_v8i __attribute__((ext_vector_type(8)));

// int8 tile rows are padded to 72 bytes: 8-byte aligned, and 16 consecutive rows at one k
// offset fall into distinct LDS banks.
#define MOE_I8_LDS_K (MOE_BK + 8)

// y[pair, n] = x[row(pair), :] . w[expert(pair), n, :] on the int8 WMMA. `B_SIGNED`: the
// weight bytes are signed (Q8_0) rather than nibbles 0..15 with a per-block min (Q4_K).
template <typename T, int QK, void (*load8)(const void *, int64_t, int, moe_raw8 &),
          void (*conv)(const moe_raw8 &, int, uint32_t &, uint32_t &, float &, float &), bool B_SIGNED,
          int MOE_BM, int MOE_BN>
static __device__ void moe_gemm_i8(const void *__restrict__ w, const int8_t *__restrict__ xq,
                                   const float2 *__restrict__ xm, T *__restrict__ y,
                                   const uint32_t *__restrict__ sorted, const uint32_t *__restrict__ offsets,
                                   const int2 *__restrict__ tiles, const uint32_t *__restrict__ n_tiles,
                                   const int n, const int k, const int topk, const int x_per_pair,
                                   const int64_t block_bytes) {
#if MOE_HAS_WMMA && (defined(__gfx1200__) || defined(__gfx1201__))
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
    const int lane = tid % 32;

    constexpr int META_BYTES = 2 * (MOE_BM + MOE_BN) * (int)sizeof(float2);
    constexpr int OPERAND_BYTES = (MOE_BM + MOE_BN) * MOE_I8_LDS_K + META_BYTES;
    constexpr int OUTPUT_BYTES = MOE_BM * MOE_BN * (int)sizeof(float);
    // The output is staged in whole-wave row passes; the int8 operands are small enough that
    // the wide tile's single pass is what sizes the LDS.
    constexpr int OUTPUT_PASSES = OUTPUT_BYTES <= OPERAND_BYTES              ? 1
                                  : ((MOE_BM / 32) % 2 == 0 && OUTPUT_BYTES / 2 <= OPERAND_BYTES) ? 2
                                  : ((MOE_BM / 32) % 4 == 0 && OUTPUT_BYTES / 4 <= OPERAND_BYTES) ? 4
                                                                                                   : 1;
    constexpr int PASS_ROWS = MOE_BM / OUTPUT_PASSES;
    constexpr int PASS_BYTES = PASS_ROWS * MOE_BN * (int)sizeof(float);
    constexpr int LDS_BYTES = OPERAND_BYTES > PASS_BYTES ? OPERAND_BYTES : PASS_BYTES;
    static_assert(PASS_ROWS % 32 == 0 && PASS_ROWS * OUTPUT_PASSES == MOE_BM, "output passes cover whole waves");
    __shared__ __align__(16) char lds[LDS_BYTES];
    int8_t *a_lds = (int8_t *)lds;
    int8_t *b_lds = a_lds + MOE_BM * MOE_I8_LDS_K;
    float2 *a_meta = (float2 *)(b_lds + MOE_BN * MOE_I8_LDS_K);  // [2][MOE_BM]
    float2 *b_meta = a_meta + 2 * MOE_BM;                         // [2][MOE_BN]
    float *c_lds = (float *)lds;
    __shared__ uint32_t pair_of[MOE_BM];
    __shared__ uint32_t xrow_of[MOE_BM];

    if (tid < MOE_BM) {
        const uint32_t pair = tid < rows ? sorted[row0 + tid] : 0;
        pair_of[tid] = pair;
        xrow_of[tid] = x_per_pair ? pair : pair / topk;
    }
    __syncthreads();

    constexpr int WAVES_M = MOE_BM / 32;
    constexpr int WAVES_N = (MOE_THREADS / 32) / WAVES_M;
    constexpr int FN = MOE_BN / WAVES_N / 16;
    static_assert(FN * WAVES_N * 16 == MOE_BN, "the N tile divides over the waves");
    const int wave_m = wave / WAVES_N;
    const int wave_n = wave % WAVES_N;
    const bool wave_active = wave_m * 32 < rows;
    float facc[2][FN][8];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
#pragma unroll
        for (int j = 0; j < FN; ++j) {
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                facc[i][j][e] = 0.0f;
            }
        }
    }

    const int blocks_per_row = k / QK;
    const int kblocks = k / 32;
    const char *wexp = (const char *)w + (int64_t)expert * n * blocks_per_row * block_bytes;

    // Per thread and K step: A_GROUPS 8-byte pieces of x rows, B_GROUPS eight-weight pieces.
    // A thread's k offset `c` within the step is the same for all its pieces; the threads at
    // c = 0 and c = 32 also carry their rows'/columns' block scales.
    constexpr int A_GROUPS = MOE_BM * (MOE_BK / 8) / MOE_THREADS;
    constexpr int B_GROUPS = MOE_BN * (MOE_BK / 8) / MOE_THREADS;
    static_assert(A_GROUPS * MOE_THREADS == MOE_BM * (MOE_BK / 8), "the A tile divides over the threads");
    static_assert(B_GROUPS * MOE_THREADS == MOE_BN * (MOE_BK / 8), "the B tile divides over the threads");
    const int c = (tid % (MOE_BK / 8)) * 8;
    const bool meta_thread = (c % 32) == 0;

    auto load_step = [&](const int k0, uint2 (&a_reg)[A_GROUPS], float2 (&am_reg)[A_GROUPS],
                         moe_raw8 (&b_reg)[B_GROUPS]) {
#pragma unroll
        for (int i = 0; i < A_GROUPS; ++i) {
            const int r = (tid + i * MOE_THREADS) / (MOE_BK / 8);
            a_reg[i] = make_uint2(0u, 0u);
            am_reg[i] = make_float2(0.0f, 0.0f);
            if (r < rows) {
                const size_t xrow = xrow_of[r];
                a_reg[i] = *(const uint2 *)(xq + xrow * k + k0 + c);
                if (meta_thread) {
                    am_reg[i] = xm[xrow * kblocks + (k0 + c) / 32];
                }
            }
        }
#pragma unroll
        for (int i = 0; i < B_GROUPS; ++i) {
            const int nn = (tid + i * MOE_THREADS) / (MOE_BK / 8);
            if (n0 + nn < n) {
                const int kk = k0 + c;
                const int64_t ib = (int64_t)(n0 + nn) * blocks_per_row + kk / QK;
                load8(wexp, ib, (kk % QK) / 8, b_reg[i]);
            }
        }
    };
    auto store_step = [&](const int k0, const uint2 (&a_reg)[A_GROUPS], const float2 (&am_reg)[A_GROUPS],
                          const moe_raw8 (&b_reg)[B_GROUPS]) {
#pragma unroll
        for (int i = 0; i < A_GROUPS; ++i) {
            const int r = (tid + i * MOE_THREADS) / (MOE_BK / 8);
            *(uint2 *)(a_lds + r * MOE_I8_LDS_K + c) = a_reg[i];
            if (meta_thread) {
                a_meta[(c / 32) * MOE_BM + r] = am_reg[i];
            }
        }
#pragma unroll
        for (int i = 0; i < B_GROUPS; ++i) {
            const int nn = (tid + i * MOE_THREADS) / (MOE_BK / 8);
            uint32_t lo = 0u, hi = 0u;
            float scale = 0.0f, mn = 0.0f;
            if (n0 + nn < n) {
                conv(b_reg[i], ((k0 + c) % QK) / 8, lo, hi, scale, mn);
            }
            *(uint2 *)(b_lds + nn * MOE_I8_LDS_K + c) = make_uint2(lo, hi);
            if (meta_thread) {
                b_meta[(c / 32) * MOE_BN + nn] = make_float2(scale, mn);
            }
        }
    };

    uint2 a_cur[A_GROUPS];
    float2 am_cur[A_GROUPS];
    moe_raw8 b_cur[B_GROUPS];
    load_step(0, a_cur, am_cur, b_cur);
    for (int k0 = 0; k0 < k; k0 += MOE_BK) {
        store_step(k0, a_cur, am_cur, b_cur);
        moe_lds_barrier();
        const bool more = k0 + MOE_BK < k;
        uint2 a_nxt[A_GROUPS];
        float2 am_nxt[A_GROUPS];
        moe_raw8 b_nxt[B_GROUPS];
        if (more) {
            load_step(k0 + MOE_BK, a_nxt, am_nxt, b_nxt);
        }
        if (wave_active) {
            const int a_row0 = wave_m * 32 + 8 * (lane / 16);
            const int b_row = wave_n * FN * 16 + lane % 16;
#pragma unroll
            for (int kb = 0; kb < MOE_BK / 32; ++kb) {
                float2 bm[FN];
#pragma unroll
                for (int j = 0; j < FN; ++j) {
                    bm[j] = b_meta[kb * MOE_BN + b_row + j * 16];
                }
#pragma unroll
                for (int i = 0; i < 2; ++i) {
                    // The lane's eight output rows of fragment i and their block scales.
                    const int row = a_row0 + i * 16;
                    const float4 *am4 = (const float4 *)(a_meta + kb * MOE_BM + row);
                    float2 am[8];
#pragma unroll
                    for (int e = 0; e < 4; ++e) {
                        const float4 v = am4[e];
                        am[2 * e] = make_float2(v.x, v.y);
                        am[2 * e + 1] = make_float2(v.z, v.w);
                    }
#pragma unroll
                    for (int j = 0; j < FN; ++j) {
                        moe_v8i acc = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
                        for (int kk = kb * 32; kk < kb * 32 + 32; kk += 16) {
                            const moe_v2i a = *(const moe_v2i *)(a_lds + (wave_m * 32 + i * 16 + lane % 16) * MOE_I8_LDS_K + kk + 8 * (lane / 16));
                            const moe_v2i b = *(const moe_v2i *)(b_lds + (b_row + j * 16) * MOE_I8_LDS_K + kk + 8 * (lane / 16));
                            acc = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a, B_SIGNED, b, acc, false);
                        }
#pragma unroll
                        for (int e = 0; e < 8; ++e) {
                            float v = (float)acc[e] * (am[e].x * bm[j].x);
                            if constexpr (!B_SIGNED) {
                                v -= am[e].y * bm[j].y;
                            }
                            facc[i][j][e] += v;
                        }
                    }
                }
            }
        }
        moe_lds_barrier();
        if (more) {
#pragma unroll
            for (int i = 0; i < A_GROUPS; ++i) {
                a_cur[i] = a_nxt[i];
                am_cur[i] = am_nxt[i];
            }
#pragma unroll
            for (int i = 0; i < B_GROUPS; ++i) {
                b_cur[i] = b_nxt[i];
            }
        }
    }
#pragma unroll
    for (int pass = 0; pass < OUTPUT_PASSES; ++pass) {
        const int pass_row0 = pass * PASS_ROWS;
        if (wave_m * 32 >= pass_row0 && wave_m * 32 < pass_row0 + PASS_ROWS) {
#pragma unroll
            for (int i = 0; i < 2; ++i) {
#pragma unroll
                for (int j = 0; j < FN; ++j) {
#pragma unroll
                    for (int e = 0; e < 8; ++e) {
                        const int r = wave_m * 32 - pass_row0 + i * 16 + 8 * (lane / 16) + e;
                        c_lds[r * MOE_BN + (wave_n * FN + j) * 16 + lane % 16] = facc[i][j][e];
                    }
                }
            }
        }
        __syncthreads();
        for (int e = tid; e < PASS_ROWS * MOE_BN; e += MOE_THREADS) {
            const int r = pass_row0 + e / MOE_BN;
            const int cc = e % MOE_BN;
            if (r < rows && n0 + cc < n) {
                y[(int64_t)pair_of[r] * n + n0 + cc] = moe_from_float<T>(c_lds[e]);
            }
        }
        __syncthreads();
    }
#endif
}

#define MOE_GEMM_I8(TAG, QK, BLOCK, LOAD, CONV, BSIGNED, T, DST)                                             \
    extern "C" __global__ void __launch_bounds__(MOE_THREADS) moe_gemm_i8_##TAG##_##DST##_wide(              \
        const void *w, const int8_t *xq, const float2 *xm, T *y, const uint32_t *sorted,                     \
        const uint32_t *offsets, const int2 *tiles, const uint32_t *n_tiles, const int n, const int k,        \
        const int topk, const int x_per_pair) {                                                              \
        moe_gemm_i8<T, QK, LOAD, CONV, BSIGNED, 32, 128>(w, xq, xm, y, sorted, offsets, tiles, n_tiles, n, k,  \
                                                        topk, x_per_pair, sizeof(BLOCK));                    \
    }                                                                                                        \
    extern "C" __global__ void __launch_bounds__(MOE_THREADS) moe_gemm_i8_##TAG##_##DST##_tall(              \
        const void *w, const int8_t *xq, const float2 *xm, T *y, const uint32_t *sorted,                     \
        const uint32_t *offsets, const int2 *tiles, const uint32_t *n_tiles, const int n, const int k,        \
        const int topk, const int x_per_pair) {                                                              \
        moe_gemm_i8<T, QK, LOAD, CONV, BSIGNED, 128, 64>(w, xq, xm, y, sorted, offsets, tiles, n_tiles, n, k,  \
                                                        topk, x_per_pair, sizeof(BLOCK));                    \
    }

#define MOE_GEMM_I8_ALL_DST(TAG, QK, BLOCK, LOAD, CONV, BSIGNED)        \
    MOE_GEMM_I8(TAG, QK, BLOCK, LOAD, CONV, BSIGNED, float, f32)        \
    MOE_GEMM_I8(TAG, QK, BLOCK, LOAD, CONV, BSIGNED, __half, f16)       \
    MOE_GEMM_I8(TAG, QK, BLOCK, LOAD, CONV, BSIGNED, __nv_bfloat16, bf16)

MOE_GEMM_I8_ALL_DST(q4k, QK_K, block_q4_K, moe_load8_q4_K, moe_i8_q4_K, false)
MOE_GEMM_I8_ALL_DST(q8_0, QK8_0, block_q8_0, moe_load8_q8_0, moe_i8_q8_0, true)

// Writes 1 when this code object has the int8 WMMA kernels above (gfx12), 0 otherwise.
extern "C" __global__ void moe_gemm_i8_available(int *out) {
#if MOE_HAS_WMMA && (defined(__gfx1200__) || defined(__gfx1201__))
    *out = 1;
#else
    *out = 0;
#endif
}

