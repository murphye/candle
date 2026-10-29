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

    // Tall tiles let an expert's rows share one dequantization of its weights; waves
    // whose rows are all padding skip the MMAs.
    constexpr int WAVES_M = MOE_BM / 32;
    constexpr int WAVES_N = (MOE_THREADS / 32) / WAVES_M;
    constexpr int FN = MOE_BN / WAVES_N / 16;
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
            __align__(16) bfloat16_t v[8];
            if (n0 + nn < n) {
                float f[8];
                conv8(b_reg[i], ((k0 + c) % QK) / 8, f);
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
    };

    uint4 a_cur[A_GROUPS];
    moe_raw8 b_cur[B_GROUPS];
    load_step(0, a_cur, b_cur);
    for (int k0 = 0; k0 < k; k0 += MOE_BK) {
        store_step(k0, a_cur, b_cur);
        __syncthreads();
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
        __syncthreads();
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
    for (int i = 0; i < 2; ++i) {
#pragma unroll
        for (int j = 0; j < FN; ++j) {
            store_matrix_sync(c_lds + (wave_m * 32 + i * 16) * MOE_BN + (wave_n * FN + j) * 16, acc[i][j],
                              MOE_BN, mem_row_major);
        }
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

// `wide` (32 x 128 tiles) suits few rows per expert, `tall` (128 x 64) many.
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
