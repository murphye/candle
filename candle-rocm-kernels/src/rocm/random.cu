// Counter-based random number generation for the ROCm backend.
//
// Each element is derived from (seed, offset + index) through a splitmix64-style
// mixer, so results depend only on the seed and the number of values generated
// so far, not on the launch configuration.
#include <stdint.h>

__device__ __forceinline__ uint64_t mix64(uint64_t z) {
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

__device__ __forceinline__ uint64_t rand_u64(uint64_t seed, uint64_t counter) {
    return mix64(seed ^ mix64(counter + 0x9e3779b97f4a7c15ULL));
}

// Uniform in (0, 1]: never zero, so it is safe to take the log.
__device__ __forceinline__ double uniform_f64(uint64_t seed, uint64_t counter) {
    return ((rand_u64(seed, counter) >> 11) + 1) * (1.0 / 9007199254740992.0);
}

__device__ __forceinline__ float uniform_f32(uint64_t seed, uint64_t counter) {
    return ((uint32_t)(rand_u64(seed, counter) >> 40) + 1) * (1.0f / 16777216.0f);
}

#define RAND_OPS(TYPENAME, RUST_NAME, ACC, UNIFORM, LOG, SQRT, COS, TO, PI)        \
extern "C" __global__ void rand_uniform_##RUST_NAME(                               \
    const size_t numel, const uint64_t seed, const uint64_t offset,                \
    const ACC lo, const ACC up, TYPENAME *out) {                                   \
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < numel;              \
         i += blockDim.x * gridDim.x) {                                            \
        ACC u = UNIFORM(seed, offset + i);                                         \
        out[i] = TO(lo + (up - lo) * u);                                           \
    }                                                                              \
}                                                                                  \
extern "C" __global__ void rand_normal_##RUST_NAME(                                \
    const size_t numel, const uint64_t seed, const uint64_t offset,                \
    const ACC mean, const ACC std, TYPENAME *out) {                                \
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < numel;              \
         i += blockDim.x * gridDim.x) {                                            \
        ACC u1 = UNIFORM(seed, 2 * (offset + i));                                  \
        ACC u2 = UNIFORM(seed, 2 * (offset + i) + 1);                              \
        ACC z = SQRT(-2 * LOG(u1)) * COS(2 * PI * u2);                             \
        out[i] = TO(mean + std * z);                                               \
    }                                                                              \
}

#define ID(x) (x)
RAND_OPS(float, f32, float, uniform_f32, logf, sqrtf, cosf, ID, 3.14159265358979f)
RAND_OPS(double, f64, double, uniform_f64, log, sqrt, cos, ID, 3.14159265358979323846)
RAND_OPS(__half, f16, float, uniform_f32, logf, sqrtf, cosf, __float2half, 3.14159265358979f)
RAND_OPS(__nv_bfloat16, bf16, float, uniform_f32, logf, sqrtf, cosf, __float2bfloat16, 3.14159265358979f)
