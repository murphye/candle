// CUDA's asynchronous global-to-shared copy API. There is no direct HIP
// equivalent, so copies complete synchronously and the commit/wait calls are
// no-ops; kernels still synchronize their thread blocks around the copies.
#pragma once

__device__ __forceinline__ void __pipeline_memcpy_async(void *dst, const void *src, size_t size,
                                                        size_t = 0) {
    if (size == 16) {
        *reinterpret_cast<int4 *>(dst) = *reinterpret_cast<const int4 *>(src);
    } else if (size == 8) {
        *reinterpret_cast<int2 *>(dst) = *reinterpret_cast<const int2 *>(src);
    } else if (size == 4) {
        *reinterpret_cast<int *>(dst) = *reinterpret_cast<const int *>(src);
    } else {
        for (size_t i = 0; i < size; ++i) {
            reinterpret_cast<char *>(dst)[i] = reinterpret_cast<const char *>(src)[i];
        }
    }
}

__device__ __forceinline__ void __pipeline_commit() {}

__device__ __forceinline__ void __pipeline_wait_prior(size_t) {}
