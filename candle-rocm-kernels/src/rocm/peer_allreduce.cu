// One-shot all-reduce between tensor-parallel GPUs that can write each other's memory over
// PCIe (peer access through IPC handles). Small decode-step reductions are bound by latency,
// not bandwidth: each rank pushes its partial sum into a slot of every peer's receive buffer,
// raises a flag there, waits for every peer's flag, and adds what arrived to its own partial.
//
// Each rank's receive buffer (uncached device memory) holds `data[2][world][max_elems]` and
// `flags[world]`; flags only grow, carrying the epoch of the last call. Calls alternate between
// the two data halves, so a peer that has moved on to the next call writes into the half this
// rank is not reading. The epoch lives in device memory and advances in the kernel, which keeps
// eager calls and graph replays in step.
#include "cuda_bf16.h"
#include <stdint.h>

#define PEER_AR_MAX_RANKS 8

typedef unsigned int v4u __attribute__((ext_vector_type(4)));

// `peers`: `world` receive-buffer data pointers, then `world` flag pointers, as seen from this
// rank. `n` is a multiple of 8.
extern "C" __global__ void __launch_bounds__(1024)
    peer_allreduce_bf16(const __nv_bfloat16 *__restrict__ x, __nv_bfloat16 *__restrict__ y,
                        const int n, const int max_elems, const uint64_t *__restrict__ peers,
                        const int rank, const int world, int *__restrict__ epoch) {
  __shared__ int ep;
  const int t = threadIdx.x;
  if (t == 0) {
    ep = *epoch + 1;
  }
  __syncthreads();
  const int half = ep & 1;
  const int nv = n / 8;
  const v4u *xv = reinterpret_cast<const v4u *>(x);
  for (int peer = 0; peer < world; ++peer) {
    if (peer == rank) {
      continue;
    }
    __nv_bfloat16 *slot = reinterpret_cast<__nv_bfloat16 *>(peers[peer]) +
                          (size_t)(half * world + rank) * max_elems;
    v4u *dst = reinterpret_cast<v4u *>(slot);
    for (int i = t; i < nv; i += blockDim.x) {
      dst[i] = xv[i];
    }
  }
  __threadfence_system();
  __syncthreads();
  if (t == 0) {
    for (int peer = 0; peer < world; ++peer) {
      if (peer != rank) {
        int *flags = reinterpret_cast<int *>(peers[world + peer]);
        __hip_atomic_store(flags + rank, ep, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
      }
    }
    int *own = reinterpret_cast<int *>(peers[world + rank]);
    for (int peer = 0; peer < world; ++peer) {
      if (peer != rank) {
        while (__hip_atomic_load(own + peer, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) < ep) {
        }
      }
    }
  }
  __syncthreads();
  const __nv_bfloat16 *recv = reinterpret_cast<const __nv_bfloat16 *>(peers[rank]);
  v4u *yv = reinterpret_cast<v4u *>(y);
  for (int i = t; i < nv; i += blockDim.x) {
    const v4u a = xv[i];
    const __nv_bfloat16 *av = reinterpret_cast<const __nv_bfloat16 *>(&a);
    float acc[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      acc[j] = __bfloat162float(av[j]);
    }
    for (int peer = 0; peer < world; ++peer) {
      if (peer == rank) {
        continue;
      }
      const v4u b = reinterpret_cast<const v4u *>(recv + (size_t)(half * world + peer) * max_elems)[i];
      const __nv_bfloat16 *bv = reinterpret_cast<const __nv_bfloat16 *>(&b);
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        acc[j] += __bfloat162float(bv[j]);
      }
    }
    v4u o;
    __nv_bfloat16 *ov = reinterpret_cast<__nv_bfloat16 *>(&o);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      ov[j] = __float2bfloat16(acc[j]);
    }
    yv[i] = o;
  }
  if (t == 0) {
    *epoch = ep;
  }
}
