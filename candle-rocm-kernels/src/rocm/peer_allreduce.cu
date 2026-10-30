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

// Spinning on a peer's flag gives up after this many 100 MHz ticks (10 s) and traps, so a
// stalled rank surfaces as a launch failure instead of a hung graph.
#define PEER_SPIN_TIMEOUT_TICKS 1000000000ull

// Pushes this rank's `n` elements of `x` into slot `rank` of the current half of every peer's
// receive buffer, publishes the epoch and waits for every peer's epoch. On return the current
// half of this rank's own receive buffer holds every peer's `x`. Must be called by the whole
// (single) workgroup.
static __device__ __forceinline__ int peer_exchange(const __nv_bfloat16 *__restrict__ x, const int n,
                                                     const int max_elems, const uint64_t *__restrict__ peers,
                                                     const int rank, const int world,
                                                     int *__restrict__ epoch) {
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
    const uint64_t start = wall_clock64();
    for (int peer = 0; peer < world; ++peer) {
      if (peer != rank) {
        while (__hip_atomic_load(own + peer, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) < ep) {
          if (wall_clock64() - start > PEER_SPIN_TIMEOUT_TICKS) {
            __builtin_trap();
          }
        }
      }
    }
  }
  __syncthreads();
  return ep;
}

// `peers`: `world` receive-buffer data pointers, then `world` flag pointers, as seen from this
// rank. `n` is a multiple of 8. y = sum over ranks of x.
extern "C" __global__ void __launch_bounds__(1024)
    peer_allreduce_bf16(const __nv_bfloat16 *__restrict__ x, __nv_bfloat16 *__restrict__ y,
                        const int n, const int max_elems, const uint64_t *__restrict__ peers,
                        const int rank, const int world, int *__restrict__ epoch) {
  const int ep = peer_exchange(x, n, max_elems, peers, rank, world, epoch);
  const int half = ep & 1;
  const int t = threadIdx.x;
  const int nv = n / 8;
  const v4u *xv = reinterpret_cast<const v4u *>(x);
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

// y[r * n .. (r + 1) * n) = rank r's x, for every rank: the ranks' inputs concatenated.
extern "C" __global__ void __launch_bounds__(1024)
    peer_allgather_bf16(const __nv_bfloat16 *__restrict__ x, __nv_bfloat16 *__restrict__ y,
                        const int n, const int max_elems, const uint64_t *__restrict__ peers,
                        const int rank, const int world, int *__restrict__ epoch) {
  const int ep = peer_exchange(x, n, max_elems, peers, rank, world, epoch);
  const int half = ep & 1;
  const int t = threadIdx.x;
  const int nv = n / 8;
  const __nv_bfloat16 *recv = reinterpret_cast<const __nv_bfloat16 *>(peers[rank]);
  for (int peer = 0; peer < world; ++peer) {
    const v4u *src = peer == rank
                         ? reinterpret_cast<const v4u *>(x)
                         : reinterpret_cast<const v4u *>(recv + (size_t)(half * world + peer) * max_elems);
    v4u *dst = reinterpret_cast<v4u *>(y + (size_t)peer * n);
    for (int i = t; i < nv; i += blockDim.x) {
      dst[i] = src[i];
    }
  }
  if (t == 0) {
    *epoch = ep;
  }
}

// The bandwidth-sized variant for prefill chunks (megabytes per call) is two launches, so
// that no workgroup ever waits for a peer while workgroups of its own grid are still queued
// (a one-launch grid larger than the GPU holds deadlocks: the resident workgroups spin on the
// peers' flags, which wait for the pushes of the workgroups behind them).
//
// `peer_push_bf16_wide`: the whole grid pushes this rank's x into every peer's slot, a stripe
// per workgroup, so the transfer runs at the link's rate; the last workgroup to finish
// publishes the epoch. `peer_allreduce_bf16_wide` then waits for the peers' epochs, reduces a
// stripe per workgroup, and its last workgroup advances the epoch. `counters` are two u32
// that are zero between launches: [pushed, finished].
extern "C" __global__ void __launch_bounds__(256)
    peer_push_bf16_wide(const __nv_bfloat16 *__restrict__ x, const int n, const int max_elems,
                        const uint64_t *__restrict__ peers, const int rank, const int world,
                        const int *__restrict__ epoch, unsigned int *__restrict__ counters) {
  const int ep = *epoch + 1;
  const int half = ep & 1;
  const int nv = n / 8;
  const int stride = gridDim.x * blockDim.x;
  const int first = blockIdx.x * blockDim.x + threadIdx.x;
  const v4u *xv = reinterpret_cast<const v4u *>(x);
  for (int peer = 0; peer < world; ++peer) {
    if (peer == rank) {
      continue;
    }
    v4u *dst = reinterpret_cast<v4u *>(reinterpret_cast<__nv_bfloat16 *>(peers[peer]) +
                                       (size_t)(half * world + rank) * max_elems);
    for (int i = first; i < nv; i += stride) {
      dst[i] = xv[i];
    }
  }
  // Every wave waits for its pushes to be acknowledged (the agent-scope release) before the
  // workgroup counts itself pushed, so the last workgroup's acquire of the count orders every
  // push before its flag stores, which take the same link in order. A system-scope fence per
  // workgroup here would write back the L2 two thousand times per call.
#ifdef PEER_PUSH_BLOCK_FENCE_SYSTEM
  __threadfence_system();
#else
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
#endif
  __syncthreads();
  if (threadIdx.x == 0) {
    const unsigned int pushed =
        __hip_atomic_fetch_add(counters, 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
    if (pushed == gridDim.x - 1) {
      counters[0] = 0;
      __threadfence_system();
      for (int peer = 0; peer < world; ++peer) {
        if (peer != rank) {
          int *flags = reinterpret_cast<int *>(peers[world + peer]);
          __hip_atomic_store(flags + rank, ep, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
        }
      }
    }
  }
}

extern "C" __global__ void __launch_bounds__(256)
    peer_allreduce_bf16_wide(const __nv_bfloat16 *__restrict__ x, __nv_bfloat16 *__restrict__ y,
                             const int n, const int max_elems, const uint64_t *__restrict__ peers,
                             const int rank, const int world, int *__restrict__ epoch,
                             unsigned int *__restrict__ counters) {
  __shared__ int ep_s;
  const int t = threadIdx.x;
  if (t == 0) {
    ep_s = *epoch + 1;
    int *own = reinterpret_cast<int *>(peers[world + rank]);
    const uint64_t start = wall_clock64();
    for (int peer = 0; peer < world; ++peer) {
      if (peer != rank) {
        while (__hip_atomic_load(own + peer, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) < ep_s) {
          // Many workgroups poll at once; pace them so the polls do not crowd the memory
          // path the peer's pushes are landing on.
          __builtin_amdgcn_s_sleep(2);
          if (wall_clock64() - start > PEER_SPIN_TIMEOUT_TICKS) {
            __builtin_trap();
          }
        }
      }
    }
  }
  __syncthreads();
  const int ep = ep_s;
  const int half = ep & 1;
  const int nv = n / 8;
  const int stride = gridDim.x * blockDim.x;
  const int first = blockIdx.x * blockDim.x + t;
  const v4u *xv = reinterpret_cast<const v4u *>(x);
  const __nv_bfloat16 *recv = reinterpret_cast<const __nv_bfloat16 *>(peers[rank]);
  v4u *yv = reinterpret_cast<v4u *>(y);
  for (int i = first; i < nv; i += stride) {
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
    if (__hip_atomic_fetch_add(counters + 1, 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT) ==
        gridDim.x - 1) {
      counters[1] = 0;
      *epoch = ep;
    }
  }
}
