//! A one-shot all-reduce between tensor-parallel ranks over peer GPU memory (see
//! `peer_allreduce.cu`). RCCL's small-message all-reduce costs tens of microseconds on PCIe;
//! decode steps reduce a few tens of kilobytes about a hundred times, where this takes a few.
//!
//! Each rank allocates a receive buffer, shares its IPC handle with the other ranks (through
//! whatever channel the caller has, such as an RCCL all-gather), and opens theirs.
use std::ffi::c_void;

use super::hip::{self, sys, HipSlice};
use super::{kernels, LaunchConfig, RocmDevice, WrapErr};
use crate::{DType, Result, Storage, Tensor};

pub const MAX_RANKS: usize = 8;
/// Bytes of one rank's shareable handle.
pub const HANDLE_BYTES: usize = 64;
/// The receive buffer starts with the flags, padded so the data stays aligned; small
/// allocations may be carved out of shared blocks that IPC cannot export, so it is one.
const FLAGS_BYTES: usize = 4096;
/// Reductions of at least this many elements take the whole-GPU kernel (a 2 MiB bf16 tensor
/// moves in ~80 us over PCIe 5.0 x16 when every workgroup pushes, far longer from one).
pub const WIDE_MIN_ELEMS: usize = 1 << 16;
const WIDE_BLOCK: usize = 256;
/// Largest gather the one-workgroup gather kernel takes (vocab-sharded logits of a batch).
pub const GATHER_MAX_ELEMS: usize = 1 << 20;
const WIDE_MAX_BLOCKS: usize = 2048;
const WIDE_REDUCE_MAX_BLOCKS: usize = 256;

/// This rank's receive buffer, before the peers' buffers are opened.
pub struct PeerAllReduceSetup {
    device: RocmDevice,
    rank: usize,
    world: usize,
    max_elems: usize,
    base: u64,
}

pub struct PeerAllReduce {
    device: RocmDevice,
    rank: usize,
    world: usize,
    max_elems: usize,
    base: u64,
    /// Peers' buffers mapped into this process; freed with `hipIpcCloseMemHandle`.
    opened: Vec<u64>,
    peers: HipSlice<u64>,
    epoch: HipSlice<i32>,
    /// Two monotonic counters for kernels that run the exchange with many workgroups: tickets
    /// taken at entry (the epoch is `ticket / rows + 1`) and rows pushed so far.
    counters: HipSlice<u32>,
}

impl std::fmt::Debug for PeerAllReduce {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PeerAllReduce")
            .field("rank", &self.rank)
            .field("world", &self.world)
            .field("max_elems", &self.max_elems)
            .finish()
    }
}

// SAFETY: the raw pointers are device addresses owned by this value.
unsafe impl Send for PeerAllReduce {}
unsafe impl Sync for PeerAllReduce {}

fn uncached(bytes: usize) -> Result<u64> {
    let mut ptr = std::ptr::null_mut();
    hip::check(unsafe { sys::hipExtMallocWithFlags(&mut ptr, bytes, sys::hipDeviceMallocUncached) })
        .w()?;
    hip::check(unsafe { sys::hipMemsetAsync(ptr, 0, bytes, std::ptr::null_mut()) }).w()?;
    hip::check(unsafe { sys::hipDeviceSynchronize() }).w()?;
    Ok(ptr as u64)
}

fn ipc_handle(ptr: u64) -> Result<[u8; 64]> {
    let mut handle = sys::hipIpcMemHandle_t { reserved: [0; 64] };
    hip::check(unsafe { sys::hipIpcGetMemHandle(&mut handle, ptr as *mut c_void) }).w()?;
    Ok(handle.reserved)
}

fn ipc_open(bytes: &[u8]) -> Result<u64> {
    let mut handle = sys::hipIpcMemHandle_t { reserved: [0; 64] };
    handle.reserved.copy_from_slice(bytes);
    let mut ptr = std::ptr::null_mut();
    hip::check(unsafe {
        sys::hipIpcOpenMemHandle(&mut ptr, handle, sys::hipIpcMemLazyEnablePeerAccess)
    })
    .w()?;
    Ok(ptr as u64)
}

impl PeerAllReduceSetup {
    /// Allocates this rank's receive buffer for reductions of up to `max_elems` bf16
    /// elements, and returns the handles the other ranks need to open it.
    pub fn new(
        device: &RocmDevice,
        rank: usize,
        world: usize,
        max_elems: usize,
    ) -> Result<(Self, [u8; HANDLE_BYTES])> {
        if world < 2 || world > MAX_RANKS || rank >= world || max_elems % 8 != 0 {
            crate::bail!("peer all-reduce: invalid rank {rank} of {world} for {max_elems} elements");
        }
        hip::bind_device(device.ordinal()).w()?;
        let base = uncached(FLAGS_BYTES + 2 * world * max_elems * 2)?;
        let handle = ipc_handle(base)?;
        let setup = Self {
            device: device.clone(),
            rank,
            world,
            max_elems,
            base,
        };
        Ok((setup, handle))
    }

    /// Connects ranks that all live in this process (one per GPU), through peer access instead
    /// of IPC; `setups[r]` is rank `r`'s.
    pub fn connect_in_process(setups: Vec<Self>) -> Result<Vec<PeerAllReduce>> {
        let bases = setups.iter().map(|s| s.base).collect::<Vec<_>>();
        for s in &setups {
            hip::bind_device(s.device.ordinal()).w()?;
            for other in &setups {
                if other.device.ordinal() != s.device.ordinal() {
                    let status =
                        unsafe { sys::hipDeviceEnablePeerAccess(other.device.ordinal() as i32, 0) };
                    // Already enabled is fine; the sticky error must not leak into later calls.
                    unsafe { sys::hipGetLastError() };
                    let _ = status;
                }
            }
        }
        setups
            .into_iter()
            .map(|s| s.finish(&bases, Vec::new()))
            .collect()
    }

    fn finish(self, bases: &[u64], opened: Vec<u64>) -> Result<PeerAllReduce> {
        hip::bind_device(self.device.ordinal()).w()?;
        // The kernel takes every rank's data pointer, then every rank's flags pointer.
        let table = bases
            .iter()
            .map(|b| b + FLAGS_BYTES as u64)
            .chain(bases.iter().copied())
            .collect::<Vec<_>>();
        let peers = self.device.clone_htod(&table)?;
        let epoch = self.device.alloc_zeros::<i32>(1)?;
        let counters = self.device.alloc_zeros::<u32>(2)?;
        Ok(PeerAllReduce {
            device: self.device.clone(),
            rank: self.rank,
            world: self.world,
            max_elems: self.max_elems,
            base: self.base,
            opened,
            peers,
            epoch,
            counters,
        })
    }

    /// Opens the other ranks' buffers; `handles[r]` is what rank `r`'s [`Self::new`] returned.
    pub fn connect(self, handles: &[[u8; HANDLE_BYTES]]) -> Result<PeerAllReduce> {
        if handles.len() != self.world {
            crate::bail!("peer all-reduce: {} handles for {} ranks", handles.len(), self.world);
        }
        hip::bind_device(self.device.ordinal()).w()?;
        let mut bases = vec![0u64; self.world];
        let mut opened = Vec::new();
        for (r, handle) in handles.iter().enumerate() {
            bases[r] = if r == self.rank {
                self.base
            } else {
                let base = ipc_open(handle)?;
                opened.push(base);
                base
            };
        }
        self.finish(&bases, opened)
    }
}

/// The device-side pieces of a [`PeerAllReduce`], for kernels that run the exchange
/// themselves (fused with the work that consumes the reduced values). The buffers follow the
/// layout `peer_allreduce.cu` describes; `epoch` must advance by one per exchange.
#[derive(Clone, Copy, Debug)]
pub struct PeerRawHandles {
    /// `world` data pointers, then `world` flag pointers.
    pub peers: u64,
    pub epoch: u64,
    /// Two u32 counters (tickets, rows pushed), zero at creation.
    pub counters: u64,
    pub rank: u32,
    pub world: u32,
    pub max_elems: u32,
}

impl PeerAllReduce {
    pub fn raw_handles(&self) -> PeerRawHandles {
        PeerRawHandles {
            peers: self.peers.device_ptr(),
            epoch: self.epoch.device_ptr(),
            counters: self.counters.device_ptr(),
            rank: self.rank as u32,
            world: self.world as u32,
            max_elems: self.max_elems as u32,
        }
    }

    pub fn max_elems(&self) -> usize {
        self.max_elems
    }

    /// Whether `xs` is a reduction this path takes.
    pub fn supports(&self, xs: &Tensor) -> bool {
        xs.dtype() == DType::BF16
            && xs.elem_count() <= self.max_elems
            && xs.elem_count() % 8 == 0
            && xs.device().same_device(&crate::Device::Rocm(self.device.clone()))
    }

    /// Whether `xs` is a gather this path takes: the gather kernel runs on one workgroup, so
    /// it stops at [`GATHER_MAX_ELEMS`] however large the receive buffer is.
    pub fn supports_gather(&self, xs: &Tensor) -> bool {
        self.supports(xs) && xs.elem_count() <= GATHER_MAX_ELEMS
    }

    /// The sum of `xs` over every rank; every rank must call this with the same shape.
    /// Reductions from [`WIDE_MIN_ELEMS`] up run on the whole GPU (a stripe per workgroup, so
    /// the pushes reach the link's bandwidth); smaller ones on one workgroup, where the cost
    /// is latency.
    pub fn all_reduce(&self, xs: &Tensor) -> Result<Tensor> {
        if xs.elem_count() >= WIDE_MIN_ELEMS {
            self.exchange(xs, "peer_allreduce_bf16_wide", 1)
        } else {
            self.exchange(xs, "peer_allreduce_bf16", 1)
        }
    }

    /// Every rank's `xs` stacked rank-major: `[world, ..xs.dims()]`.
    pub fn all_gather(&self, xs: &Tensor) -> Result<Tensor> {
        self.exchange(xs, "peer_allgather_bf16", self.world)
    }

    /// Runs `kernel` over `xs`; its output holds `out_mult` copies of `xs`' shape.
    fn exchange(&self, xs: &Tensor, kernel: &str, out_mult: usize) -> Result<Tensor> {
        if !self.supports(xs) {
            crate::bail!("peer collective does not take {:?} {:?}", xs.dtype(), xs.shape());
        }
        let xs = xs.contiguous()?;
        let n = xs.elem_count();
        let (guard, layout) = xs.storage_and_layout();
        let Storage::Rocm(storage) = &*guard else {
            crate::bail!("peer all-reduce needs a rocm tensor");
        };
        let x = storage
            .as_rocm_slice::<half::bf16>()?
            .slice(layout.start_offset()..)
            .device_ptr();
        if x % 16 != 0 {
            crate::bail!("peer all-reduce needs 16-byte aligned input");
        }
        let out = unsafe { self.device.alloc::<half::bf16>(n * out_mult)? };
        let y = out.device_ptr();
        let (n, max_elems, rank, world) = (
            n as i32,
            self.max_elems as i32,
            self.rank as i32,
            self.world as i32,
        );
        let wide = kernel.ends_with("_wide");
        // Wide: 16 bytes per lane per pass, enough workgroups to keep the link busy; the
        // reduce launch reads local memory only and polls the flags first, so fewer.
        let vectors = (n as usize).div_ceil(8);
        let wide_cfg = |max_blocks: usize| LaunchConfig {
            grid_dim: (vectors.div_ceil(WIDE_BLOCK).clamp(1, max_blocks) as u32, 1, 1),
            block_dim: (WIDE_BLOCK as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let cfg = if wide {
            wide_cfg(WIDE_REDUCE_MAX_BLOCKS)
        } else {
            LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (1024, 1, 1),
                shared_mem_bytes: 0,
            }
        };
        if wide {
            // The push runs as its own launch so that no workgroup waits for a peer while
            // workgroups of its own grid are still queued (see peer_allreduce.cu).
            let push = self
                .device
                .get_or_load_func("peer_push_bf16_wide", &kernels::PEER_ALLREDUCE)?;
            let mut builder = push.builder();
            builder.arg(&x);
            builder.arg(&n);
            builder.arg(&max_elems);
            builder.arg(&self.peers);
            builder.arg(&rank);
            builder.arg(&world);
            builder.arg(&self.epoch);
            builder.arg(&self.counters);
            // SAFETY: ffi; the peers' buffers hold `max_elems` elements per slot and n <= max_elems.
            unsafe { builder.launch(wide_cfg(WIDE_MAX_BLOCKS)) }.w()?;
        }
        let func = self.device.get_or_load_func(kernel, &kernels::PEER_ALLREDUCE)?;
        let mut builder = func.builder();
        builder.arg(&x);
        builder.arg(&y);
        builder.arg(&n);
        builder.arg(&max_elems);
        builder.arg(&self.peers);
        builder.arg(&rank);
        builder.arg(&world);
        builder.arg(&self.epoch);
        if wide {
            builder.arg(&self.counters);
        }
        // SAFETY: ffi; the peers' buffers hold `max_elems` elements per slot and n <= max_elems.
        unsafe { builder.launch(cfg) }.w()?;
        drop(guard);
        let storage = Storage::Rocm(super::RocmStorage::wrap_rocm_slice(out, self.device.clone()));
        let mut dims = xs.dims().to_vec();
        if out_mult > 1 {
            dims.insert(0, out_mult);
        }
        Ok(Tensor::from_storage(
            storage,
            dims,
            crate::op::BackpropOp::none(),
            false,
        ))
    }
}

impl Drop for PeerAllReduce {
    fn drop(&mut self) {
        if hip::bind_device(self.device.ordinal()).is_ok() {
            unsafe {
                for &ptr in &self.opened {
                    sys::hipIpcCloseMemHandle(ptr as *mut c_void);
                }
                sys::hipFree(self.base as *mut c_void);
            }
        }
    }
}
