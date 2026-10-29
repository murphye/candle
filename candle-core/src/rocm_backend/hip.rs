//! A small safe layer over the HIP runtime, shaped after the parts of `cudarc`
//! used by the CUDA backend so that kernel launch code can be shared closely.
//!
//! All work for a device is submitted on a single non-blocking stream.
//! Allocations are stream ordered (`hipMallocAsync`/`hipFreeAsync`), so
//! dropping a buffer never synchronizes the device. Small freed buffers are
//! kept in a per-stream cache and handed out again by later allocations, which
//! is safe because every user of a buffer is ordered on that same stream.
pub use candle_rocm_kernels::ffi as sys;
use std::ffi::{c_void, CStr, CString};
use std::marker::PhantomData;
use std::ops::{Bound, RangeBounds};
use std::sync::Arc;

/// An error code returned by the HIP runtime.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct HipError(pub sys::hipError_t);

impl HipError {
    pub fn name(&self) -> &'static str {
        let ptr = unsafe { sys::hipGetErrorName(self.0) };
        if ptr.is_null() {
            return "unknown";
        }
        unsafe { CStr::from_ptr(ptr) }.to_str().unwrap_or("unknown")
    }

    pub fn description(&self) -> &'static str {
        let ptr = unsafe { sys::hipGetErrorString(self.0) };
        if ptr.is_null() {
            return "";
        }
        unsafe { CStr::from_ptr(ptr) }.to_str().unwrap_or("")
    }
}

impl std::fmt::Debug for HipError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "HipError({}, {})", self.0, self.name())
    }
}

impl std::fmt::Display for HipError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.name(), self.description())
    }
}

impl std::error::Error for HipError {}

/// Converts a HIP return code into a `Result`.
pub fn check(code: sys::hipError_t) -> Result<(), HipError> {
    if code == sys::hipSuccess {
        Ok(())
    } else {
        Err(HipError(code))
    }
}

/// An error status returned by rocBLAS.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct RocblasError(pub sys::rocblas_status);

impl std::fmt::Debug for RocblasError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "RocblasError({self})")
    }
}

impl std::fmt::Display for RocblasError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let ptr = unsafe { sys::rocblas_status_to_string(self.0) };
        if ptr.is_null() {
            write!(f, "rocblas status {}", self.0)
        } else {
            let s = unsafe { CStr::from_ptr(ptr) }.to_string_lossy();
            write!(f, "{s} ({})", self.0)
        }
    }
}

impl std::error::Error for RocblasError {}

pub fn check_rocblas(status: sys::rocblas_status) -> Result<(), RocblasError> {
    if status == sys::rocblas_status_success {
        Ok(())
    } else {
        Err(RocblasError(status))
    }
}

thread_local! {
    static CURRENT_DEVICE: std::cell::Cell<i32> = const { std::cell::Cell::new(-1) };
}

/// Set by an exit handler that runs before the HIP runtime's own teardown. Threads
/// still dropping device objects at that point must not call into the runtime.
static EXITING: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

extern "C" fn mark_exiting() {
    EXITING.store(true, std::sync::atomic::Ordering::Release);
}

/// Makes `ordinal` the current HIP device for this thread. HIP resolves streams,
/// allocations and modules against the calling thread's current device, so
/// every entry point that touches a device goes through here first. Fails once
/// the process is exiting, which turns the HIP calls in `Drop` impls into no-ops.
pub fn bind_device(ordinal: usize) -> Result<(), HipError> {
    static REGISTER_EXIT: std::sync::Once = std::sync::Once::new();
    if EXITING.load(std::sync::atomic::Ordering::Acquire) {
        return Err(HipError(sys::hipErrorDeinitialized));
    }
    let ordinal = ordinal as i32;
    CURRENT_DEVICE.with(|current| {
        if current.get() != ordinal {
            check(unsafe { sys::hipSetDevice(ordinal) })?;
            current.set(ordinal);
            // Registered after the runtime initialized, so it runs before the runtime's teardown.
            REGISTER_EXIT.call_once(|| unsafe {
                libc::atexit(mark_exiting);
            });
        }
        Ok(())
    })
}

pub fn device_count() -> Result<usize, HipError> {
    let mut count = 0;
    check(unsafe { sys::hipGetDeviceCount(&mut count) })?;
    Ok(count as usize)
}

pub fn device_name(ordinal: usize) -> Result<String, HipError> {
    let mut buf = [0 as std::ffi::c_char; 256];
    check(unsafe { sys::hipDeviceGetName(buf.as_mut_ptr(), buf.len() as i32, ordinal as i32) })?;
    Ok(unsafe { CStr::from_ptr(buf.as_ptr()) }
        .to_string_lossy()
        .into_owned())
}

/// Whether the device is an integrated GPU sharing system memory, such as the
/// graphics core of a Ryzen CPU.
pub fn device_is_integrated(ordinal: usize) -> Result<bool, HipError> {
    let mut value = 0;
    check(unsafe {
        sys::hipDeviceGetAttribute(
            &mut value,
            sys::hipDeviceAttributeIntegrated,
            ordinal as i32,
        )
    })?;
    Ok(value != 0)
}

/// Returns `(free, total)` device memory in bytes.
pub fn mem_get_info(ordinal: usize) -> Result<(usize, usize), HipError> {
    bind_device(ordinal)?;
    let (mut free, mut total) = (0, 0);
    check(unsafe { sys::hipMemGetInfo(&mut free, &mut total) })?;
    Ok((free, total))
}

/// Marker for types that can be passed by value to a kernel or copied to device
/// memory byte for byte.
///
/// # Safety
/// The type must be plain old data with the same layout as on the device.
pub unsafe trait DeviceRepr: Copy + 'static {}

/// Marker for types whose all-zero bit pattern is a valid value.
///
/// # Safety
/// Zeroed memory must be a valid instance of the type.
pub unsafe trait ValidAsZeroBits {}

macro_rules! device_repr {
    ($($t:ty),*) => {
        $(
            unsafe impl DeviceRepr for $t {}
            unsafe impl ValidAsZeroBits for $t {}
        )*
    };
}
device_repr!(
    bool,
    u8,
    u16,
    u32,
    u64,
    usize,
    i8,
    i16,
    i32,
    i64,
    isize,
    f32,
    f64,
    half::f16,
    half::bf16,
    float8::F8E4M3
);

/// Keeps memory freed with `hipFreeAsync` in the device's default pool instead of
/// returning it to the driver at every synchronization. With the default
/// threshold of zero, large temporary buffers are remapped on each use, which
/// costs milliseconds per allocation.
fn retain_freed_pool_memory(ordinal: usize) -> Result<(), HipError> {
    let mut pool = std::ptr::null_mut();
    check(unsafe { sys::hipDeviceGetDefaultMemPool(&mut pool, ordinal as i32) })?;
    let mut threshold = u64::MAX;
    check(unsafe {
        sys::hipMemPoolSetAttribute(
            pool,
            sys::hipMemPoolAttrReleaseThreshold,
            &mut threshold as *mut u64 as *mut c_void,
        )
    })
}

/// Pinned host buffers for small host-to-device copies. Copies from pageable
/// memory have to finish before the source may be reused, which stalls the host
/// until the stream drains. Staged copies are asynchronous: each slot records an
/// event after its copy and is only reused once that event has completed.
#[derive(Debug)]
struct Staging {
    base: *mut u8,
    events: Vec<sys::hipEvent_t>,
    next: usize,
}

const STAGING_SLOT_BYTES: usize = 256 * 1024;
const STAGING_SLOTS: usize = 32;

impl Staging {
    fn new() -> Result<Self, HipError> {
        let mut base = std::ptr::null_mut();
        check(unsafe { sys::hipHostMalloc(&mut base, STAGING_SLOT_BYTES * STAGING_SLOTS, 0) })?;
        let mut events = Vec::with_capacity(STAGING_SLOTS);
        for _ in 0..STAGING_SLOTS {
            let mut event = std::ptr::null_mut();
            check(unsafe { sys::hipEventCreateWithFlags(&mut event, sys::hipEventDisableTiming) })?;
            // Record once so the first wait on every slot returns immediately.
            events.push(event);
        }
        Ok(Self {
            base: base as *mut u8,
            events,
            next: 0,
        })
    }
}

/// Largest allocation kept in the free-buffer cache. `hipFreeAsync` costs about
/// 5us of host time, more than a kernel launch, which dominates small ops, and
/// `hipMallocAsync` of large blocks can take milliseconds even when the pool
/// holds enough free memory (the grouped MoE path reuses ~1 GB per layer).
const CACHE_MAX_BLOCK_BYTES: usize = 1536 << 20;
/// Upper bound on the bytes held by the cache of one stream.
const CACHE_MAX_TOTAL_BYTES: usize = 2 << 30;

/// The bytes actually allocated for a request: 512 byte granularity for small
/// blocks, then four size classes per power of two (at most 25% slack).
fn cached_block_bytes(bytes: usize) -> usize {
    if bytes <= 4096 {
        return bytes.div_ceil(512) * 512;
    }
    let step = (bytes.next_power_of_two() / 8).max(512);
    bytes.div_ceil(step) * step
}

/// Freed device buffers by block size, reused in stream order.
#[derive(Debug, Default)]
struct BlockCache {
    free: std::collections::HashMap<usize, Vec<u64>>,
    total_bytes: usize,
    /// Blocks that another device's stream may still read (peer copies), keyed by
    /// pointer: the events that must complete before the block can be reused.
    guards: std::collections::HashMap<u64, Vec<sys::hipEvent_t>>,
    /// Released guarded blocks waiting for their events: (ptr, block bytes, events).
    pending: Vec<(u64, usize, Vec<sys::hipEvent_t>)>,
}

/// Device buffers allocated while a stream is captured into a graph. They are never
/// handed to eager work: a replay rewrites them, so they only return to this pool, which
/// every graph of the stream shares (replays are ordered on the stream, and a graph's
/// intermediates are dead once it finishes).
#[derive(Debug, Default)]
struct GraphPool {
    free: std::collections::HashMap<usize, Vec<u64>>,
    owned: std::collections::HashSet<u64>,
    /// Blocks written from the host during a capture. Graphs read them on every replay,
    /// so they never return to the free lists.
    frozen: std::collections::HashSet<u64>,
    bytes: usize,
}

/// A stream on a given device. All candle work for a device is ordered on one.
#[derive(Debug)]
pub struct HipStream {
    raw: sys::hipStream_t,
    ordinal: usize,
    staging: std::sync::Mutex<Option<Staging>>,
    cache: std::sync::Mutex<BlockCache>,
    /// Counts the operations that may write device memory: allocations, copies,
    /// launches and every hand-out of the raw stream to foreign code.
    epoch: std::sync::atomic::AtomicU64,
    derived: std::sync::Mutex<Option<Derived>>,
    derived_ids: std::sync::atomic::AtomicU64,
    capturing: std::sync::atomic::AtomicBool,
    has_graph_pool: std::sync::atomic::AtomicBool,
    graph_pool: std::sync::Mutex<GraphPool>,
    /// Eager buffers released during a capture, freed once it ends: a free issued
    /// mid-capture would be recorded into the graph instead.
    deferred_frees: std::sync::Mutex<Vec<u64>>,
    /// Carries host uploads made during a capture; created on first use.
    upload_stream: std::sync::OnceLock<usize>,
}

/// A buffer computed from another buffer (e.g. activations quantized for a
/// quantized matmul), reusable while the stream's epoch is unchanged.
/// The buffer is held as a raw allocation: a `HipSlice` would keep its stream alive.
#[derive(Debug)]
struct Derived {
    epoch: u64,
    key: (u64, usize, usize),
    ptr: u64,
    bytes: usize,
    cached: bool,
    /// Distinguishes this derived buffer from a later one with the same key (device
    /// addresses are reused), for renewals that must name the buffer they vouch for.
    id: u64,
}

// SAFETY: HIP streams may be used from any thread once created.
unsafe impl Send for HipStream {}
unsafe impl Sync for HipStream {}

impl HipStream {
    pub fn new(ordinal: usize) -> Result<Arc<Self>, HipError> {
        bind_device(ordinal)?;
        retain_freed_pool_memory(ordinal)?;
        let mut raw = std::ptr::null_mut();
        check(unsafe { sys::hipStreamCreateWithFlags(&mut raw, sys::hipStreamNonBlocking) })?;
        let staging = Staging::new()?;
        for &event in &staging.events {
            check(unsafe { sys::hipEventRecord(event, raw) })?;
        }
        Ok(Arc::new(Self {
            raw,
            ordinal,
            staging: std::sync::Mutex::new(Some(staging)),
            cache: std::sync::Mutex::new(BlockCache::default()),
            epoch: std::sync::atomic::AtomicU64::new(0),
            derived: std::sync::Mutex::new(None),
            derived_ids: std::sync::atomic::AtomicU64::new(1),
            capturing: std::sync::atomic::AtomicBool::new(false),
            has_graph_pool: std::sync::atomic::AtomicBool::new(false),
            graph_pool: std::sync::Mutex::new(GraphPool::default()),
            deferred_frees: std::sync::Mutex::new(Vec::new()),
            upload_stream: std::sync::OnceLock::new(),
        }))
    }

    /// The raw stream. Callers may enqueue anything on it, so this counts as a write.
    pub fn raw(&self) -> sys::hipStream_t {
        self.touch();
        self.raw
    }

    fn touch(&self) {
        self.epoch.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    fn epoch(&self) -> u64 {
        self.epoch.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Device address of the buffer derived from `key` by the last call to
    /// [`Self::set_derived`], if nothing has been allocated, copied or launched
    /// on this stream since then.
    pub fn derived(&self, key: (u64, usize, usize)) -> Option<u64> {
        let derived = self.derived.lock().unwrap();
        derived
            .as_ref()
            .filter(|d| d.key == key && d.epoch == self.epoch())
            .map(|d| d.ptr)
    }

    /// Remembers `data` as derived from `key`, valid until the next write to the stream.
    /// Call it after enqueuing the work that reads `data`. Returns the buffer's identity for
    /// [`Self::renew_derived_id`].
    pub fn set_derived(&self, key: (u64, usize, usize), data: HipSlice<u8>) -> u64 {
        let data = std::mem::ManuallyDrop::new(data);
        // SAFETY: the stream handle is read out once and dropped; `data` is never used again.
        drop(unsafe { std::ptr::read(&data.stream) });
        let id = self
            .derived_ids
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let new = Derived {
            epoch: self.epoch(),
            key,
            ptr: data.ptr,
            bytes: data.len,
            cached: data.cached,
            id,
        };
        let old = self.derived.lock().unwrap().replace(new);
        if let Some(old) = old {
            self.free_derived(old);
        }
        id
    }

    /// [`Self::renew_derived`] for the buffer `set_derived` returned `id` for, if it is still
    /// the current one: a later buffer under the same key (a reused address) is left alone.
    pub fn renew_derived_id(&self, id: u64) -> bool {
        match self.derived.lock().unwrap().as_mut() {
            Some(d) if d.id == id => {
                d.epoch = self.epoch();
                true
            }
            _ => false,
        }
    }

    fn free_derived(&self, d: Derived) {
        if d.cached {
            self.release(d.ptr, cached_block_bytes(d.bytes));
        } else if self.is_capturing_graph() {
            self.deferred_frees.lock().unwrap().push(d.ptr);
        } else if self.bind().is_ok() {
            unsafe { sys::hipFreeAsync(d.ptr as *mut c_void, self.raw) };
        }
    }

    /// Extends the derived buffer's validity over the writes made since it was set;
    /// the caller vouches that none of them touched the source buffer.
    pub fn renew_derived(&self) {
        if let Some(d) = self.derived.lock().unwrap().as_mut() {
            d.epoch = self.epoch();
        }
    }

    pub fn ordinal(&self) -> usize {
        self.ordinal
    }

    pub fn bind(&self) -> Result<(), HipError> {
        bind_device(self.ordinal)
    }

    pub fn synchronize(&self) -> Result<(), HipError> {
        trace_transfer("sync", 0);
        self.reject_while_capturing()?;
        self.bind()?;
        check(unsafe { sys::hipStreamSynchronize(self.raw) })
    }

    pub fn is_capturing(&self) -> Result<bool, HipError> {
        let mut status = sys::hipStreamCaptureStatusNone;
        check(unsafe { sys::hipStreamIsCapturing(self.raw, &mut status) })?;
        Ok(status != sys::hipStreamCaptureStatusNone)
    }

    /// # Safety
    /// The contents of the returned buffer are uninitialized.
    pub unsafe fn alloc<T: DeviceRepr>(
        self: &Arc<Self>,
        len: usize,
    ) -> Result<HipSlice<T>, HipError> {
        self.touch();
        let bytes = len * std::mem::size_of::<T>();
        let mut ptr = 0;
        let mut cached = false;
        if bytes > 0 && self.is_capturing_graph() {
            return Ok(HipSlice {
                ptr: self.graph_alloc(cached_block_bytes(bytes))?,
                len,
                cached: true,
                stream: self.clone(),
                marker: PhantomData,
            });
        }
        if bytes > 0 {
            if bytes <= CACHE_MAX_BLOCK_BYTES {
                let block = cached_block_bytes(bytes);
                ptr = self.take_cached(block);
                if ptr == 0 {
                    ptr = self.malloc(block)?;
                }
                cached = true;
            } else {
                ptr = self.malloc(bytes)?;
            }
        }
        Ok(HipSlice {
            ptr,
            len,
            cached,
            stream: self.clone(),
            marker: PhantomData,
        })
    }

    fn is_capturing_graph(&self) -> bool {
        self.capturing.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// A block from the graph pool, allocated outside the stream's order: a stream-ordered
    /// allocation would become a node of the graph being captured.
    fn graph_alloc(&self, block: usize) -> Result<u64, HipError> {
        let mut pool = self.graph_pool.lock().unwrap();
        if let Some(ptr) = pool.free.get_mut(&block).and_then(Vec::pop) {
            return Ok(ptr);
        }
        self.bind()?;
        let mut ptr = std::ptr::null_mut();
        check(unsafe { sys::hipMalloc(&mut ptr, block) })?;
        let ptr = ptr as u64;
        pool.owned.insert(ptr);
        pool.bytes += block;
        self.has_graph_pool
            .store(true, std::sync::atomic::Ordering::Relaxed);
        Ok(ptr)
    }

    /// A new graph-pool block for data uploaded during a capture: never reused.
    fn graph_alloc_frozen(&self, block: usize) -> Result<u64, HipError> {
        self.bind()?;
        let mut ptr = std::ptr::null_mut();
        check(unsafe { sys::hipMalloc(&mut ptr, block) })?;
        let ptr = ptr as u64;
        let mut pool = self.graph_pool.lock().unwrap();
        pool.owned.insert(ptr);
        pool.frozen.insert(ptr);
        pool.bytes += block;
        self.has_graph_pool
            .store(true, std::sync::atomic::Ordering::Relaxed);
        Ok(ptr)
    }

    /// Returns `ptr` to the graph pool if it came from there.
    fn release_to_graph_pool(&self, ptr: u64, block: usize) -> bool {
        if !self.has_graph_pool.load(std::sync::atomic::Ordering::Relaxed) {
            return false;
        }
        let mut pool = self.graph_pool.lock().unwrap();
        if !pool.owned.contains(&ptr) {
            return false;
        }
        if !pool.frozen.contains(&ptr) {
            pool.free.entry(block).or_default().push(ptr);
        }
        true
    }

    /// Bytes held by the graph pool.
    pub fn graph_pool_bytes(&self) -> usize {
        self.graph_pool.lock().unwrap().bytes
    }

    /// Starts capturing the work enqueued on this stream into a graph. Until
    /// [`Self::end_capture`], allocations come from the graph pool and host copies fail.
    pub fn begin_capture(&self) -> Result<(), HipError> {
        // Derived buffers computed eagerly must not stand in for work the graph has to record.
        self.touch();
        self.bind()?;
        check(unsafe { sys::hipStreamBeginCapture(self.raw, sys::hipStreamCaptureModeRelaxed) })?;
        self.capturing
            .store(true, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }

    /// Ends the capture started by [`Self::begin_capture`] and instantiates it.
    pub fn end_capture(&self) -> Result<HipGraph, HipError> {
        self.capturing
            .store(false, std::sync::atomic::Ordering::Relaxed);
        self.touch();
        self.bind()?;
        let mut graph = std::ptr::null_mut();
        let ended = check(unsafe { sys::hipStreamEndCapture(self.raw, &mut graph) });
        for ptr in std::mem::take(&mut *self.deferred_frees.lock().unwrap()) {
            unsafe { sys::hipFreeAsync(ptr as *mut c_void, self.raw) };
        }
        ended?;
        let mut exec = std::ptr::null_mut();
        let instantiated = check(unsafe { sys::hipGraphInstantiateWithFlags(&mut exec, graph, 0) });
        if let Err(e) = instantiated {
            unsafe { sys::hipGraphDestroy(graph) };
            return Err(e);
        }
        Ok(HipGraph {
            graph,
            exec,
            ordinal: self.ordinal,
        })
    }

    /// A second stream on this device for uploads that must not join a capture.
    fn upload_stream(&self) -> Result<sys::hipStream_t, HipError> {
        if let Some(&raw) = self.upload_stream.get() {
            return Ok(raw as sys::hipStream_t);
        }
        let mut raw = std::ptr::null_mut();
        check(unsafe { sys::hipStreamCreateWithFlags(&mut raw, sys::hipStreamNonBlocking) })?;
        let raw = *self.upload_stream.get_or_init(|| raw as usize);
        Ok(raw as sys::hipStream_t)
    }

    fn reject_while_capturing(&self) -> Result<(), HipError> {
        if self.is_capturing_graph() {
            return Err(capture_unsupported());
        }
        Ok(())
    }

    fn take_cached(&self, block: usize) -> u64 {
        let mut cache = self.cache.lock().unwrap();
        if !cache.pending.is_empty() {
            Self::reap_pending(&mut cache);
        }
        match cache.free.get_mut(&block).and_then(Vec::pop) {
            Some(ptr) => {
                cache.total_bytes -= block;
                ptr
            }
            None => 0,
        }
    }

    /// Moves pending blocks whose guarding events have completed to the free lists.
    fn reap_pending(cache: &mut BlockCache) {
        let mut i = 0;
        while i < cache.pending.len() {
            let done = cache.pending[i]
                .2
                .iter()
                .all(|&e| unsafe { sys::hipEventQuery(e) } == sys::hipSuccess);
            if done {
                let (ptr, block, events) = cache.pending.swap_remove(i);
                for e in events {
                    unsafe { sys::hipEventDestroy(e) };
                }
                cache.total_bytes += block;
                cache.free.entry(block).or_default().push(ptr);
            } else {
                i += 1;
            }
        }
    }

    /// Keeps `ptr` out of reuse until `event` (recorded on another stream that reads
    /// it) has completed, without making this stream wait for that stream.
    pub(crate) fn guard_until(&self, ptr: u64, event: sys::hipEvent_t) {
        self.cache.lock().unwrap().guards.entry(ptr).or_default().push(event);
    }

    /// Releases an uncached block: stream-ordered free, after any guarding events.
    fn release_uncached(&self, ptr: u64) {
        if self.is_capturing_graph() {
            self.deferred_frees.lock().unwrap().push(ptr);
            return;
        }
        let events = self.cache.lock().unwrap().guards.remove(&ptr);
        if self.bind().is_ok() {
            for e in events.into_iter().flatten() {
                unsafe {
                    sys::hipStreamWaitEvent(self.raw, e, 0);
                    sys::hipEventDestroy(e);
                }
            }
            unsafe { sys::hipFreeAsync(ptr as *mut c_void, self.raw) };
        }
    }

    /// Returns a block to the cache, or frees it when the cache is full.
    fn release(&self, ptr: u64, block: usize) {
        if self.release_to_graph_pool(ptr, block) {
            return;
        }
        {
            let mut cache = self.cache.lock().unwrap();
            if let Some(events) = cache.guards.remove(&ptr) {
                cache.pending.push((ptr, block, events));
                return;
            }
            if cache.total_bytes + block <= CACHE_MAX_TOTAL_BYTES {
                cache.total_bytes += block;
                cache.free.entry(block).or_default().push(ptr);
                return;
            }
        }
        if self.is_capturing_graph() {
            self.deferred_frees.lock().unwrap().push(ptr);
            return;
        }
        if self.bind().is_ok() {
            unsafe { sys::hipFreeAsync(ptr as *mut c_void, self.raw) };
        }
    }

    /// Frees every cached block back to the memory pool.
    pub fn release_cached(&self) -> Result<(), HipError> {
        let blocks: Vec<u64> = {
            let mut cache = self.cache.lock().unwrap();
            // Pending blocks are freed too once their readers are done.
            for (ptr, _, events) in std::mem::take(&mut cache.pending) {
                for e in events {
                    unsafe {
                        sys::hipEventSynchronize(e);
                        sys::hipEventDestroy(e);
                    }
                }
                cache.free.entry(0).or_default().push(ptr);
            }
            cache.total_bytes = 0;
            cache.free.drain().flat_map(|(_, ptrs)| ptrs).collect()
        };
        if blocks.is_empty() {
            return Ok(());
        }
        self.bind()?;
        for ptr in blocks {
            check(unsafe { sys::hipFreeAsync(ptr as *mut c_void, self.raw) })?;
        }
        Ok(())
    }

    /// Makes work submitted to `other` from now on wait for the work already
    /// submitted to this stream, without blocking the host.
    pub fn order_before(&self, other: &HipStream) -> Result<(), HipError> {
        self.bind()?;
        let mut event = std::ptr::null_mut();
        check(unsafe { sys::hipEventCreateWithFlags(&mut event, sys::hipEventDisableTiming) })?;
        check(unsafe { sys::hipEventRecord(event, self.raw) })?;
        other.bind()?;
        let waited = check(unsafe { sys::hipStreamWaitEvent(other.raw, event, 0) });
        // Destroying a recorded event is deferred by the runtime until it completes.
        self.bind()?;
        check(unsafe { sys::hipEventDestroy(event) })?;
        waited
    }

    /// `hipMallocAsync`, retrying once with the cache emptied when out of memory.
    fn malloc(&self, bytes: usize) -> Result<u64, HipError> {
        self.bind()?;
        let mut ptr = std::ptr::null_mut();
        let status = unsafe { sys::hipMallocAsync(&mut ptr, bytes, self.raw) };
        if status == sys::hipErrorOutOfMemory {
            self.release_cached()?;
            // The pool only reuses freed memory once the frees have executed.
            self.synchronize()?;
            check(unsafe { sys::hipMallocAsync(&mut ptr, bytes, self.raw) })?;
            // The failed attempt left a sticky error that kernel wrappers checking
            // hipGetLastError after their launch would report as their own.
            unsafe { sys::hipGetLastError() };
        } else {
            check(status)?;
        }
        Ok(ptr as u64)
    }

    pub fn alloc_zeros<T: DeviceRepr + ValidAsZeroBits>(
        self: &Arc<Self>,
        len: usize,
    ) -> Result<HipSlice<T>, HipError> {
        let mut slice = unsafe { self.alloc::<T>(len)? };
        self.memset_zeros(&mut slice)?;
        Ok(slice)
    }

    pub fn memset_zeros<T: DeviceRepr + ValidAsZeroBits, Dst: DevicePtrMut<T>>(
        &self,
        dst: &mut Dst,
    ) -> Result<(), HipError> {
        let bytes = dst.len() * std::mem::size_of::<T>();
        if bytes == 0 {
            return Ok(());
        }
        self.bind()?;
        self.touch();
        check(unsafe { sys::hipMemsetAsync(dst.device_ptr() as *mut c_void, 0, bytes, self.raw) })
    }

    pub fn clone_htod<T: DeviceRepr>(self: &Arc<Self>, src: &[T]) -> Result<HipSlice<T>, HipError> {
        if self.is_capturing_graph() && !src.is_empty() {
            // A block captured kernels may also use would be overwritten on replay by
            // whatever ran there first; the upload gets a block of its own.
            let block = cached_block_bytes(std::mem::size_of_val(src));
            let mut dst = HipSlice {
                ptr: self.graph_alloc_frozen(block)?,
                len: src.len(),
                cached: true,
                stream: self.clone(),
                marker: PhantomData,
            };
            self.memcpy_htod(src, &mut dst)?;
            return Ok(dst);
        }
        let mut dst = unsafe { self.alloc::<T>(src.len())? };
        self.memcpy_htod(src, &mut dst)?;
        Ok(dst)
    }

    /// Copies host data to the device. The call returns once `src` has been
    /// consumed, so the host buffer may be reused immediately. Small copies go
    /// through pinned staging buffers and do not wait for the stream.
    pub fn memcpy_htod<T: DeviceRepr, Dst: DevicePtrMut<T>>(
        &self,
        src: &[T],
        dst: &mut Dst,
    ) -> Result<(), HipError> {
        self.touch();
        assert!(dst.len() >= src.len(), "memcpy_htod: destination too small");
        let bytes = std::mem::size_of_val(src);
        if bytes == 0 {
            return Ok(());
        }
        trace_transfer("htod", bytes);
        if self.is_capturing_graph() {
            // Host data uploaded during a capture is constant for the graph (shapes and
            // strides): copy it now rather than recording a copy that would replay whatever
            // the host buffer holds later. Only blocks reserved by `clone_htod` qualify:
            // captured kernels never touch them, so the copy can't be overwritten on replay.
            if !self.graph_pool.lock().unwrap().frozen.contains(&dst.device_ptr()) {
                return Err(capture_unsupported());
            }
            self.bind()?;
            let upload = self.upload_stream()?;
            check(unsafe {
                sys::hipMemcpyAsync(
                    dst.device_ptr() as *mut c_void,
                    src.as_ptr() as *const c_void,
                    bytes,
                    sys::hipMemcpyHostToDevice,
                    upload,
                )
            })?;
            check(unsafe { sys::hipStreamSynchronize(upload) })?;
            return Ok(());
        }
        self.bind()?;
        if bytes <= STAGING_SLOT_BYTES {
            let mut staging = self.staging.lock().unwrap();
            if let Some(staging) = staging.as_mut() {
                let slot = staging.next;
                staging.next = (slot + 1) % STAGING_SLOTS;
                let event = staging.events[slot];
                // Wait until the previous copy out of this slot has been consumed.
                check(unsafe { sys::hipEventSynchronize(event) })?;
                let pinned = unsafe { staging.base.add(slot * STAGING_SLOT_BYTES) };
                unsafe { std::ptr::copy_nonoverlapping(src.as_ptr() as *const u8, pinned, bytes) };
                check(unsafe {
                    sys::hipMemcpyAsync(
                        dst.device_ptr() as *mut c_void,
                        pinned as *const c_void,
                        bytes,
                        sys::hipMemcpyHostToDevice,
                        self.raw,
                    )
                })?;
                return check(unsafe { sys::hipEventRecord(event, self.raw) });
            }
        }
        check(unsafe {
            sys::hipMemcpyAsync(
                dst.device_ptr() as *mut c_void,
                src.as_ptr() as *const c_void,
                bytes,
                sys::hipMemcpyHostToDevice,
                self.raw,
            )
        })?;
        // Pageable host memory may still be read after hipMemcpyAsync returns.
        self.synchronize()
    }

    pub fn clone_dtoh<T: DeviceRepr, Src: DevicePtr<T>>(
        &self,
        src: &Src,
    ) -> Result<Vec<T>, HipError> {
        let len = src.len();
        let mut dst = Vec::<T>::with_capacity(len);
        let bytes = len * std::mem::size_of::<T>();
        if bytes > 0 {
            trace_transfer("dtoh", bytes);
            self.reject_while_capturing()?;
            self.bind()?;
            // Copy straight into the spare capacity, then expose it once written.
            check(unsafe {
                sys::hipMemcpyAsync(
                    dst.as_mut_ptr() as *mut c_void,
                    src.device_ptr() as *const c_void,
                    bytes,
                    sys::hipMemcpyDeviceToHost,
                    self.raw,
                )
            })?;
            self.synchronize()?;
        }
        // SAFETY: the synchronized copy above initialized all `len` elements.
        unsafe { dst.set_len(len) };
        Ok(dst)
    }

    pub fn memcpy_dtoh<T: DeviceRepr, Src: DevicePtr<T>>(
        &self,
        src: &Src,
        dst: &mut [T],
    ) -> Result<(), HipError> {
        assert!(dst.len() >= src.len(), "memcpy_dtoh: destination too small");
        let bytes = src.len() * std::mem::size_of::<T>();
        if bytes == 0 {
            return Ok(());
        }
        trace_transfer("dtoh", bytes);
        self.reject_while_capturing()?;
        self.bind()?;
        check(unsafe {
            sys::hipMemcpyAsync(
                dst.as_mut_ptr() as *mut c_void,
                src.device_ptr() as *const c_void,
                bytes,
                sys::hipMemcpyDeviceToHost,
                self.raw,
            )
        })?;
        self.synchronize()
    }

    pub fn memcpy_dtod<T, Src: DevicePtr<T>, Dst: DevicePtrMut<T>>(
        &self,
        src: &Src,
        dst: &mut Dst,
    ) -> Result<(), HipError> {
        self.touch();
        assert!(dst.len() >= src.len(), "memcpy_dtod: destination too small");
        let bytes = src.len() * std::mem::size_of::<T>();
        if bytes == 0 {
            return Ok(());
        }
        self.bind()?;
        check(unsafe {
            sys::hipMemcpyAsync(
                dst.device_ptr() as *mut c_void,
                src.device_ptr() as *const c_void,
                bytes,
                sys::hipMemcpyDeviceToDevice,
                self.raw,
            )
        })
    }

    /// Copies `src`, which may live on another device, into a new buffer on this
    /// stream's device.
    pub fn clone_dtod<T: DeviceRepr>(
        self: &Arc<Self>,
        src: &HipSlice<T>,
    ) -> Result<HipSlice<T>, HipError> {
        let mut dst = unsafe { self.alloc::<T>(src.len())? };
        let bytes = src.len() * std::mem::size_of::<T>();
        if bytes == 0 {
            return Ok(dst);
        }
        if src.stream.ordinal == self.ordinal {
            self.memcpy_dtod(src, &mut dst)?;
        } else {
            // Writes to src are ordered on its own stream: this stream waits for them.
            src.stream.order_before(self)?;
            self.bind()?;
            check(unsafe {
                sys::hipMemcpyPeerAsync(
                    dst.device_ptr() as *mut c_void,
                    self.ordinal as i32,
                    src.device_ptr() as *const c_void,
                    src.stream.ordinal as i32,
                    bytes,
                    self.raw,
                )
            })?;
            // Instead of making src's stream wait for this one (and with it for
            // everything queued here before the copy), keep src's block out of
            // reuse until the copy has run. The source device can then run ahead.
            let mut event = std::ptr::null_mut();
            check(unsafe { sys::hipEventCreateWithFlags(&mut event, sys::hipEventDisableTiming) })?;
            check(unsafe { sys::hipEventRecord(event, self.raw) })?;
            src.stream.guard_until(src.ptr, event);
        }
        // Leave this stream's device current: raw kernel launchers rely on it.
        self.bind()?;
        Ok(dst)
    }
}

impl Drop for HipStream {
    fn drop(&mut self) {
        if bind_device(self.ordinal).is_ok() {
            if let Some(d) = self.derived.get_mut().ok().and_then(Option::take) {
                self.free_derived(d);
            }
            let _ = self.release_cached();
            unsafe {
                sys::hipStreamSynchronize(self.raw);
                if let Some(staging) = self.staging.get_mut().ok().and_then(Option::take) {
                    for event in staging.events {
                        sys::hipEventDestroy(event);
                    }
                    sys::hipHostFree(staging.base as *mut c_void);
                }
                sys::hipStreamDestroy(self.raw);
            }
        }
    }
}

/// Read access to a contiguous region of device memory.
pub trait DevicePtr<T> {
    fn device_ptr(&self) -> u64;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Write access to a contiguous region of device memory.
pub trait DevicePtrMut<T>: DevicePtr<T> {}

/// An owned device buffer, freed on its stream when dropped.
pub struct HipSlice<T> {
    ptr: u64,
    len: usize,
    /// Allocated through the stream's block cache, with `cached_block_bytes` bytes.
    cached: bool,
    stream: Arc<HipStream>,
    marker: PhantomData<*const T>,
}

// SAFETY: the buffer is plain device memory; access is ordered by the stream.
unsafe impl<T> Send for HipSlice<T> {}
unsafe impl<T> Sync for HipSlice<T> {}

impl<T> std::fmt::Debug for HipSlice<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "HipSlice({:#x}, len={})", self.ptr, self.len)
    }
}

impl<T> Drop for HipSlice<T> {
    fn drop(&mut self) {
        if self.ptr == 0 {
            return;
        }
        if self.cached {
            let bytes = self.len * std::mem::size_of::<T>();
            self.stream.release(self.ptr, cached_block_bytes(bytes));
        } else {
            self.stream.release_uncached(self.ptr);
        }
    }
}

fn resolve_range(range: impl RangeBounds<usize>, len: usize) -> (usize, usize) {
    let start = match range.start_bound() {
        Bound::Included(&s) => s,
        Bound::Excluded(&s) => s + 1,
        Bound::Unbounded => 0,
    };
    let end = match range.end_bound() {
        Bound::Included(&e) => e + 1,
        Bound::Excluded(&e) => e,
        Bound::Unbounded => len,
    };
    assert!(
        start <= end && end <= len,
        "slice range {start}..{end} out of bounds for length {len}"
    );
    (start, end)
}

impl<T> HipSlice<T> {
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn stream(&self) -> &Arc<HipStream> {
        &self.stream
    }

    pub fn ordinal(&self) -> usize {
        self.stream.ordinal
    }

    pub fn device_ptr(&self) -> u64 {
        self.ptr
    }

    pub fn slice(&self, range: impl RangeBounds<usize>) -> HipView<'_, T> {
        let (start, end) = resolve_range(range, self.len);
        HipView {
            ptr: self.ptr + (start * std::mem::size_of::<T>()) as u64,
            len: end - start,
            marker: PhantomData,
        }
    }

    pub fn slice_mut(&mut self, range: impl RangeBounds<usize>) -> HipViewMut<'_, T> {
        let (start, end) = resolve_range(range, self.len);
        HipViewMut {
            ptr: self.ptr + (start * std::mem::size_of::<T>()) as u64,
            len: end - start,
            marker: PhantomData,
        }
    }

    pub fn as_view(&self) -> HipView<'_, T> {
        self.slice(..)
    }

    /// Takes ownership of a raw device allocation made with `hipMallocAsync` on
    /// `stream`'s device.
    ///
    /// # Safety
    /// `ptr` must point to at least `len` elements and must not be freed elsewhere.
    pub unsafe fn from_raw(ptr: u64, len: usize, stream: Arc<HipStream>) -> Self {
        Self {
            ptr,
            len,
            cached: false,
            stream,
            marker: PhantomData,
        }
    }

    /// Releases ownership of the allocation without freeing it.
    pub fn leak(self) -> u64 {
        let ptr = self.ptr;
        std::mem::forget(self);
        ptr
    }
}

impl<T: DeviceRepr> HipSlice<T> {
    pub fn try_clone(&self) -> Result<Self, HipError> {
        let mut dst = unsafe { self.stream.alloc::<T>(self.len)? };
        self.stream.memcpy_dtod(self, &mut dst)?;
        Ok(dst)
    }
}

/// Copies the buffer on its stream. Panics if the device allocation fails, like
/// cudarc's `CudaSlice`; use `try_clone` to handle the error.
impl<T: DeviceRepr> Clone for HipSlice<T> {
    fn clone(&self) -> Self {
        self.try_clone().expect("failed to clone device buffer")
    }
}

impl<T> DevicePtr<T> for HipSlice<T> {
    fn device_ptr(&self) -> u64 {
        self.ptr
    }
    fn len(&self) -> usize {
        self.len
    }
}
impl<T> DevicePtrMut<T> for HipSlice<T> {}

/// A borrowed, read-only region of a `HipSlice`.
#[derive(Clone, Copy)]
pub struct HipView<'a, T> {
    ptr: u64,
    len: usize,
    marker: PhantomData<&'a T>,
}

impl<T> HipView<'_, T> {
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn device_ptr(&self) -> u64 {
        self.ptr
    }

    pub fn slice(&self, range: impl RangeBounds<usize>) -> HipView<'_, T> {
        let (start, end) = resolve_range(range, self.len);
        HipView {
            ptr: self.ptr + (start * std::mem::size_of::<T>()) as u64,
            len: end - start,
            marker: PhantomData,
        }
    }
}

impl<T> DevicePtr<T> for HipView<'_, T> {
    fn device_ptr(&self) -> u64 {
        self.ptr
    }
    fn len(&self) -> usize {
        self.len
    }
}

/// A borrowed, writable region of a `HipSlice`.
pub struct HipViewMut<'a, T> {
    ptr: u64,
    len: usize,
    marker: PhantomData<&'a mut T>,
}

impl<T> HipViewMut<'_, T> {
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn device_ptr(&self) -> u64 {
        self.ptr
    }
}

impl<T> DevicePtr<T> for HipViewMut<'_, T> {
    fn device_ptr(&self) -> u64 {
        self.ptr
    }
    fn len(&self) -> usize {
        self.len
    }
}
impl<T> DevicePtrMut<T> for HipViewMut<'_, T> {}

/// A loaded code object.
#[derive(Debug)]
pub struct HipModule {
    raw: sys::hipModule_t,
    ordinal: usize,
}

unsafe impl Send for HipModule {}
unsafe impl Sync for HipModule {}

impl HipModule {
    pub fn load_data(ordinal: usize, image: &[u8]) -> Result<Arc<Self>, HipError> {
        bind_device(ordinal)?;
        let mut raw = std::ptr::null_mut();
        check(unsafe { sys::hipModuleLoadData(&mut raw, image.as_ptr() as *const c_void) })?;
        Ok(Arc::new(Self { raw, ordinal }))
    }

    pub fn load_function(self: &Arc<Self>, name: &str) -> Result<HipFunction, HipError> {
        bind_device(self.ordinal)?;
        let cname = CString::new(name).map_err(|_| HipError(1))?;
        let mut raw = std::ptr::null_mut();
        check(unsafe { sys::hipModuleGetFunction(&mut raw, self.raw, cname.as_ptr()) })?;
        Ok(HipFunction {
            raw,
            _module: self.clone(),
        })
    }
}

impl Drop for HipModule {
    fn drop(&mut self) {
        if bind_device(self.ordinal).is_ok() {
            unsafe { sys::hipModuleUnload(self.raw) };
        }
    }
}

/// A kernel entry point, keeping its module alive.
#[derive(Debug, Clone)]
pub struct HipFunction {
    raw: sys::hipFunction_t,
    _module: Arc<HipModule>,
}

unsafe impl Send for HipFunction {}
unsafe impl Sync for HipFunction {}

impl HipFunction {
    pub fn raw(&self) -> sys::hipFunction_t {
        self.raw
    }
}

#[derive(Clone, Copy, Debug)]
pub struct LaunchConfig {
    pub grid_dim: (u32, u32, u32),
    pub block_dim: (u32, u32, u32),
    pub shared_mem_bytes: u32,
}

impl LaunchConfig {
    /// One thread per element in blocks of 1024 threads.
    pub fn for_num_elems(n: u32) -> Self {
        const NUM_THREADS: u32 = 1024;
        Self {
            grid_dim: (n.div_ceil(NUM_THREADS), 1, 1),
            block_dim: (NUM_THREADS, 1, 1),
            shared_mem_bytes: 0,
        }
    }
}

/// A value that can be passed as a kernel argument.
///
/// # Safety
/// `param_ptr` must return a pointer to the argument's bytes as the kernel
/// expects them, valid for as long as `self` is borrowed.
pub unsafe trait KernelArg {
    fn param_ptr(&self) -> *mut c_void;
}

unsafe impl<T: DeviceRepr> KernelArg for T {
    fn param_ptr(&self) -> *mut c_void {
        self as *const T as *mut c_void
    }
}

// Device buffers are passed as their device address, which is stored in the
// buffer itself so that the pointer handed to HIP outlives the call.
unsafe impl<T> KernelArg for HipSlice<T> {
    fn param_ptr(&self) -> *mut c_void {
        &self.ptr as *const u64 as *mut c_void
    }
}

unsafe impl<T> KernelArg for HipView<'_, T> {
    fn param_ptr(&self) -> *mut c_void {
        &self.ptr as *const u64 as *mut c_void
    }
}

unsafe impl<T> KernelArg for HipViewMut<'_, T> {
    fn param_ptr(&self) -> *mut c_void {
        &self.ptr as *const u64 as *mut c_void
    }
}

/// Collects the arguments of a kernel launch.
pub struct LaunchArgs<'a> {
    func: &'a HipFunction,
    stream: &'a HipStream,
    args: Vec<*mut c_void>,
}

impl<'a> LaunchArgs<'a> {
    pub fn new(func: &'a HipFunction, stream: &'a HipStream) -> Self {
        Self {
            func,
            stream,
            args: Vec::with_capacity(16),
        }
    }

    pub fn arg<A: KernelArg + ?Sized>(&mut self, arg: &'a A) -> &mut Self {
        self.args.push(arg.param_ptr());
        self
    }

    /// # Safety
    /// The arguments must match the kernel's signature and the launch must not
    /// access memory out of bounds.
    pub unsafe fn launch(mut self, cfg: LaunchConfig) -> Result<(), HipError> {
        let (gx, gy, gz) = cfg.grid_dim;
        if gx == 0 || gy == 0 || gz == 0 {
            return Ok(());
        }
        let (bx, by, bz) = cfg.block_dim;
        self.stream.bind()?;
        self.stream.touch();
        check(sys::hipModuleLaunchKernel(
            self.func.raw,
            gx,
            gy,
            gz,
            bx,
            by,
            bz,
            cfg.shared_mem_bytes,
            self.stream.raw,
            self.args.as_mut_ptr(),
            std::ptr::null_mut(),
        ))
    }
}

/// A rocBLAS handle bound to a stream.
#[derive(Debug)]
pub struct RocBlas {
    handle: sys::rocblas_handle,
    stream: Arc<HipStream>,
}

unsafe impl Send for RocBlas {}
unsafe impl Sync for RocBlas {}

impl RocBlas {
    pub fn new(stream: Arc<HipStream>) -> Result<Self, RocblasError> {
        stream.bind().map_err(|_| RocblasError(-1))?;
        let mut handle = std::ptr::null_mut();
        check_rocblas(unsafe { sys::rocblas_create_handle(&mut handle) })?;
        check_rocblas(unsafe { sys::rocblas_set_stream(handle, stream.raw()) })?;
        Ok(Self { handle, stream })
    }

    pub fn handle(&self) -> sys::rocblas_handle {
        self.handle
    }

    pub fn stream(&self) -> &Arc<HipStream> {
        &self.stream
    }
}

impl Drop for RocBlas {
    fn drop(&mut self) {
        if self.stream.bind().is_ok() {
            unsafe { sys::rocblas_destroy_handle(self.handle) };
        }
    }
}

/// With `CANDLE_ROCM_TRACE_TRANSFERS` set, prints the first mistral.rs frame of every
/// host transfer, to audit what a forward pass exchanges with the host.
pub fn trace_transfer(kind: &str, bytes: usize) {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*ON.get_or_init(|| std::env::var_os("CANDLE_ROCM_TRACE_TRANSFERS").is_some()) {
        return;
    }
    let bt = std::backtrace::Backtrace::force_capture().to_string();
    let site = bt
        .lines()
        .filter(|l| l.contains("mistralrs_core::") || l.contains("mistralrs_quant::"))
        .take(2)
        .map(|l| l.trim().split_once(": ").map_or(l.trim(), |(_, f)| f).to_string())
        .collect::<Vec<_>>()
        .join(" <- ");
    eprintln!("TRANSFER {kind} {bytes} {site}");
}

/// The error for an operation a capture can't record. With `CANDLE_ROCM_CAPTURE_TRACE`
/// set, the call site's backtrace is printed, to find what keeps a forward from capturing.
fn capture_unsupported() -> HipError {
    if std::env::var_os("CANDLE_ROCM_CAPTURE_TRACE").is_some() {
        eprintln!(
            "operation not permitted while capturing:\n{}",
            std::backtrace::Backtrace::force_capture()
        );
    }
    HipError(sys::hipErrorStreamCaptureUnsupported)
}

/// An instantiated graph captured from a [`HipStream`].
#[derive(Debug)]
pub struct HipGraph {
    graph: sys::hipGraph_t,
    exec: sys::hipGraphExec_t,
    ordinal: usize,
}

// SAFETY: graph handles may be used from any thread.
unsafe impl Send for HipGraph {}
unsafe impl Sync for HipGraph {}

impl HipGraph {
    /// Replays the graph on `stream`, which must belong to the device it was captured on.
    pub fn launch(&self, stream: &HipStream) -> Result<(), HipError> {
        stream.touch();
        stream.bind()?;
        check(unsafe { sys::hipGraphLaunch(self.exec, stream.raw) })
    }
}

impl Drop for HipGraph {
    fn drop(&mut self) {
        if bind_device(self.ordinal).is_ok() {
            unsafe {
                sys::hipGraphExecDestroy(self.exec);
                sys::hipGraphDestroy(self.graph);
            }
        }
    }
}
