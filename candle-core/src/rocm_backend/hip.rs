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

/// Makes `ordinal` the current HIP device for this thread. HIP resolves streams,
/// allocations and modules against the calling thread's current device, so
/// every entry point that touches a device goes through here first.
pub fn bind_device(ordinal: usize) -> Result<(), HipError> {
    let ordinal = ordinal as i32;
    CURRENT_DEVICE.with(|current| {
        if current.get() != ordinal {
            check(unsafe { sys::hipSetDevice(ordinal) })?;
            current.set(ordinal);
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
/// 5us of host time, more than a kernel launch, which dominates small ops.
const CACHE_MAX_BLOCK_BYTES: usize = 64 << 20;
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
}

/// A stream on a given device. All candle work for a device is ordered on one.
#[derive(Debug)]
pub struct HipStream {
    raw: sys::hipStream_t,
    ordinal: usize,
    staging: std::sync::Mutex<Option<Staging>>,
    cache: std::sync::Mutex<BlockCache>,
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
        }))
    }

    pub fn raw(&self) -> sys::hipStream_t {
        self.raw
    }

    pub fn ordinal(&self) -> usize {
        self.ordinal
    }

    pub fn bind(&self) -> Result<(), HipError> {
        bind_device(self.ordinal)
    }

    pub fn synchronize(&self) -> Result<(), HipError> {
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
        let bytes = len * std::mem::size_of::<T>();
        let mut ptr = 0;
        let mut cached = false;
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

    fn take_cached(&self, block: usize) -> u64 {
        let mut cache = self.cache.lock().unwrap();
        match cache.free.get_mut(&block).and_then(Vec::pop) {
            Some(ptr) => {
                cache.total_bytes -= block;
                ptr
            }
            None => 0,
        }
    }

    /// Returns a block to the cache, or frees it when the cache is full.
    fn release(&self, ptr: u64, block: usize) {
        {
            let mut cache = self.cache.lock().unwrap();
            if cache.total_bytes + block <= CACHE_MAX_TOTAL_BYTES {
                cache.total_bytes += block;
                cache.free.entry(block).or_default().push(ptr);
                return;
            }
        }
        if self.bind().is_ok() {
            unsafe { sys::hipFreeAsync(ptr as *mut c_void, self.raw) };
        }
    }

    /// Frees every cached block back to the memory pool.
    pub fn release_cached(&self) -> Result<(), HipError> {
        let blocks: Vec<u64> = {
            let mut cache = self.cache.lock().unwrap();
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
        check(unsafe { sys::hipMemsetAsync(dst.device_ptr() as *mut c_void, 0, bytes, self.raw) })
    }

    pub fn clone_htod<T: DeviceRepr>(self: &Arc<Self>, src: &[T]) -> Result<HipSlice<T>, HipError> {
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
        assert!(dst.len() >= src.len(), "memcpy_htod: destination too small");
        let bytes = std::mem::size_of_val(src);
        if bytes == 0 {
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
            // Writes to src are ordered on its own stream: this stream waits for
            // them, and src's stream waits for the copy before src can be reused.
            // Neither wait blocks the host.
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
            self.order_before(&src.stream)?;
        }
        // Leave this stream's device current: raw kernel launchers rely on it.
        self.bind()?;
        Ok(dst)
    }
}

impl Drop for HipStream {
    fn drop(&mut self) {
        if bind_device(self.ordinal).is_ok() {
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
        } else if self.stream.bind().is_ok() {
            unsafe { sys::hipFreeAsync(self.ptr as *mut c_void, self.stream.raw) };
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
