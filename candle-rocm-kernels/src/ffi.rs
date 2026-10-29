//! Hand-written bindings for the subset of the HIP runtime and rocBLAS used by
//! candle. Only functions whose C ABI has been stable across ROCm releases are
//! declared; version-dependent structs such as `hipDeviceProp_t` are avoided.
#![allow(non_camel_case_types, non_upper_case_globals)]
use std::ffi::{c_char, c_int, c_uint, c_void};

pub type hipError_t = c_int;
pub type hipStream_t = *mut c_void;
pub type hipModule_t = *mut c_void;
pub type hipFunction_t = *mut c_void;
pub type hipEvent_t = *mut c_void;
pub type hipDeviceptr_t = *mut c_void;

pub const hipSuccess: hipError_t = 0;
pub const hipErrorOutOfMemory: hipError_t = 2;
pub const hipErrorDeinitialized: hipError_t = 4;
pub const hipErrorStreamCaptureUnsupported: hipError_t = 900;

pub type hipMemcpyKind = c_int;
pub const hipMemcpyHostToDevice: hipMemcpyKind = 1;
pub const hipMemcpyDeviceToHost: hipMemcpyKind = 2;
pub const hipMemcpyDeviceToDevice: hipMemcpyKind = 3;

pub const hipStreamNonBlocking: c_uint = 0x1;
pub const hipEventDisableTiming: c_uint = 0x2;

// Values in the CUDA-compatible range of hipDeviceAttribute_t, stable across releases.
pub type hipDeviceAttribute_t = c_int;
pub const hipDeviceAttributeIntegrated: hipDeviceAttribute_t = 16;

pub type hipMemPool_t = *mut c_void;
pub type hipMemPoolAttr = c_int;
pub const hipMemPoolAttrReleaseThreshold: hipMemPoolAttr = 0x4;

pub type hipGraph_t = *mut c_void;
pub type hipGraphExec_t = *mut c_void;
pub type hipStreamCaptureMode = c_int;
pub const hipStreamCaptureModeRelaxed: hipStreamCaptureMode = 2;

pub type hipStreamCaptureStatus = c_int;
pub const hipStreamCaptureStatusNone: hipStreamCaptureStatus = 0;
pub const hipStreamCaptureStatusActive: hipStreamCaptureStatus = 1;

extern "C" {
    pub fn hipInit(flags: c_uint) -> hipError_t;
    pub fn hipGetErrorName(error: hipError_t) -> *const c_char;
    pub fn hipGetErrorString(error: hipError_t) -> *const c_char;
    pub fn hipGetLastError() -> hipError_t;

    pub fn hipGetDeviceCount(count: *mut c_int) -> hipError_t;
    pub fn hipSetDevice(device: c_int) -> hipError_t;
    pub fn hipGetDevice(device: *mut c_int) -> hipError_t;
    pub fn hipDeviceSynchronize() -> hipError_t;
    pub fn hipDeviceGetAttribute(
        value: *mut c_int,
        attr: hipDeviceAttribute_t,
        device: c_int,
    ) -> hipError_t;
    pub fn hipDeviceGetName(name: *mut c_char, len: c_int, device: c_int) -> hipError_t;
    pub fn hipDeviceCanAccessPeer(can: *mut c_int, device: c_int, peer: c_int) -> hipError_t;
    pub fn hipDeviceEnablePeerAccess(peer: c_int, flags: c_uint) -> hipError_t;
    pub fn hipMemGetInfo(free: *mut usize, total: *mut usize) -> hipError_t;

    pub fn hipStreamCreateWithFlags(stream: *mut hipStream_t, flags: c_uint) -> hipError_t;
    pub fn hipStreamDestroy(stream: hipStream_t) -> hipError_t;
    pub fn hipStreamSynchronize(stream: hipStream_t) -> hipError_t;
    pub fn hipStreamWaitEvent(stream: hipStream_t, event: hipEvent_t, flags: c_uint) -> hipError_t;
    pub fn hipStreamIsCapturing(
        stream: hipStream_t,
        status: *mut hipStreamCaptureStatus,
    ) -> hipError_t;
    pub fn hipStreamBeginCapture(stream: hipStream_t, mode: hipStreamCaptureMode) -> hipError_t;
    pub fn hipStreamEndCapture(stream: hipStream_t, graph: *mut hipGraph_t) -> hipError_t;
    pub fn hipGraphInstantiateWithFlags(
        exec: *mut hipGraphExec_t,
        graph: hipGraph_t,
        flags: u64,
    ) -> hipError_t;
    pub fn hipGraphLaunch(exec: hipGraphExec_t, stream: hipStream_t) -> hipError_t;
    pub fn hipGraphExecDestroy(exec: hipGraphExec_t) -> hipError_t;
    pub fn hipGraphDestroy(graph: hipGraph_t) -> hipError_t;
    pub fn hipMalloc(ptr: *mut *mut c_void, size: usize) -> hipError_t;
    pub fn hipMemcpy(
        dst: *mut c_void,
        src: *const c_void,
        size: usize,
        kind: hipMemcpyKind,
    ) -> hipError_t;

    pub fn hipEventCreateWithFlags(event: *mut hipEvent_t, flags: c_uint) -> hipError_t;
    pub fn hipEventRecord(event: hipEvent_t, stream: hipStream_t) -> hipError_t;
    pub fn hipEventSynchronize(event: hipEvent_t) -> hipError_t;
    pub fn hipEventDestroy(event: hipEvent_t) -> hipError_t;
    pub fn hipEventQuery(event: hipEvent_t) -> hipError_t;

    pub fn hipDeviceGetDefaultMemPool(pool: *mut hipMemPool_t, device: c_int) -> hipError_t;
    pub fn hipMemPoolSetAttribute(
        pool: hipMemPool_t,
        attr: hipMemPoolAttr,
        value: *mut c_void,
    ) -> hipError_t;
    pub fn hipHostMalloc(ptr: *mut *mut c_void, size: usize, flags: c_uint) -> hipError_t;
    pub fn hipHostFree(ptr: *mut c_void) -> hipError_t;
    pub fn hipMallocAsync(ptr: *mut *mut c_void, size: usize, stream: hipStream_t) -> hipError_t;
    pub fn hipFreeAsync(ptr: *mut c_void, stream: hipStream_t) -> hipError_t;
    pub fn hipMemcpyAsync(
        dst: *mut c_void,
        src: *const c_void,
        size: usize,
        kind: hipMemcpyKind,
        stream: hipStream_t,
    ) -> hipError_t;
    pub fn hipMemcpyPeerAsync(
        dst: *mut c_void,
        dst_device: c_int,
        src: *const c_void,
        src_device: c_int,
        size: usize,
        stream: hipStream_t,
    ) -> hipError_t;
    pub fn hipMemsetAsync(
        dst: *mut c_void,
        value: c_int,
        size: usize,
        stream: hipStream_t,
    ) -> hipError_t;

    pub fn hipModuleLoadData(module: *mut hipModule_t, image: *const c_void) -> hipError_t;
    pub fn hipModuleUnload(module: hipModule_t) -> hipError_t;
    pub fn hipModuleGetFunction(
        function: *mut hipFunction_t,
        module: hipModule_t,
        name: *const c_char,
    ) -> hipError_t;
    #[allow(clippy::too_many_arguments)]
    pub fn hipModuleLaunchKernel(
        f: hipFunction_t,
        grid_x: c_uint,
        grid_y: c_uint,
        grid_z: c_uint,
        block_x: c_uint,
        block_y: c_uint,
        block_z: c_uint,
        shared_mem_bytes: c_uint,
        stream: hipStream_t,
        kernel_params: *mut *mut c_void,
        extra: *mut *mut c_void,
    ) -> hipError_t;
}

pub type rocblas_handle = *mut c_void;
pub type rocblas_status = c_int;
pub const rocblas_status_success: rocblas_status = 0;

pub type rocblas_operation = c_int;
pub const rocblas_operation_none: rocblas_operation = 111;
pub const rocblas_operation_transpose: rocblas_operation = 112;

pub type rocblas_datatype = c_int;
pub const rocblas_datatype_f16_r: rocblas_datatype = 150;
pub const rocblas_datatype_f32_r: rocblas_datatype = 151;
pub const rocblas_datatype_f64_r: rocblas_datatype = 152;
pub const rocblas_datatype_bf16_r: rocblas_datatype = 168;

pub type rocblas_gemm_algo = c_int;
pub const rocblas_gemm_algo_standard: rocblas_gemm_algo = 0;

pub type rocblas_gemm_flags = c_uint;
pub const rocblas_gemm_flags_none: rocblas_gemm_flags = 0;

extern "C" {
    pub fn rocblas_create_handle(handle: *mut rocblas_handle) -> rocblas_status;
    pub fn rocblas_destroy_handle(handle: rocblas_handle) -> rocblas_status;
    pub fn rocblas_set_stream(handle: rocblas_handle, stream: hipStream_t) -> rocblas_status;
    pub fn rocblas_status_to_string(status: rocblas_status) -> *const c_char;
    #[allow(clippy::too_many_arguments)]
    pub fn rocblas_gemm_strided_batched_ex(
        handle: rocblas_handle,
        trans_a: rocblas_operation,
        trans_b: rocblas_operation,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: *const c_void,
        a: *const c_void,
        a_type: rocblas_datatype,
        lda: c_int,
        stride_a: i64,
        b: *const c_void,
        b_type: rocblas_datatype,
        ldb: c_int,
        stride_b: i64,
        beta: *const c_void,
        c: *const c_void,
        c_type: rocblas_datatype,
        ldc: c_int,
        stride_c: i64,
        d: *mut c_void,
        d_type: rocblas_datatype,
        ldd: c_int,
        stride_d: i64,
        batch_count: c_int,
        compute_type: rocblas_datatype,
        algo: rocblas_gemm_algo,
        solution_index: i32,
        flags: rocblas_gemm_flags,
    ) -> rocblas_status;
}
