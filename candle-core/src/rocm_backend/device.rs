use super::hip::{
    DevicePtr, DevicePtrMut, DeviceRepr, HipFunction, HipModule, HipSlice, HipStream, LaunchArgs,
    LaunchConfig, RocBlas, ValidAsZeroBits,
};
use crate::backend::{BackendDevice, BackendStorage};
use crate::{CpuStorage, CpuStorageRef, DType, Result, Shape};
pub use candle_rocm_kernels as kernels;
use float8::F8E4M3;
use half::{bf16, f16};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};

use super::{RocmError, RocmStorage, RocmStorageSlice, WrapErr};

/// Unique identifier for rocm devices.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DeviceId(usize);

impl DeviceId {
    fn new() -> Self {
        // https://users.rust-lang.org/t/idiomatic-rust-way-to-generate-unique-id/33805
        use std::sync::atomic;
        static COUNTER: atomic::AtomicUsize = atomic::AtomicUsize::new(1);
        Self(COUNTER.fetch_add(1, atomic::Ordering::Relaxed))
    }
}

/// State of the counter based generator in the `RANDOM` kernels.
struct RngState {
    seed: u64,
    offset: u64,
}

pub struct ModuleStore {
    mdls: [Option<Arc<HipModule>>; kernels::ALL_IDS.len()],
    /// Functions already looked up, per module, so a launch skips hipModuleGetFunction.
    funcs: [HashMap<String, HipFunction>; kernels::ALL_IDS.len()],
    /// Functions found by [`RocmDevice::get_or_load_func_keyed`], keyed without a name string.
    keyed: HashMap<FuncKey, HipFunction>,
}

/// Addresses of static name parts plus a small variant number.
type FuncKey = (usize, usize, usize, usize);

#[derive(Clone)]
pub struct RocmDevice {
    id: DeviceId,
    ordinal: usize,
    modules: Arc<RwLock<ModuleStore>>,
    custom_modules: Arc<RwLock<HashMap<String, Arc<HipModule>>>>,
    stream: Arc<HipStream>,
    pub(crate) blas: Arc<RocBlas>,
    rng: Arc<Mutex<RngState>>,
}

impl std::fmt::Debug for RocmDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "RocmDevice({:?})", self.id)
    }
}

impl RocmDevice {
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn alloc<T: DeviceRepr>(&self, len: usize) -> Result<HipSlice<T>> {
        self.stream.alloc::<T>(len).w()
    }

    pub fn alloc_zeros<T: DeviceRepr + ValidAsZeroBits>(&self, len: usize) -> Result<HipSlice<T>> {
        self.stream.alloc_zeros::<T>(len).w()
    }

    pub fn memcpy_htod<T: DeviceRepr, Dst: DevicePtrMut<T>>(
        &self,
        src: &[T],
        dst: &mut Dst,
    ) -> Result<()> {
        self.stream.memcpy_htod(src, dst).w()
    }

    pub fn clone_dtoh<T: DeviceRepr, Src: DevicePtr<T>>(&self, src: &Src) -> Result<Vec<T>> {
        self.stream.clone_dtoh(src).w()
    }

    pub fn memcpy_dtod<T, Src: DevicePtr<T>, Dst: DevicePtrMut<T>>(
        &self,
        src: &Src,
        dst: &mut Dst,
    ) -> Result<()> {
        self.stream.memcpy_dtod(src, dst).w()
    }

    pub fn memcpy_dtoh<T: DeviceRepr, Src: DevicePtr<T>>(
        &self,
        src: &Src,
        dst: &mut [T],
    ) -> Result<()> {
        self.stream.memcpy_dtoh(src, dst).w()
    }

    pub fn clone_htod<T: DeviceRepr>(&self, src: &[T]) -> Result<HipSlice<T>> {
        self.stream.clone_htod(src).w()
    }
}

pub struct RocmFunc {
    func: HipFunction,
    stream: Arc<HipStream>,
}

impl std::ops::Deref for RocmFunc {
    type Target = HipFunction;

    fn deref(&self) -> &Self::Target {
        &self.func
    }
}

impl RocmFunc {
    pub fn into_hip_function(self) -> HipFunction {
        self.func
    }

    pub fn builder(&self) -> LaunchArgs<'_> {
        LaunchArgs::new(&self.func, &self.stream)
    }
}

impl RocmDevice {
    pub fn hip_stream(&self) -> Arc<HipStream> {
        self.stream.clone()
    }

    /// Starts capturing this device's work into a graph; see [`HipStream::begin_capture`].
    pub fn begin_capture(&self) -> Result<()> {
        self.stream.begin_capture().w()
    }

    /// Ends the capture started by [`Self::begin_capture`] and instantiates the graph.
    pub fn end_capture(&self) -> Result<super::hip::HipGraph> {
        self.stream.end_capture().w()
    }

    /// Replays a graph captured on this device.
    pub fn launch_graph(&self, graph: &super::hip::HipGraph) -> Result<()> {
        graph.launch(&self.stream).w()
    }

    pub fn id(&self) -> DeviceId {
        self.id
    }

    pub fn ordinal(&self) -> usize {
        self.ordinal
    }

    /// Whether this is an integrated GPU sharing system memory.
    pub fn is_integrated(&self) -> Result<bool> {
        super::hip::device_is_integrated(self.ordinal).w()
    }

    /// Device name as reported by the HIP runtime.
    pub fn name(&self) -> Result<String> {
        super::hip::device_name(self.ordinal).w()
    }

    /// Returns `(free, total)` memory on this device in bytes.
    pub fn mem_get_info(&self) -> Result<(usize, usize)> {
        super::hip::mem_get_info(self.ordinal).w()
    }

    /// Loads `fn_name` from a code object (typically produced by `hipcc --genco`),
    /// caching the module under `module_name`.
    pub fn get_or_load_custom_func(
        &self,
        fn_name: &str,
        module_name: &str,
        code: &[u8],
    ) -> Result<RocmFunc> {
        let ms = self.custom_modules.read().unwrap();
        if let Some(mdl) = ms.get(module_name).as_ref() {
            let func = mdl.load_function(fn_name).w()?;
            return Ok(RocmFunc {
                func,
                stream: self.stream.clone(),
            });
        }
        drop(ms);
        let mut ms = self.custom_modules.write().unwrap();
        let module = HipModule::load_data(self.ordinal, code).map_err(|hip| RocmError::Load {
            hip,
            module_name: module_name.to_string(),
        })?;
        ms.insert(module_name.to_string(), module.clone());
        let func = module.load_function(fn_name).w()?;
        Ok(RocmFunc {
            func,
            stream: self.stream.clone(),
        })
    }

    pub fn get_or_load_func(&self, fn_name: &str, mdl: &kernels::Module) -> Result<RocmFunc> {
        let index = mdl.index();
        if let Some(func) = self.modules.read().unwrap().funcs[index].get(fn_name) {
            return Ok(RocmFunc {
                func: func.clone(),
                stream: self.stream.clone(),
            });
        }
        let mut ms = self.modules.write().unwrap();
        let module = match ms.mdls[index].as_ref() {
            Some(module) => module.clone(),
            None => {
                let module = HipModule::load_data(self.ordinal, mdl.code()).map_err(|hip| {
                    RocmError::Load {
                        hip,
                        module_name: format!("{:?}", kernels::ALL_IDS[index]),
                    }
                })?;
                ms.mdls[index] = Some(module.clone());
                module
            }
        };
        let func = module.load_function(fn_name).w()?;
        ms.funcs[index].insert(fn_name.to_string(), func.clone());
        Ok(RocmFunc {
            func,
            stream: self.stream.clone(),
        })
    }

    /// [`Self::get_or_load_func`] for hot launch paths: the function is cached under the
    /// addresses of static name parts and a variant number, so a hit neither formats nor
    /// hashes a name. `name` builds the full kernel name on a miss.
    pub fn get_or_load_func_keyed(
        &self,
        parts: (&'static str, &'static str, &'static str),
        variant: usize,
        mdl: &kernels::Module,
        name: impl FnOnce() -> String,
    ) -> Result<RocmFunc> {
        let key = (
            parts.0.as_ptr() as usize,
            parts.1.as_ptr() as usize,
            parts.2.as_ptr() as usize,
            variant,
        );
        if let Some(func) = self.modules.read().unwrap().keyed.get(&key) {
            return Ok(RocmFunc {
                func: func.clone(),
                stream: self.stream.clone(),
            });
        }
        let func = self.get_or_load_func(&name(), mdl)?;
        self.modules
            .write()
            .unwrap()
            .keyed
            .insert(key, func.func.clone());
        Ok(func)
    }

    pub fn rocblas_handle(&self) -> Arc<RocBlas> {
        self.blas.clone()
    }

    fn rand_fill<T: DeviceRepr + crate::WithDType>(
        &self,
        name: &str,
        elem_count: usize,
        p0: f64,
        p1: f64,
    ) -> Result<HipSlice<T>> {
        let (seed, offset) = {
            let mut rng = self.rng.lock().unwrap();
            let offset = rng.offset;
            rng.offset += elem_count as u64;
            (rng.seed, offset)
        };
        let func = self.get_or_load_func(&super::kernel_name::<T>(name), &kernels::RANDOM)?;
        // SAFETY: Set later by running the kernel.
        let out = unsafe { self.alloc::<T>(elem_count)? };
        let mut builder = func.builder();
        let n = elem_count;
        builder.arg(&n);
        builder.arg(&seed);
        builder.arg(&offset);
        // The kernels take f32 parameters for f32/f16/bf16 and f64 for f64.
        let (p0_f32, p1_f32) = (p0 as f32, p1 as f32);
        if T::DTYPE == DType::F64 {
            builder.arg(&p0);
            builder.arg(&p1);
        } else {
            builder.arg(&p0_f32);
            builder.arg(&p1_f32);
        }
        builder.arg(&out);
        let cfg = LaunchConfig::for_num_elems(elem_count as u32);
        // SAFETY: ffi.
        unsafe { builder.launch(cfg) }.w()?;
        Ok(out)
    }
}

impl RocmDevice {
    pub fn new_with_stream(ordinal: usize) -> Result<Self> {
        Self::new(ordinal)
    }
}

impl BackendDevice for RocmDevice {
    type Storage = RocmStorage;

    fn new(ordinal: usize) -> Result<Self> {
        let count = super::hip::device_count().w()?;
        if ordinal >= count {
            crate::bail!("rocm device {ordinal} requested but only {count} device(s) available")
        }
        let stream = HipStream::new(ordinal).w()?;
        let blas = RocBlas::new(stream.clone()).w()?;
        let module_store = ModuleStore {
            mdls: [const { None }; kernels::ALL_IDS.len()],
            funcs: std::array::from_fn(|_| HashMap::new()),
            keyed: HashMap::new(),
        };
        Ok(Self {
            id: DeviceId::new(),
            ordinal,
            stream,
            blas: Arc::new(blas),
            rng: Arc::new(Mutex::new(RngState {
                seed: 299792458,
                offset: 0,
            })),
            modules: Arc::new(RwLock::new(module_store)),
            custom_modules: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    fn set_seed(&self, seed: u64) -> Result<()> {
        *self.rng.lock().unwrap() = RngState { seed, offset: 0 };
        Ok(())
    }

    fn get_current_seed(&self) -> Result<u64> {
        Ok(self.rng.lock().unwrap().seed)
    }

    fn location(&self) -> crate::DeviceLocation {
        crate::DeviceLocation::Rocm {
            gpu_id: self.ordinal,
        }
    }

    fn same_device(&self, rhs: &Self) -> bool {
        self.id == rhs.id
    }

    fn zeros_impl(&self, shape: &Shape, dtype: DType) -> Result<RocmStorage> {
        let elem_count = shape.elem_count();
        let slice = match dtype {
            DType::U8 => RocmStorageSlice::U8(self.alloc_zeros::<u8>(elem_count)?),
            DType::U32 => RocmStorageSlice::U32(self.alloc_zeros::<u32>(elem_count)?),
            DType::I16 => RocmStorageSlice::I16(self.alloc_zeros::<i16>(elem_count)?),
            DType::I32 => RocmStorageSlice::I32(self.alloc_zeros::<i32>(elem_count)?),
            DType::I64 => RocmStorageSlice::I64(self.alloc_zeros::<i64>(elem_count)?),
            DType::BF16 => RocmStorageSlice::BF16(self.alloc_zeros::<bf16>(elem_count)?),
            DType::F16 => RocmStorageSlice::F16(self.alloc_zeros::<f16>(elem_count)?),
            DType::F32 => RocmStorageSlice::F32(self.alloc_zeros::<f32>(elem_count)?),
            DType::F64 => RocmStorageSlice::F64(self.alloc_zeros::<f64>(elem_count)?),
            DType::F8E4M3 => RocmStorageSlice::F8E4M3(self.alloc_zeros::<F8E4M3>(elem_count)?),
            DType::F6E2M3 | DType::F6E3M2 | DType::F4 | DType::F8E8M0 => {
                return Err(
                    RocmError::InternalError("Dummy types not supported in ROCm backend").into(),
                )
            }
        };
        Ok(RocmStorage {
            slice,
            device: self.clone(),
        })
    }

    fn rand_uniform(&self, shape: &Shape, dtype: DType, lo: f64, up: f64) -> Result<RocmStorage> {
        let n = shape.elem_count();
        let slice = match dtype {
            DType::F32 => RocmStorageSlice::F32(self.rand_fill("rand_uniform", n, lo, up)?),
            DType::F64 => RocmStorageSlice::F64(self.rand_fill("rand_uniform", n, lo, up)?),
            DType::F16 => RocmStorageSlice::F16(self.rand_fill("rand_uniform", n, lo, up)?),
            DType::BF16 => RocmStorageSlice::BF16(self.rand_fill("rand_uniform", n, lo, up)?),
            _ => Err(RocmError::UnsupportedDtype {
                dtype,
                op: "rand_uniform",
            })
            .w()?,
        };
        Ok(RocmStorage {
            slice,
            device: self.clone(),
        })
    }

    fn rand_normal(&self, shape: &Shape, dtype: DType, mean: f64, std: f64) -> Result<RocmStorage> {
        let n = shape.elem_count();
        let slice = match dtype {
            DType::F32 => RocmStorageSlice::F32(self.rand_fill("rand_normal", n, mean, std)?),
            DType::F64 => RocmStorageSlice::F64(self.rand_fill("rand_normal", n, mean, std)?),
            DType::F16 => RocmStorageSlice::F16(self.rand_fill("rand_normal", n, mean, std)?),
            DType::BF16 => RocmStorageSlice::BF16(self.rand_fill("rand_normal", n, mean, std)?),
            _ => Err(RocmError::UnsupportedDtype {
                dtype,
                op: "rand_normal",
            })
            .w()?,
        };
        Ok(RocmStorage {
            slice,
            device: self.clone(),
        })
    }

    unsafe fn alloc_uninit(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        let elem_count = shape.elem_count();
        let slice = match dtype {
            DType::U8 => RocmStorageSlice::U8(self.alloc::<u8>(elem_count)?),
            DType::U32 => RocmStorageSlice::U32(self.alloc::<u32>(elem_count)?),
            DType::I16 => RocmStorageSlice::I16(self.alloc::<i16>(elem_count)?),
            DType::I32 => RocmStorageSlice::I32(self.alloc::<i32>(elem_count)?),
            DType::I64 => RocmStorageSlice::I64(self.alloc::<i64>(elem_count)?),
            DType::BF16 => RocmStorageSlice::BF16(self.alloc::<bf16>(elem_count)?),
            DType::F16 => RocmStorageSlice::F16(self.alloc::<f16>(elem_count)?),
            DType::F32 => RocmStorageSlice::F32(self.alloc::<f32>(elem_count)?),
            DType::F64 => RocmStorageSlice::F64(self.alloc::<f64>(elem_count)?),
            DType::F8E4M3 => RocmStorageSlice::F8E4M3(self.alloc::<F8E4M3>(elem_count)?),
            DType::F6E2M3 | DType::F6E3M2 | DType::F4 | DType::F8E8M0 => {
                return Err(
                    RocmError::InternalError("Dummy types not supported in ROCm backend").into(),
                )
            }
        };
        Ok(RocmStorage {
            slice,
            device: self.clone(),
        })
    }

    fn storage_from_slice<T: crate::WithDType>(&self, s: &[T]) -> Result<Self::Storage> {
        let slice = match T::cpu_storage_ref(s) {
            CpuStorageRef::U8(s) => RocmStorageSlice::U8(self.clone_htod(s)?),
            CpuStorageRef::U32(s) => RocmStorageSlice::U32(self.clone_htod(s)?),
            CpuStorageRef::I16(s) => RocmStorageSlice::I16(self.clone_htod(s)?),
            CpuStorageRef::I32(s) => RocmStorageSlice::I32(self.clone_htod(s)?),
            CpuStorageRef::I64(s) => RocmStorageSlice::I64(self.clone_htod(s)?),
            CpuStorageRef::BF16(s) => RocmStorageSlice::BF16(self.clone_htod(s)?),
            CpuStorageRef::F16(s) => RocmStorageSlice::F16(self.clone_htod(s)?),
            CpuStorageRef::F32(s) => RocmStorageSlice::F32(self.clone_htod(s)?),
            CpuStorageRef::F64(s) => RocmStorageSlice::F64(self.clone_htod(s)?),
            CpuStorageRef::F8E4M3(s) => RocmStorageSlice::F8E4M3(self.clone_htod(s)?),
            CpuStorageRef::F4(_)
            | CpuStorageRef::F6E2M3(_)
            | CpuStorageRef::F6E3M2(_)
            | CpuStorageRef::F8E8M0(_) => {
                return Err(RocmError::UnsupportedDtype {
                    dtype: T::DTYPE,
                    op: "storage_from_slice",
                }
                .into());
            }
        };
        Ok(RocmStorage {
            slice,
            device: self.clone(),
        })
    }

    fn storage_from_cpu_storage(&self, storage: &CpuStorage) -> Result<RocmStorage> {
        let slice = match storage {
            CpuStorage::U8(s) => RocmStorageSlice::U8(self.clone_htod(s)?),
            CpuStorage::U32(s) => RocmStorageSlice::U32(self.clone_htod(s)?),
            CpuStorage::I16(s) => RocmStorageSlice::I16(self.clone_htod(s)?),
            CpuStorage::I32(s) => RocmStorageSlice::I32(self.clone_htod(s)?),
            CpuStorage::I64(s) => RocmStorageSlice::I64(self.clone_htod(s)?),
            CpuStorage::BF16(s) => RocmStorageSlice::BF16(self.clone_htod(s)?),
            CpuStorage::F16(s) => RocmStorageSlice::F16(self.clone_htod(s)?),
            CpuStorage::F32(s) => RocmStorageSlice::F32(self.clone_htod(s)?),
            CpuStorage::F64(s) => RocmStorageSlice::F64(self.clone_htod(s)?),
            CpuStorage::F8E4M3(s) => RocmStorageSlice::F8E4M3(self.clone_htod(s)?),
            CpuStorage::F4(_)
            | CpuStorage::F6E2M3(_)
            | CpuStorage::F6E3M2(_)
            | CpuStorage::F8E8M0(_) => {
                return Err(RocmError::UnsupportedDtype {
                    dtype: storage.dtype(),
                    op: "storage_from_cpu_storage",
                }
                .into());
            }
        };
        Ok(RocmStorage {
            slice,
            device: self.clone(),
        })
    }

    fn storage_from_cpu_storage_owned(&self, storage: CpuStorage) -> Result<RocmStorage> {
        self.storage_from_cpu_storage(&storage)
    }

    fn synchronize(&self) -> Result<()> {
        self.stream.synchronize().w()
    }
}
