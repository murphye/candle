#![allow(dead_code)]
use crate::op::{BinaryOpT, CmpOp, ReduceOp, UnaryOpT};
use crate::{CpuStorage, DType, Error, Layout, Result, Shape};

#[derive(Debug, Clone)]
pub struct RocmDevice;

#[derive(Debug)]
pub struct RocmStorage;

#[derive(thiserror::Error, Debug)]
pub enum RocmError {
    #[error("{0}")]
    Message(String),
}

impl From<String> for RocmError {
    fn from(e: String) -> Self {
        RocmError::Message(e)
    }
}

impl RocmStorage {
    pub fn transfer_to_device(&self, _dst: &RocmDevice) -> Result<Self> {
        Err(Error::NotCompiledWithRocmSupport)
    }
}

macro_rules! fail {
    () => {
        unimplemented!("rocm support has not been enabled, add `rocm` feature to enable.")
    };
}

impl crate::backend::BackendStorage for RocmStorage {
    type Device = RocmDevice;

    fn try_clone(&self, _: &Layout) -> Result<Self> {
        Err(Error::NotCompiledWithRocmSupport)
    }

    fn dtype(&self) -> DType {
        fail!()
    }

    fn device(&self) -> &Self::Device {
        fail!()
    }

    fn const_set(&mut self, _: crate::scalar::Scalar, _: &Layout) -> Result<()> {
        Err(Error::NotCompiledWithRocmSupport)
    }

    fn to_cpu_storage(&self) -> Result<CpuStorage> {
        Err(Error::NotCompiledWithRocmSupport)
    }

    fn affine(&self, _: &Layout, _: f64, _: f64) -> Result<Self> {
        Err(Error::NotCompiledWithRocmSupport)
    }

    fn powf(&self, _: &Layout, _: f64) -> Result<Self> {
        Err(Error::NotCompiledWithRocmSupport)
    }

    fn elu(&self, _: &Layout, _: f64) -> Result<Self> {
        Err(Error::NotCompiledWithRocmSupport)
    }

    fn reduce_op(&self, _: ReduceOp, _: &Layout, _: &[usize]) -> Result<Self> {
        Err(Error::NotCompiledWithRocmSupport)
    }

    fn cmp(&self, _: CmpOp, _: &Self, _: &Layout, _: &Layout) -> Result<Self> {
        Err(Error::NotCompiledWithRocmSupport)
    }

    fn to_dtype(&self, _: &Layout, _: DType) -> Result<Self> {
        Err(Error::NotCompiledWithRocmSupport)
    }

    fn unary_impl<B: UnaryOpT>(&self, _: &Layout) -> Result<Self> {
        Err(Error::NotCompiledWithRocmSupport)
    }

    fn binary_impl<B: BinaryOpT>(&self, _: &Self, _: &Layout, _: &Layout) -> Result<Self> {
        Err(Error::NotCompiledWithRocmSupport)
    }

    fn where_cond(&self, _: &Layout, _: &Self, _: &Layout, _: &Self, _: &Layout) -> Result<Self> {
        Err(Error::NotCompiledWithRocmSupport)
    }

    fn conv1d(
        &self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &crate::conv::ParamsConv1D,
    ) -> Result<Self> {
        Err(Error::NotCompiledWithRocmSupport)
    }

    fn conv_transpose1d(
        &self,
        _l: &Layout,
        _kernel: &Self,
        _kernel_l: &Layout,
        _params: &crate::conv::ParamsConvTranspose1D,
    ) -> Result<Self> {
        Err(Error::NotCompiledWithRocmSupport)
    }

    fn conv2d(
        &self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &crate::conv::ParamsConv2D,
    ) -> Result<Self> {
        Err(Error::NotCompiledWithRocmSupport)
    }

    fn conv_transpose2d(
        &self,
        _l: &Layout,
        _kernel: &Self,
        _kernel_l: &Layout,
        _params: &crate::conv::ParamsConvTranspose2D,
    ) -> Result<Self> {
        Err(Error::NotCompiledWithRocmSupport)
    }

    fn index_select(&self, _: &Self, _: &Layout, _: &Layout, _: usize) -> Result<Self> {
        Err(Error::NotCompiledWithRocmSupport)
    }
    fn gather(&self, _: &Layout, _: &Self, _: &Layout, _: usize) -> Result<Self> {
        Err(Error::NotCompiledWithRocmSupport)
    }

    fn scatter_set(
        &mut self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: usize,
    ) -> Result<()> {
        Err(Error::NotCompiledWithRocmSupport)
    }

    fn scatter_add_set(
        &mut self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: usize,
    ) -> Result<()> {
        Err(Error::NotCompiledWithRocmSupport)
    }

    fn index_add(
        &self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: usize,
    ) -> Result<Self> {
        Err(Error::NotCompiledWithRocmSupport)
    }

    fn matmul(
        &self,
        _: &Self,
        _: (usize, usize, usize, usize),
        _: &Layout,
        _: &Layout,
    ) -> Result<Self> {
        Err(Error::NotCompiledWithRocmSupport)
    }

    fn copy_strided_src(&self, _: &mut Self, _: usize, _: &Layout) -> Result<()> {
        Err(Error::NotCompiledWithRocmSupport)
    }

    fn copy2d(
        &self,
        _: &mut Self,
        _: usize,
        _: usize,
        _: usize,
        _: usize,
        _: usize,
        _: usize,
    ) -> Result<()> {
        Err(Error::NotCompiledWithRocmSupport)
    }

    fn avg_pool2d(&self, _: &Layout, _: (usize, usize), _: (usize, usize)) -> Result<Self> {
        Err(Error::NotCompiledWithRocmSupport)
    }

    fn max_pool2d(&self, _: &Layout, _: (usize, usize), _: (usize, usize)) -> Result<Self> {
        Err(Error::NotCompiledWithRocmSupport)
    }

    fn upsample_nearest1d(&self, _: &Layout, _: usize) -> Result<Self> {
        Err(Error::NotCompiledWithRocmSupport)
    }

    fn upsample_nearest2d(&self, _: &Layout, _: usize, _: usize) -> Result<Self> {
        Err(Error::NotCompiledWithRocmSupport)
    }

    fn upsample_bilinear2d(
        &self,
        _: &Layout,
        _: usize,
        _: usize,
        _: bool,
        _: Option<f64>,
        _: Option<f64>,
    ) -> Result<Self> {
        Err(Error::NotCompiledWithRocmSupport)
    }
}

impl crate::backend::BackendDevice for RocmDevice {
    type Storage = RocmStorage;
    fn new(_: usize) -> Result<Self> {
        Err(Error::NotCompiledWithRocmSupport)
    }

    fn set_seed(&self, _: u64) -> Result<()> {
        Err(Error::NotCompiledWithRocmSupport)
    }

    fn get_current_seed(&self) -> Result<u64> {
        Err(Error::NotCompiledWithRocmSupport)
    }

    fn location(&self) -> crate::DeviceLocation {
        fail!()
    }

    fn same_device(&self, _: &Self) -> bool {
        fail!()
    }

    fn zeros_impl(&self, _shape: &Shape, _dtype: DType) -> Result<Self::Storage> {
        Err(Error::NotCompiledWithRocmSupport)
    }

    unsafe fn alloc_uninit(&self, _shape: &Shape, _dtype: DType) -> Result<Self::Storage> {
        Err(Error::NotCompiledWithRocmSupport)
    }

    fn storage_from_slice<T: crate::WithDType>(&self, _: &[T]) -> Result<Self::Storage> {
        Err(Error::NotCompiledWithRocmSupport)
    }

    fn storage_from_cpu_storage(&self, _: &CpuStorage) -> Result<Self::Storage> {
        Err(Error::NotCompiledWithRocmSupport)
    }

    fn storage_from_cpu_storage_owned(&self, _: CpuStorage) -> Result<Self::Storage> {
        Err(Error::NotCompiledWithRocmSupport)
    }

    fn rand_uniform(&self, _: &Shape, _: DType, _: f64, _: f64) -> Result<Self::Storage> {
        Err(Error::NotCompiledWithRocmSupport)
    }

    fn rand_normal(&self, _: &Shape, _: DType, _: f64, _: f64) -> Result<Self::Storage> {
        Err(Error::NotCompiledWithRocmSupport)
    }

    fn synchronize(&self) -> Result<()> {
        Ok(())
    }
}
