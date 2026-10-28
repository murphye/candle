#![allow(unused)]
use super::GgmlDType;
use crate::{Error, Result, RocmDevice, RocmStorage};

pub struct QRocmStorage {
    dtype: GgmlDType,
    device: RocmDevice,
}

impl QRocmStorage {
    pub fn zeros(_: &RocmDevice, _: usize, _: GgmlDType) -> Result<Self> {
        Err(Error::NotCompiledWithRocmSupport)
    }

    pub fn dtype(&self) -> GgmlDType {
        self.dtype
    }

    pub fn device(&self) -> &RocmDevice {
        &self.device
    }

    pub fn dequantize(&self, _elem_count: usize) -> Result<RocmStorage> {
        Err(Error::NotCompiledWithRocmSupport)
    }

    pub fn dequantize_f16(&self, _elem_count: usize) -> Result<RocmStorage> {
        Err(Error::NotCompiledWithRocmSupport)
    }

    pub fn quantize(&mut self, _src: &RocmStorage) -> Result<()> {
        Err(Error::NotCompiledWithRocmSupport)
    }

    pub fn quantize_imatrix(
        &mut self,
        _src: &RocmStorage,
        _imatrix_weights: &[f32],
        _n_per_row: usize,
    ) -> Result<()> {
        Err(Error::NotCompiledWithRocmSupport)
    }

    pub fn quantize_imatrix_onto(
        &mut self,
        _src: &crate::CpuStorage,
        _imatrix_weights: &[f32],
        _n_per_row: usize,
    ) -> Result<()> {
        Err(Error::NotCompiledWithRocmSupport)
    }

    pub fn quantize_onto(&mut self, _src: &crate::CpuStorage) -> Result<()> {
        Err(Error::NotCompiledWithRocmSupport)
    }

    pub fn device_ptr(&self) -> Result<*const u8> {
        Err(Error::NotCompiledWithRocmSupport)
    }

    pub fn storage_size_in_bytes(&self) -> usize {
        0
    }

    pub fn embedding(
        &self,
        _rows: usize,
        _hidden: usize,
        _ids: &RocmStorage,
        _ids_l: &crate::Layout,
    ) -> Result<RocmStorage> {
        Err(Error::NotCompiledWithRocmSupport)
    }

    pub fn fwd(
        &self,
        _self_shape: &crate::Shape,
        _storage: &RocmStorage,
        _layout: &crate::Layout,
    ) -> Result<(RocmStorage, crate::Shape)> {
        Err(Error::NotCompiledWithRocmSupport)
    }

    pub fn data(&self) -> Result<Vec<u8>> {
        Err(Error::NotCompiledWithRocmSupport)
    }

    pub fn indexed_moe_forward(
        &self,
        _: &crate::Shape,
        _: &RocmStorage,
        _: &crate::Layout,
        _: &RocmStorage,
        _: &crate::Layout,
    ) -> Result<(RocmStorage, crate::Shape)> {
        Err(Error::NotCompiledWithRocmSupport)
    }
}

pub fn load_quantized<T: super::GgmlType + Send + Sync + 'static>(
    _device: &RocmDevice,
    _data: &[T],
) -> Result<super::QStorage> {
    Err(Error::NotCompiledWithRocmSupport)
}
