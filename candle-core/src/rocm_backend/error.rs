use super::hip::{HipError, RocblasError};
use crate::{DType, Layout};

/// ROCm related errors
#[derive(thiserror::Error, Debug)]
pub enum RocmError {
    #[error(transparent)]
    Hip(#[from] HipError),

    #[error(transparent)]
    Rocblas(#[from] RocblasError),

    #[error("missing kernel '{module_name}'")]
    MissingKernel { module_name: String },

    #[error("unsupported dtype {dtype:?} for {op}")]
    UnsupportedDtype { dtype: DType, op: &'static str },

    #[error("internal error '{0}'")]
    InternalError(&'static str),

    #[error("matmul is only supported for contiguous tensors lstride: {lhs_stride:?} rstride: {rhs_stride:?} mnk: {mnk:?}")]
    MatMulNonContiguous {
        lhs_stride: Layout,
        rhs_stride: Layout,
        mnk: (usize, usize, usize),
    },

    #[error("{msg}, expected: {expected:?}, got: {got:?}")]
    UnexpectedDType {
        msg: &'static str,
        expected: DType,
        got: DType,
    },

    #[error("{hip} when loading {module_name}")]
    Load { hip: HipError, module_name: String },
}

impl From<RocmError> for crate::Error {
    fn from(val: RocmError) -> Self {
        crate::Error::Rocm(Box::new(val)).bt()
    }
}

pub trait WrapErr<O> {
    fn w(self) -> std::result::Result<O, crate::Error>;
}

impl<O, E: Into<RocmError>> WrapErr<O> for std::result::Result<O, E> {
    fn w(self) -> std::result::Result<O, crate::Error> {
        self.map_err(|e| crate::Error::Rocm(Box::new(e.into())).bt())
    }
}
