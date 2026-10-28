#![cfg(feature = "rocm")]

use anyhow::Result;
use candle_core::{DType, Device, Tensor};

fn max_rel_diff(a: &Tensor, b: &Tensor) -> Result<f32> {
    let a = a.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let b = b.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let scale = b.iter().fold(1e-3f32, |m, v| m.max(v.abs()));
    Ok(a.iter().zip(&b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max) / scale)
}

/// Decode-shaped matmuls against a transposed weight take the GEMV kernel; the
/// result must match the CPU for every dtype, row count and output width.
#[test]
fn gemv_matches_cpu() -> Result<()> {
    let dev = Device::new_rocm(0)?;
    for dtype in [DType::F32, DType::F16, DType::BF16] {
        for (rows, n, k) in [(1, 288, 2560), (1, 1, 2560), (3, 640, 512), (8, 37, 64)] {
            let x = Tensor::randn(0f32, 1.0, (rows, k), &Device::Cpu)?.to_dtype(dtype)?;
            let w = Tensor::randn(0f32, 1.0, (n, k), &Device::Cpu)?.to_dtype(dtype)?;
            let expected = x.to_dtype(DType::F32)?.matmul(&w.to_dtype(DType::F32)?.t()?)?;
            let got = x.to_device(&dev)?.matmul(&w.to_device(&dev)?.t()?)?;
            let diff = max_rel_diff(&got, &expected)?;
            let tol = if dtype == DType::F32 { 1e-5 } else { 1e-2 };
            assert!(diff < tol, "{dtype:?} rows={rows} n={n} k={k}: rel diff {diff}");
        }
    }
    Ok(())
}

/// A weight broadcast over a batch is folded into the row count.
#[test]
fn gemv_broadcast_batch_matches_cpu() -> Result<()> {
    let dev = Device::new_rocm(0)?;
    let x = Tensor::randn(0f32, 1.0, (2, 1, 256), &Device::Cpu)?;
    let w = Tensor::randn(0f32, 1.0, (64, 256), &Device::Cpu)?;
    let expected = x.broadcast_matmul(&w.t()?)?;
    let got = x.to_device(&dev)?.broadcast_matmul(&w.to_device(&dev)?.t()?)?;
    assert!(max_rel_diff(&got, &expected)? < 1e-5);
    Ok(())
}

/// Shapes the GEMV kernel does not take still go through BLAS.
#[test]
fn gemv_fallbacks_match_cpu() -> Result<()> {
    let dev = Device::new_rocm(0)?;
    // Non-transposed weight, too many rows, and k not a multiple of the vector width.
    let cases = [((1, 128), (128, 32), false), ((16, 128), (32, 128), true), ((1, 130), (32, 130), true)];
    for ((m, k), wshape, transposed) in cases {
        let x = Tensor::randn(0f32, 1.0, (m, k), &Device::Cpu)?;
        let w = Tensor::randn(0f32, 1.0, wshape, &Device::Cpu)?;
        let rhs = |w: &Tensor| if transposed { w.t() } else { Ok(w.clone()) };
        let expected = x.matmul(&rhs(&w)?)?;
        let got = x.to_device(&dev)?.matmul(&rhs(&w.to_device(&dev)?)?)?;
        assert!(max_rel_diff(&got, &expected)? < 1e-4);
    }
    Ok(())
}
