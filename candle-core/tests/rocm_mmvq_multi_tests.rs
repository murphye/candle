#![cfg(feature = "rocm")]

use candle_core::quantized::{rocm::fast_mmvq_multi, GgmlDType, QMatMul, QTensor};
use candle_core::{DType, Device, Module, Result, Tensor};

fn max_rel_err(got: &Tensor, want: &Tensor) -> Result<f32> {
    let got = got.to_dtype(DType::F32)?.to_device(&Device::Cpu)?;
    let want = want.to_dtype(DType::F32)?.to_device(&Device::Cpu)?;
    let err = (got - &want)?.abs()?.max_all()?.to_scalar::<f32>()?;
    let scale = want.abs()?.max_all()?.to_scalar::<f32>()?.max(1e-3);
    Ok(err / scale)
}

/// Four weights of different formats and row counts in one launch match the one-by-one path.
#[test]
fn multi_segment_matches_single_matmuls() -> Result<()> {
    let Some(dev) = Device::new_rocm(0).ok() else { return Ok(()) };
    let k = 2560;
    let specs = [(2304usize, GgmlDType::Q6K), (3072, GgmlDType::Q4K), (48, GgmlDType::Q4K), (640, GgmlDType::Q8_0)];
    let weights = specs
        .iter()
        .enumerate()
        .map(|(i, (n, dtype))| {
            let w = Tensor::from_vec(
                (0..n * k).map(|j| ((j as f32 * 0.0137 + i as f32).sin()) * 0.4).collect::<Vec<_>>(),
                (*n, k),
                &Device::Cpu,
            )?;
            Ok(std::sync::Arc::new(QTensor::quantize_onto(&w, *dtype, &dev)?))
        })
        .collect::<Result<Vec<_>>>()?;
    for rows in [1usize, 3, 8] {
        let x = Tensor::from_vec(
            (0..rows * k).map(|j| ((j as f32 * 0.011).cos()) * 0.6).collect::<Vec<_>>(),
            (1, rows, k),
            &Device::Cpu,
        )?
        .to_dtype(DType::BF16)?
        .to_device(&dev)?;
        let refs = weights.iter().map(|w| w.as_ref()).collect::<Vec<_>>();
        let outs = fast_mmvq_multi(&refs, &x)?.expect("multi-segment path");
        assert_eq!(outs.len(), weights.len());
        for (out, w) in outs.iter().zip(&weights) {
            let want = QMatMul::from_arc(w.clone())?.forward(&x)?;
            assert_eq!(out.dims(), want.dims());
            let err = max_rel_err(out, &want)?;
            assert!(err < 1e-2, "{:?} rows={rows}: rel err {err}", w.dtype());
        }
    }
    Ok(())
}
