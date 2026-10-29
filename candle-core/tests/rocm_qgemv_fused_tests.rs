#![cfg(feature = "rocm")]

use candle_core::quantized::{rocm::qgemv_gate_mean, rocm::qgemv_pair, GgmlDType, QTensor};
use candle_core::{DType, Device, Result, Tensor, D};

fn sigmoid(x: &Tensor) -> Result<Tensor> {
    (x.neg()?.exp()? + 1.0)?.recip()
}

fn silu(x: &Tensor) -> Result<Tensor> {
    x * sigmoid(x)?
}

fn max_rel_err(got: &Tensor, want: &Tensor) -> Result<f32> {
    let got = got.to_dtype(DType::F32)?.to_device(&Device::Cpu)?;
    let want = want.to_dtype(DType::F32)?.to_device(&Device::Cpu)?;
    let err = (got - &want)?.abs()?.max_all()?.to_scalar::<f32>()?;
    let scale = want.abs()?.max_all()?.to_scalar::<f32>()?.max(1e-3);
    Ok(err / scale)
}

fn quantized(rows: usize, k: usize, dtype: GgmlDType, salt: f32, dev: &Device) -> Result<(QTensor, Tensor)> {
    let w = Tensor::from_vec(
        (0..rows * k).map(|i| ((i as f32 * 0.071 + salt).sin()) * 0.3).collect::<Vec<_>>(),
        (rows, k),
        &Device::Cpu,
    )?;
    let q = QTensor::quantize(&w, dtype)?;
    // The reference uses the dequantized weights, so only the kernel's arithmetic is compared.
    let deq = q.dequantize(&Device::Cpu)?;
    Ok((QTensor::quantize_onto(&w, dtype, dev)?, deq))
}

#[test]
fn fused_pair_matches_reference() -> Result<()> {
    let Ok(dev) = Device::new_rocm(0) else { return Ok(()) };
    let k = 1024;
    for (t1, t2) in [(GgmlDType::Q4K, GgmlDType::Q4K), (GgmlDType::Q6K, GgmlDType::Q4K), (GgmlDType::Q5_0, GgmlDType::Q8_0)] {
        let (q1, w1) = quantized(40, k, t1, 0.1, &dev)?;
        let (q2, w2) = quantized(4, k, t2, 0.9, &dev)?;
        for rows in [1usize, 3, 8] {
            let x = Tensor::from_vec(
                (0..rows * k).map(|i| ((i as f32 * 0.013).cos()) * 0.5).collect::<Vec<_>>(),
                (rows, k),
                &Device::Cpu,
            )?
            .to_dtype(DType::BF16)?;
            let (y1, y2) = qgemv_pair(&x.to_device(&dev)?, &q1, Some(0.25), &q2)?.expect("fused path");
            let xf = x.to_dtype(DType::F32)?;
            let want1 = silu(&(xf.matmul(&w1.t()?)? * 0.25)?)?;
            let want2 = xf.matmul(&w2.t()?)?;
            assert!(max_rel_err(&y1, &want1)? < 2e-2, "{t1:?} rows={rows}: {}", max_rel_err(&y1, &want1)?);
            assert!(max_rel_err(&y2, &want2)? < 2e-2, "{t2:?} rows={rows}: {}", max_rel_err(&y2, &want2)?);
        }
    }
    Ok(())
}

#[test]
fn fused_gate_mean_matches_reference() -> Result<()> {
    let Ok(dev) = Device::new_rocm(0) else { return Ok(()) };
    for (k, t) in [(320, GgmlDType::Q5_0), (256, GgmlDType::Q4K), (512, GgmlDType::Q6K)] {
        let (groups, hidden) = (4, 96);
        let (q, w) = quantized(groups * hidden, k, t, 0.4, &dev)?;
        for rows in [1usize, 3, 8] {
            let lo = Tensor::from_vec(
                (0..rows * k).map(|i| ((i as f32 * 0.029).sin()) * 0.4).collect::<Vec<_>>(),
                (rows, k),
                &Device::Cpu,
            )?
            .to_dtype(DType::BF16)?;
            let xn = Tensor::from_vec(
                (0..rows * groups * hidden).map(|i| ((i as f32 * 0.017).cos())).collect::<Vec<_>>(),
                (rows, groups * hidden),
                &Device::Cpu,
            )?
            .to_dtype(DType::BF16)?;
            let got = qgemv_gate_mean(&lo.to_device(&dev)?, &q, &xn.to_device(&dev)?, groups)?.expect("fused path");
            let gate = sigmoid(&lo.to_dtype(DType::F32)?.matmul(&w.t()?)?)?;
            let want = (xn.to_dtype(DType::F32)? * gate)?
                .reshape((rows, groups, hidden))?
                .mean(D::Minus2)?;
            assert!(max_rel_err(&got, &want)? < 2e-2, "{t:?} rows={rows}: {}", max_rel_err(&got, &want)?);
        }
    }
    Ok(())
}
