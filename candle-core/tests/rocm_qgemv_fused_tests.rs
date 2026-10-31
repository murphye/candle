#![cfg(feature = "rocm")]

use candle_core::quantized::{
    rocm::qgemv_gate_mean, rocm::qgemv_pair, GgmlDType, QMatMul, QTensor,
};
use candle_core::{DType, Device, Module, Result, Tensor, D};

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
    // k = 10240 with Q4_K/Q6_K and Q4_K takes the shape-specialized kernels; 12 and 16 rows
    // take the int8 WMMA pair (Q4_K, Q6_K and Q5_0 only), whose activations are q8_1.
    for (k, t1, t2) in [
        (1024, GgmlDType::Q4K, GgmlDType::Q4K),
        (1024, GgmlDType::Q6K, GgmlDType::Q4K),
        (1024, GgmlDType::Q5_0, GgmlDType::Q8_0),
        (1024, GgmlDType::Q5_0, GgmlDType::Q6K),
        (10240, GgmlDType::Q4K, GgmlDType::Q4K),
        (10240, GgmlDType::Q6K, GgmlDType::Q4K),
    ] {
        let (q1, w1) = quantized(40, k, t1, 0.1, &dev)?;
        let (q2, w2) = quantized(4, k, t2, 0.9, &dev)?;
        let (q1, q2) = (std::sync::Arc::new(q1), std::sync::Arc::new(q2));
        let wide = t2 != GgmlDType::Q8_0;
        for rows in [1usize, 3, 8, 12, 16] {
            if rows > 8 && !wide {
                continue;
            }
            let x = Tensor::from_vec(
                (0..rows * k).map(|i| ((i as f32 * 0.013).cos()) * 0.5).collect::<Vec<_>>(),
                (rows, k),
                &Device::Cpu,
            )?
            .to_dtype(DType::BF16)?;
            let xd = x.to_device(&dev)?;
            let (y1, y2) = qgemv_pair(&xd, &q1, Some(0.25), &q2)?.expect("fused path");
            let xf = x.to_dtype(DType::F32)?;
            let want1 = silu(&(xf.matmul(&w1.t()?)? * 0.25)?)?;
            let want2 = xf.matmul(&w2.t()?)?;
            // Past 8 rows the activations are quantized to q8_1, as in the unfused matmuls at
            // these widths: compare with those tightly and with the f32 reference loosely.
            let tol = if rows > 8 { 5e-2 } else { 2e-2 };
            assert!(
                max_rel_err(&y1, &want1)? < tol,
                "{t1:?} rows={rows}: {}",
                max_rel_err(&y1, &want1)?
            );
            assert!(
                max_rel_err(&y2, &want2)? < tol,
                "{t2:?} rows={rows}: {}",
                max_rel_err(&y2, &want2)?
            );
            if rows > 8 {
                let unfused1 = QMatMul::from_arc(q1.clone())?
                    .forward(&xd)?
                    .to_dtype(DType::F32)?;
                let unfused1 = silu(&(unfused1 * 0.25)?)?;
                let unfused2 = QMatMul::from_arc(q2.clone())?.forward(&xd)?;
                assert!(
                    max_rel_err(&y1, &unfused1)? < 1e-2,
                    "{t1:?} rows={rows} vs unfused: {}",
                    max_rel_err(&y1, &unfused1)?
                );
                assert!(
                    max_rel_err(&y2, &unfused2)? < 1e-2,
                    "{t2:?} rows={rows} vs unfused: {}",
                    max_rel_err(&y2, &unfused2)?
                );
            }
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
        // Past 8 rows only the specialized (k = 320, Q5_0) kernel takes the fused path.
        for rows in [1usize, 3, 8, 12, 16] {
            if rows > 8 && (k, t) != (320, GgmlDType::Q5_0) {
                continue;
            }
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

/// Per-call times of the fused mixer kernels at Qwen3.8-Flash-Next's shapes (TP2 rank: 10240
/// streams in, 320 low-rank, 4 x 2560 out) and decode/verify widths.
/// `cargo test --release --features rocm --test rocm_qgemv_fused_tests -- --ignored --nocapture`
#[test]
#[ignore]
fn fused_mixer_timing() -> Result<()> {
    let Ok(dev) = Device::new_rocm(0) else {
        return Ok(());
    };
    let (q_down, _) = quantized(320, 10240, GgmlDType::Q4K, 0.1, &dev)?;
    let (q_inject, _) = quantized(4, 10240, GgmlDType::Q4K, 0.2, &dev)?;
    let (q_up, _) = quantized(4 * 2560, 320, GgmlDType::Q5_0, 0.3, &dev)?;
    for rows in [1usize, 3, 8, 12, 16] {
        let x = Tensor::randn(0f32, 1.0, (rows, 10240), &dev)?.to_dtype(DType::BF16)?;
        let lo = Tensor::randn(0f32, 1.0, (rows, 320), &dev)?.to_dtype(DType::BF16)?;
        let time = |f: &dyn Fn() -> Result<()>| -> Result<f64> {
            for _ in 0..5 {
                f()?;
            }
            dev.synchronize()?;
            let t = std::time::Instant::now();
            for _ in 0..200 {
                f()?;
            }
            dev.synchronize()?;
            Ok(t.elapsed().as_secs_f64() * 1e6 / 200.0)
        };
        let pair = time(&|| {
            qgemv_pair(&x, &q_down, Some(0.25), &q_inject)?.expect("fused path");
            Ok(())
        })?;
        let gate = time(&|| {
            qgemv_gate_mean(&lo, &q_up, &x, 4)?.expect("fused path");
            Ok(())
        })?;
        println!("rows {rows:2}: pair {pair:6.1} us, gate mean {gate:6.1} us");
    }
    Ok(())
}
