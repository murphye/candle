use anyhow::Result;
use candle_core::quantized::rocm::{qgemv_gate_mean, qgemv_pair};
use candle_core::quantized::{GgmlDType, QTensor};
use candle_core::{DType, Device, Tensor};

/// The two fused hyper-connection mixer launches at Qwen3.8-Flash-Next shapes (hc 4, hidden
/// 2560, low rank 320): `ROWS=1,3` picks the activation row counts.
fn main() -> Result<()> {
    let device = Device::new_rocm(0)?;
    let batches: Vec<usize> = std::env::var("ROWS")
        .ok()
        .map(|v| v.split(',').filter_map(|r| r.parse().ok()).collect())
        .unwrap_or_else(|| vec![1, 3]);
    let (hc, hidden, lowrank) = (4usize, 2560usize, 320usize);
    let k = hc * hidden;
    let quant = |rows: usize, cols: usize, dtype: GgmlDType| -> Result<QTensor> {
        let w = Tensor::randn(0f32, 0.3, (rows, cols), &device)?;
        Ok(QTensor::quantize(&w, dtype)?)
    };
    for down_dtype in [GgmlDType::Q4K, GgmlDType::Q6K] {
        let down = quant(lowrank, k, down_dtype)?;
        let inject = quant(hc, k, GgmlDType::Q4K)?;
        let up = quant(k, lowrank, GgmlDType::Q5_0)?;
        for &rows in &batches {
            let xn = Tensor::randn(0f32, 1.0, (rows, k), &device)?.to_dtype(DType::BF16)?;
            let time = |f: &dyn Fn() -> Result<()>| -> Result<f64> {
                for _ in 0..20 {
                    f()?;
                }
                device.synchronize()?;
                let iters = 300;
                let start = std::time::Instant::now();
                for _ in 0..iters {
                    f()?;
                }
                device.synchronize()?;
                Ok(start.elapsed().as_secs_f64() * 1e6 / iters as f64)
            };
            let pair = time(&|| {
                qgemv_pair(&xn, &down, Some(1.0 / hc as f32), &inject)?.expect("fused");
                Ok(())
            })?;
            let (lo, _) = qgemv_pair(&xn, &down, Some(1.0 / hc as f32), &inject)?.expect("fused");
            let gate = time(&|| {
                qgemv_gate_mean(&lo, &up, &xn, hc)?.expect("fused");
                Ok(())
            })?;
            println!(
                "down {down_dtype:?} rows={rows}: pair {pair:6.1} us, gate_mean {gate:6.1} us"
            );
        }
    }
    Ok(())
}
