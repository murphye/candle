use anyhow::Result;
use candle_core::quantized::{GgmlDType, QMatMul, QTensor};
use candle_core::{DType, Device, Module, Tensor};

/// Decode-batch quantized matmuls at Qwen3.8-Flash-Next projection shapes, per rank of a
/// two-way tensor-parallel split. `ROWS=1,3` picks the activation row counts.
fn main() -> Result<()> {
    let device = Device::new_rocm(0)?;
    let batches: Vec<usize> = std::env::var("ROWS")
        .ok()
        .map(|v| v.split(',').filter_map(|r| r.parse().ok()).collect())
        .unwrap_or_else(|| vec![1, 3]);
    for (name, n, k, dtype) in [
        ("gdn qkv q6", 7168usize, 2560usize, GgmlDType::Q6K),
        ("gdn qkv q4", 7168, 2560, GgmlDType::Q4K),
        ("gdn z", 3072, 2560, GgmlDType::Q4K),
        ("gdn out", 2560, 3072, GgmlDType::Q4K),
        ("qsa q", 6144, 2560, GgmlDType::Q4K),
        ("qsa o", 2560, 3072, GgmlDType::Q4K),
        ("shexp up", 640, 2560, GgmlDType::Q4K),
        ("shexp down", 2560, 640, GgmlDType::Q8_0),
        ("lm head/2", 124160, 2560, GgmlDType::Q6K),
        // Few-row weights (k > 8 * rows): the split-K kernels at verify widths.
        ("hc gate", 4, 10240, GgmlDType::Q8_0),
        ("gdn ab", 24, 2560, GgmlDType::Q4K),
        ("small 256", 256, 2560, GgmlDType::Q6K),
        ("hc 320", 320, 10240, GgmlDType::Q4K),
    ] {
      for &rows in &batches {
        let w = Tensor::randn(0f32, 1.0, (n, k), &device)?;
        let qt = QTensor::quantize(&w, dtype)?;
        let bytes = qt.storage_size_in_bytes() as f64;
        let qmm = QMatMul::from_qtensor(qt)?;
        let x = Tensor::randn(0f32, 1.0, (1, rows, k), &device)?.to_dtype(DType::BF16)?;
        for _ in 0..10 {
            qmm.forward(&x)?;
        }
        device.synchronize()?;
        let iters = 200;
        let start = std::time::Instant::now();
        for _ in 0..iters {
            qmm.forward(&x)?;
        }
        device.synchronize()?;
        let us = start.elapsed().as_secs_f64() * 1e6 / iters as f64;
        println!(
            "{name:>10} {dtype:?} {n}x{k} rows={rows}: {us:7.1} us, {:6.0} GB/s",
            bytes / us / 1e3
        );
      }
    }
    Ok(())
}
