use anyhow::Result;
use candle_core::quantized::{GgmlDType, QMatMul, QTensor};
use candle_core::{DType, Device, Module, Tensor};

/// Single-token quantized matmuls at Qwen3.8-Flash-Next projection shapes.
fn main() -> Result<()> {
    let device = Device::new_rocm(0)?;
    for (name, n, k, dtype) in [
        ("gdn qkv", 10240usize, 2560usize, GgmlDType::Q6K),
        ("gdn gate", 6144, 2560, GgmlDType::Q4K),
        ("gdn out", 2560, 6144, GgmlDType::Q4K),
        ("shexp up", 640, 2560, GgmlDType::Q4K),
        ("shexp down", 2560, 640, GgmlDType::Q8_0),
        ("hc down", 320, 10240, GgmlDType::Q4K),
        ("hc up", 10240, 320, GgmlDType::Q5_0),
        ("lm head", 248320, 2560, GgmlDType::Q6K),
    ] {
        let w = Tensor::randn(0f32, 1.0, (n, k), &device)?;
        let qt = QTensor::quantize(&w, dtype)?;
        let bytes = qt.storage_size_in_bytes() as f64;
        let qmm = QMatMul::from_qtensor(qt)?;
        let x = Tensor::randn(0f32, 1.0, (1, 1, k), &device)?.to_dtype(DType::BF16)?;
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
            "{name:>10} {dtype:?} {n}x{k}: {us:7.1} us, {:6.0} GB/s",
            bytes / us / 1e3
        );
    }
    Ok(())
}
