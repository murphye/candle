use anyhow::Result;
use candle_core::quantized::{GgmlDType, QMatMul, QTensor};
use candle_core::{DType, Device, Module, Tensor};

/// Times quantized matmuls on the first ROCm device for decode and prefill sizes.
fn main() -> Result<()> {
    let device = Device::new_rocm(0)?;
    let (n, k) = (9728, 2560);
    let w = Tensor::randn(0f32, 1.0, (n, k), &device)?;
    for dtype in [GgmlDType::Q4K, GgmlDType::Q8_0] {
        let qmm = QMatMul::from_qtensor(QTensor::quantize(&w, dtype)?)?;
        let deq = QTensor::quantize(&w, dtype)?
            .dequantize(&device)?
            .to_dtype(DType::BF16)?;
        for m in [1usize, 8, 24, 128, 512] {
            for act in [DType::F32, DType::BF16] {
                let x = Tensor::randn(0f32, 1.0, (1, m, k), &device)?.to_dtype(act)?;
                let _ = qmm.forward(&x)?;
                device.synchronize()?;
                let iters = 20;
                let start = std::time::Instant::now();
                for _ in 0..iters {
                    let _ = qmm.forward(&x)?;
                }
                device.synchronize()?;
                let q = start.elapsed().as_secs_f64() * 1e3 / iters as f64;
                let xb = x.to_dtype(DType::BF16)?;
                let _ = xb.broadcast_matmul(&deq.t()?)?;
                device.synchronize()?;
                let start = std::time::Instant::now();
                for _ in 0..iters {
                    let _ = xb.broadcast_matmul(&deq.t()?)?;
                }
                device.synchronize()?;
                let d = start.elapsed().as_secs_f64() * 1e3 / iters as f64;
                println!("{dtype:?} m={m:<4} act={act:?}: qmatmul {q:.3} ms, bf16 gemm {d:.3} ms");
            }
        }
    }
    Ok(())
}
