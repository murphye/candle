use anyhow::Result;
use candle_core::quantized::{GgmlDType, QTensor};
use candle_core::{DType, Device, Tensor};

/// Times the indexed MoE kernel at Qwen3.8-Flash-Next (REAP-288) expert shapes.
fn main() -> Result<()> {
    let device = Device::new_rocm(0)?;
    // `INTER=320` gives the per-rank shapes of a two-way tensor-parallel split.
    let inter: usize = std::env::var("INTER").ok().and_then(|v| v.parse().ok()).unwrap_or(640);
    let (experts, topk, hidden) = (288usize, 10usize, 2560usize);
    for (name, n, k, dtype) in [
        ("gate/up", inter, hidden, GgmlDType::Q4K),
        ("down", hidden, inter, GgmlDType::Q8_0),
        ("down", hidden, inter, GgmlDType::Q5_0),
    ] {
        let w = Tensor::randn(0f32, 1.0, (experts, n, k), &device)?;
        let qw = QTensor::quantize(&w, dtype)?;
        for tokens in [1usize, 8, 64, 160, 512, 2048] {
            let ids: Vec<u32> = (0..tokens * topk)
                .map(|i| ((i * 7919 + i / topk * 104729) % experts) as u32)
                .collect();
            let ids = Tensor::from_vec(ids, (tokens, topk), &device)?;
            // gate/up share one input row per token; down has one row per expert.
            let x_t = if name == "down" { topk } else { 1 };
            let x = Tensor::randn(0f32, 1.0, (tokens, x_t, k), &device)?.to_dtype(DType::BF16)?;
            let _ = qw.indexed_moe_forward(&x, &ids)?;
            device.synchronize()?;
            let iters = 10;
            let start = std::time::Instant::now();
            for _ in 0..iters {
                let _ = qw.indexed_moe_forward(&x, &ids)?;
            }
            device.synchronize()?;
            let ms = start.elapsed().as_secs_f64() * 1e3 / iters as f64;
            println!("{name:>7} {dtype:?} tokens={tokens:<4}: {ms:.3} ms");
        }
    }
    Ok(())
}
