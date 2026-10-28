use anyhow::Result;
use candle_core::{DType, Device, Tensor};

/// Measures the host cost of small ops on ROCm, where decode is launch bound.
fn main() -> Result<()> {
    let device = Device::new_rocm(0)?;
    let x = Tensor::randn(0f32, 1.0, 2560, &device)?.to_dtype(DType::BF16)?;
    let y = x.clone();
    let iters = 5000;
    for (name, f) in [
        ("affine", &(|x: &Tensor, _: &Tensor| x.affine(0.5, 0.0)) as &dyn Fn(&Tensor, &Tensor) -> candle_core::Result<Tensor>),
        ("silu", &|x: &Tensor, _: &Tensor| x.silu()),
        ("add", &|x: &Tensor, y: &Tensor| x + y),
        ("cast", &|x: &Tensor, _: &Tensor| x.to_dtype(DType::F32)),
    ] {
        for _ in 0..100 {
            f(&x, &y)?;
        }
        device.synchronize()?;
        let start = std::time::Instant::now();
        for _ in 0..iters {
            f(&x, &y)?;
        }
        let host = start.elapsed().as_secs_f64() * 1e6 / iters as f64;
        device.synchronize()?;
        let total = start.elapsed().as_secs_f64() * 1e6 / iters as f64;
        println!("{name:>7}: host {host:.2} us/op, with sync {total:.2} us/op");
    }
    Ok(())
}
