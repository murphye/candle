use anyhow::Result;
use candle_core::{DType, Device, Tensor, D};

/// Runs a handful of ops on the first ROCm device and compares them with the CPU.
fn main() -> Result<()> {
    let device = Device::new_rocm(0)?;
    let rocm = device.as_rocm_device()?;
    let (free, total) = rocm.mem_get_info()?;
    println!(
        "device: {:?}, memory {:.1}/{:.1} GiB free",
        device.location(),
        free as f64 / (1 << 30) as f64,
        total as f64 / (1 << 30) as f64
    );

    let a = Tensor::randn(0f32, 1.0, (4, 256, 512), &Device::Cpu)?;
    let b = Tensor::randn(0f32, 1.0, (4, 512, 128), &Device::Cpu)?;
    let check = |name: &str, cpu: &Tensor, gpu: &Tensor| -> Result<()> {
        let diff = (cpu - gpu.to_device(&Device::Cpu)?.to_dtype(DType::F32)?)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()?;
        println!("{name:>12}: max abs diff {diff:.3e}");
        Ok(())
    };

    let (ga, gb) = (a.to_device(&device)?, b.to_device(&device)?);
    check("matmul f32", &a.matmul(&b)?, &ga.matmul(&gb)?)?;
    check(
        "matmul bf16",
        &a.matmul(&b)?,
        &ga.to_dtype(DType::BF16)?
            .matmul(&gb.to_dtype(DType::BF16)?)?,
    )?;
    check(
        "exp-sum",
        &candle_core::Tensor::exp(&a)?.sum_keepdim(D::Minus1)?,
        &ga.exp()?.sum_keepdim(D::Minus1)?,
    )?;
    check("transpose", &a.t()?.contiguous()?, &ga.t()?.contiguous()?)?;
    check(
        "argmax",
        &a.argmax(D::Minus1)?.to_dtype(DType::F32)?,
        &ga.argmax(D::Minus1)?,
    )?;

    let r = Tensor::randn(0f32, 1.0, (1 << 20,), &device)?;
    let mean = r.mean_all()?.to_scalar::<f32>()?;
    let var = r.sqr()?.mean_all()?.to_scalar::<f32>()?;
    println!("       randn: mean {mean:.4}, var {var:.4}");

    let x = ga.to_dtype(DType::F16)?;
    for _ in 0..3 {
        let start = std::time::Instant::now();
        for _ in 0..100 {
            let _ = x.matmul(&gb.to_dtype(DType::F16)?)?;
        }
        device.synchronize()?;
        println!(" 100x matmul: {:?}", start.elapsed());
    }
    Ok(())
}
