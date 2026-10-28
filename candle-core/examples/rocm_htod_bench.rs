use anyhow::Result;
use candle_core::{DType, Device, Tensor};

/// Times small host-to-device uploads, as done for indices and positions,
/// before and after large temporary allocations.
fn main() -> Result<()> {
    let device = Device::new_rocm(0)?;
    let time = |label: &str, len: usize| -> Result<()> {
        let data: Vec<u32> = (0..len as u32).collect();
        device.synchronize()?;
        let iters = 20;
        let start = std::time::Instant::now();
        for _ in 0..iters {
            let _ = Tensor::from_vec(data.clone(), len, &device)?;
        }
        device.synchronize()?;
        println!(
            "{label}: upload {len:>8} u32: {:.1} us",
            start.elapsed().as_secs_f64() * 1e6 / iters as f64
        );
        Ok(())
    };
    for len in [4096usize, 20736, 65536] {
        time("fresh", len)?;
    }
    // A large temporary, like a dequantized expert stack.
    for _ in 0..3 {
        let big = Tensor::zeros((288, 640, 2560), DType::BF16, &device)?;
        drop(big);
        device.synchronize()?;
    }
    for len in [4096usize, 20480, 20736, 21504, 32768, 40000, 65536] {
        time("after 1GB temporaries", len)?;
    }
    for len in [4096usize, 20736, 65536] {
        device.synchronize()?;
        let start = std::time::Instant::now();
        for _ in 0..20 {
            let _ = Tensor::zeros(len, DType::U32, &device)?;
        }
        device.synchronize()?;
        println!(
            "alloc+memset {len:>8} u32: {:.1} us",
            start.elapsed().as_secs_f64() * 1e6 / 20.0
        );
    }
    Ok(())
}
