use anyhow::Result;
use candle_core::{Device, Tensor};

/// Per-node cost of replaying a HIP graph of tiny dependent kernels: `N=1000` nodes.
fn main() -> Result<()> {
    let device = Device::new_rocm(0)?;
    let rocm = device.as_rocm_device()?.clone();
    let n: usize = std::env::var("N").ok().and_then(|v| v.parse().ok()).unwrap_or(1000);
    let x = Tensor::ones(1024, candle_core::DType::F32, &device)?;
    device.synchronize()?;
    rocm.begin_capture()?;
    let captured = (|| -> candle_core::Result<Tensor> {
        let mut y = x.clone();
        for _ in 0..n {
            y = (y * 1.0001)?;
        }
        Ok(y)
    })();
    let graph = rocm.end_capture()?;
    let _y = captured?;
    for _ in 0..5 {
        rocm.launch_graph(&graph)?;
    }
    device.synchronize()?;
    let iters = 50;
    let start = std::time::Instant::now();
    for _ in 0..iters {
        rocm.launch_graph(&graph)?;
    }
    device.synchronize()?;
    let per_graph = start.elapsed().as_secs_f64() * 1e6 / iters as f64;
    println!("{n} nodes: {per_graph:.0} us per replay, {:.2} us per node", per_graph / n as f64);
    Ok(())
}
