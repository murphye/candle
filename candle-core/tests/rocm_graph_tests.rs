#![cfg(feature = "rocm")]

use candle_core::{Device, Result, Tensor};

fn device() -> Option<Device> {
    Device::new_rocm(0).ok()
}

/// Copies `src` into `dst` in place on the device (outside any capture).
fn write_into(dst: &Tensor, src: &Tensor) -> Result<()> {
    dst.slice_set(src, 0, 0)
}

#[test]
fn captured_ops_replay_against_updated_inputs() -> Result<()> {
    let Some(dev) = device() else { return Ok(()) };
    let rocm = dev.as_rocm_device()?.clone();
    let input = Tensor::zeros((4, 8), candle_core::DType::F32, &dev)?;
    let weight = Tensor::arange(0f32, 8., &dev)?.reshape((1, 8))?;
    dev.synchronize()?;

    rocm.begin_capture()?;
    let captured = (|| {
        let scaled = input.broadcast_mul(&weight)?;
        let shifted = (scaled + 1.0)?;
        shifted.sum_keepdim(1)
    })();
    let graph = rocm.end_capture()?;
    let output = captured?;

    for step in 1..4 {
        let fresh = Tensor::full(step as f32, (4, 8), &dev)?;
        write_into(&input, &fresh)?;
        rocm.launch_graph(&graph)?;
        let got = output.flatten_all()?.to_vec1::<f32>()?;
        // sum_j (step * j + 1) for j in 0..8
        let expected = step as f32 * 28.0 + 8.0;
        assert!(got.iter().all(|v| (v - expected).abs() < 1e-4), "{got:?} != {expected}");
    }
    Ok(())
}

#[test]
fn reads_fail_and_uploads_stay_constant_while_capturing() -> Result<()> {
    let Some(dev) = device() else { return Ok(()) };
    let rocm = dev.as_rocm_device()?.clone();
    let x = Tensor::ones(3, candle_core::DType::F32, &dev)?;
    dev.synchronize()?;
    rocm.begin_capture()?;
    let download = x.to_vec1::<f32>();
    let captured = Tensor::new(&[1f32, 2., 3.], &dev).and_then(|c| &x + c);
    let graph = rocm.end_capture()?;
    assert!(download.is_err(), "device-to-host copy succeeded during capture");
    let output = captured?;
    // Eager uploads of the same size after the capture must not land in the captured constant.
    let _other = Tensor::new(&[9f32, 9., 9.], &dev)?;
    rocm.launch_graph(&graph)?;
    assert_eq!(output.to_vec1::<f32>()?, [2., 3., 4.]);
    Ok(())
}

#[test]
fn graph_buffers_are_not_handed_to_eager_work() -> Result<()> {
    let Some(dev) = device() else { return Ok(()) };
    let rocm = dev.as_rocm_device()?.clone();
    let input = Tensor::ones(1024, candle_core::DType::F32, &dev)?;
    dev.synchronize()?;
    rocm.begin_capture()?;
    let captured = (|| {
        let tmp = (&input * 2.0)?; // an intermediate, freed inside the capture
        (&tmp + 1.0)?.sum_all()
    })();
    let graph = rocm.end_capture()?;
    let output = captured?;
    // An eager tensor of the intermediate's size must not reuse its block: the replay below
    // rewrites that block.
    let eager = Tensor::full(7f32, 1024, &dev)?;
    rocm.launch_graph(&graph)?;
    assert_eq!(output.to_vec0::<f32>()?, 3.0 * 1024.0);
    assert!(eager.to_vec1::<f32>()?.iter().all(|v| *v == 7.0));
    Ok(())
}

