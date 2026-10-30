#![cfg(feature = "rocm")]

use candle_core::rocm_backend::peer::{PeerAllReduce, PeerAllReduceSetup};
use candle_core::{DType, Device, Result, Tensor};

#[test]
fn peer_allreduce_sums_across_gpus() -> Result<()> {
    let devices = [Device::new_rocm(0), Device::new_rocm(1)];
    let [Ok(d0), Ok(d1)] = devices else {
        return Ok(());
    };
    let setups = [&d0, &d1]
        .iter()
        .enumerate()
        .map(|(rank, d)| {
            PeerAllReduceSetup::new(d.as_rocm_device().unwrap(), rank, 2, 6 << 20).map(|(s, _)| s)
        })
        .collect::<Result<Vec<_>>>()?;
    let ranks: Vec<PeerAllReduce> = PeerAllReduceSetup::connect_in_process(setups)?;
    // Sizes on both sides of the one-workgroup / whole-GPU split, interleaved so the two
    // kernels share the epoch and the receive-buffer halves.
    for n in [8usize, 3 * 2560, 128 * 2560, 8 * 2560, 2048 * 2560, 16, 1024 * 2560 + 8] {
        // Several calls in a row exercise both halves of the receive buffers.
        for call in 0..5 {
            let inputs = [&d0, &d1]
                .iter()
                .enumerate()
                .map(|(rank, d)| {
                    Tensor::arange(0u32, n as u32, d)?
                        .to_dtype(DType::F32)?
                        .affine(0.01, (rank * 3 + call) as f64)?
                        .to_dtype(DType::BF16)
                })
                .collect::<Result<Vec<_>>>()?;
            // Both ranks launch before either is waited on, as two processes would.
            let outs = ranks
                .iter()
                .zip(&inputs)
                .map(|(r, x)| r.all_reduce(x))
                .collect::<Result<Vec<_>>>()?;
            // Compared on the host: with peer access enabled, device-to-device copies of
            // pool memory would fault.
            let host = |t: &Tensor| t.to_dtype(DType::F32)?.to_device(&Device::Cpu);
            let want = (host(&inputs[0])? + host(&inputs[1])?)?;
            for out in &outs {
                let got = host(out)?;
                let err = (got - &want)?.abs()?.max_all()?.to_scalar::<f32>()?;
                // One bf16 rounding of the sum: a relative step of 2^-8.
                let scale = want.abs()?.max_all()?.to_scalar::<f32>()?;
                assert!(err <= scale / 128.0, "n={n} call={call}: max error {err} at scale {scale}");
            }
        }
    }
    Ok(())
}

#[test]
fn peer_allgather_stacks_ranks() -> Result<()> {
    let devices = [Device::new_rocm(0), Device::new_rocm(1)];
    let [Ok(d0), Ok(d1)] = devices else {
        return Ok(());
    };
    let setups = [&d0, &d1]
        .iter()
        .enumerate()
        .map(|(rank, d)| {
            PeerAllReduceSetup::new(d.as_rocm_device().unwrap(), rank, 2, 1 << 20).map(|(s, _)| s)
        })
        .collect::<Result<Vec<_>>>()?;
    let ranks: Vec<PeerAllReduce> = PeerAllReduceSetup::connect_in_process(setups)?;
    let host = |t: &Tensor| t.to_dtype(DType::F32)?.to_device(&Device::Cpu);
    for n in [16usize, 3 * 124160] {
        let inputs = [&d0, &d1]
            .iter()
            .enumerate()
            .map(|(rank, d)| {
                Tensor::arange(0u32, n as u32, d)?
                    .to_dtype(DType::F32)?
                    .affine(0.001, rank as f64 * 10.0)?
                    .to_dtype(DType::BF16)?
                    .reshape((1, n))
            })
            .collect::<Result<Vec<_>>>()?;
        let outs = ranks
            .iter()
            .zip(&inputs)
            .map(|(r, x)| r.all_gather(x))
            .collect::<Result<Vec<_>>>()?;
        let want = Tensor::cat(&[host(&inputs[0])?, host(&inputs[1])?], 0)?;
        for out in &outs {
            assert_eq!(out.dims(), [2, 1, n]);
            let err = (host(out)?.reshape((2, n))? - &want)?.abs()?.max_all()?.to_scalar::<f32>()?;
            assert_eq!(err, 0.0, "n={n}: gathered rows differ");
        }
    }
    Ok(())
}

/// Prints the time per call of a prefill-chunk-sized reduction (2048 x 2560 bf16, 10 MiB);
/// run with `--ignored --nocapture`.
#[test]
#[ignore]
fn peer_allreduce_wide_timing() -> Result<()> {
    let devices = [Device::new_rocm(0), Device::new_rocm(1)];
    let [Ok(d0), Ok(d1)] = devices else {
        return Ok(());
    };
    let setups = [&d0, &d1]
        .iter()
        .enumerate()
        .map(|(rank, d)| {
            PeerAllReduceSetup::new(d.as_rocm_device().unwrap(), rank, 2, 6 << 20).map(|(s, _)| s)
        })
        .collect::<Result<Vec<_>>>()?;
    let ranks: Vec<PeerAllReduce> = PeerAllReduceSetup::connect_in_process(setups)?;
    for n in [256 * 2560usize, 2048 * 2560] {
        let inputs = [&d0, &d1]
            .iter()
            .map(|d| Tensor::randn(0f32, 1.0, n, d)?.to_dtype(DType::BF16))
            .collect::<Result<Vec<_>>>()?;
        for _ in 0..5 {
            for (r, x) in ranks.iter().zip(&inputs) {
                r.all_reduce(x)?;
            }
        }
        d0.synchronize()?;
        d1.synchronize()?;
        let reps = 50;
        let start = std::time::Instant::now();
        for _ in 0..reps {
            for (r, x) in ranks.iter().zip(&inputs) {
                r.all_reduce(x)?;
            }
        }
        d0.synchronize()?;
        d1.synchronize()?;
        let us = start.elapsed().as_secs_f64() * 1e6 / reps as f64;
        println!("peer all-reduce of {n} bf16 ({:.1} MiB): {us:.0} us per call", n as f64 * 2.0 / 1048576.0);
    }
    Ok(())
}
