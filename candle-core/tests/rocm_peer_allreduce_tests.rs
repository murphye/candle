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
            PeerAllReduceSetup::new(d.as_rocm_device().unwrap(), rank, 2, 1 << 15).map(|(s, _)| s)
        })
        .collect::<Result<Vec<_>>>()?;
    let ranks: Vec<PeerAllReduce> = PeerAllReduceSetup::connect_in_process(setups)?;
    for n in [8usize, 3 * 2560, 8 * 2560] {
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
