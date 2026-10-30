//! Checks the ROCm quantized matmul paths (MMVQ for small batches, dequantize
//! plus GEMM for larger ones) against the CPU implementation.
#![cfg(feature = "rocm")]

use candle_core::quantized::{GgmlDType, QMatMul, QTensor};
use candle_core::{DType, Device, Module, Result, Tensor};

fn max_rel_diff(a: &Tensor, b: &Tensor) -> Result<f32> {
    let a = a.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
    let b = b.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
    let scale = a.abs()?.max_all()?.to_scalar::<f32>()?.max(1e-6);
    Ok((a - b)?.abs()?.max_all()?.to_scalar::<f32>()? / scale)
}

#[test]
fn quantized_matmul_matches_cpu() -> Result<()> {
    let rocm = Device::new_rocm(0)?;
    let (n, k) = (96, 512);
    let w = Tensor::randn(0f32, 1.0, (n, k), &Device::Cpu)?;
    for dtype in [
        GgmlDType::Q4_0,
        GgmlDType::Q5_1,
        GgmlDType::Q8_0,
        GgmlDType::Q2K,
        GgmlDType::Q4K,
        GgmlDType::Q6K,
    ] {
        let cpu = QMatMul::from_qtensor(QTensor::quantize(&w, dtype)?)?;
        let gpu = QMatMul::from_qtensor(QTensor::quantize(&w.to_device(&rocm)?, dtype)?)?;
        // Batch sizes on both sides of the MMVQ / dequantize-GEMM switch.
        for m in [1usize, 3, 8, 9, 24, 64, 65] {
            for act in [DType::F32, DType::BF16, DType::F16] {
                let x = Tensor::randn(0f32, 1.0, (1, m, k), &Device::Cpu)?.to_dtype(act)?;
                let expected = cpu.forward(&x.to_dtype(DType::F32)?)?;
                let got = gpu.forward(&x.to_device(&rocm)?)?;
                assert_eq!(got.dtype(), act);
                assert_eq!(got.dims(), &[1, m, n]);
                let diff = max_rel_diff(&expected, &got)?;
                // q8_1 activations and half precision outputs bound the accuracy.
                assert!(diff < 0.05, "{dtype:?} m={m} {act:?}: relative diff {diff}");
            }
        }
    }
    Ok(())
}

#[test]
fn indexed_moe_forward_matches_cpu() -> Result<()> {
    // Few pairs use the fused kernel; many pairs use the grouped dequantize + GEMM path.
    indexed_moe_case(8, 64, 512, 5, 2)?;
    indexed_moe_case(16, 64, 512, 160, 10)?;
    // Partial output tiles and uneven expert loads in the grouped kernel (tall tiles).
    indexed_moe_case(24, 200, 768, 300, 8)?;
    // Few rows per expert: the grouped kernel's wide tiles.
    indexed_moe_case(64, 200, 512, 200, 8)
}

fn indexed_moe_case(
    num_experts: usize,
    n: usize,
    k: usize,
    tokens: usize,
    topk: usize,
) -> Result<()> {
    let rocm = Device::new_rocm(0)?;
    let w = Tensor::randn(0f32, 1.0, (num_experts, n, k), &Device::Cpu)?;
    let ids: Vec<u32> = (0..tokens * topk)
        .map(|i| ((i * 3 + i / topk) % num_experts) as u32)
        .collect();
    let ids_cpu = Tensor::from_vec(ids.clone(), (tokens, topk), &Device::Cpu)?;
    for dtype in [
        GgmlDType::Q4_0,
        GgmlDType::Q5_0,
        GgmlDType::Q5_1,
        GgmlDType::Q8_0,
        GgmlDType::Q4K,
        GgmlDType::Q6K,
    ] {
        let qw = QTensor::quantize(&w.to_device(&rocm)?, dtype)?;
        let w_deq = QTensor::quantize(&w, dtype)?.dequantize(&Device::Cpu)?;
        // One input row per token, shared by its experts, and one row per expert.
        for x_t in [1, topk] {
            let x = Tensor::randn(0f32, 1.0, (tokens, x_t, k), &Device::Cpu)?;
            let ids_gpu = ids_cpu.to_device(&rocm)?;
            let got = qw.indexed_moe_forward(&x.to_device(&rocm)?, &ids_gpu)?;
            assert_eq!(got.dims(), &[tokens, topk, n]);
            // Half precision activations come back in their own dtype.
            let mut half_outputs = Vec::new();
            for act in [DType::BF16, DType::F16] {
                let xh = x.to_dtype(act)?.to_device(&rocm)?;
                let out = qw.indexed_moe_forward(&xh, &ids_gpu)?;
                assert_eq!(out.dtype(), act);
                half_outputs.push((act, out));
            }
            let mut rows = Vec::new();
            for t in 0..tokens {
                for j in 0..topk {
                    let xr = x.get(t)?.get(if x_t == 1 { 0 } else { j })?;
                    let we = w_deq.get(ids[t * topk + j] as usize)?;
                    rows.push(we.matmul(&xr.unsqueeze(1)?)?.squeeze(1)?);
                }
            }
            let expected = Tensor::stack(&rows, 0)?.reshape((tokens, topk, n))?;
            let diff = max_rel_diff(&expected, &got)?;
            assert!(diff < 0.05, "{dtype:?} x_t={x_t} experts={num_experts} n={n} k={k} tokens={tokens}: relative diff {diff}");
            for (act, out) in half_outputs {
                let diff = max_rel_diff(&expected, &out)?;
                assert!(diff < 0.05, "{dtype:?} x_t={x_t} {act:?}: relative diff {diff}");
            }
        }
    }
    Ok(())
}

#[test]
fn indexed_moe_glu_matches_separate_projections() -> Result<()> {
    let rocm = Device::new_rocm(0)?;
    let (num_experts, n, k, tokens, topk) = (8, 64, 512, 3, 2);
    let ids: Vec<u32> = (0..tokens * topk).map(|i| ((i * 3 + 1) % num_experts) as u32).collect();
    let ids = Tensor::from_vec(ids, (tokens, topk), &rocm)?;
    for dtype in [GgmlDType::Q4K, GgmlDType::Q8_0, GgmlDType::Q5_0] {
        let gate = QTensor::quantize(&Tensor::randn(0f32, 1.0, (num_experts, n, k), &rocm)?, dtype)?;
        let up = QTensor::quantize(&Tensor::randn(0f32, 1.0, (num_experts, n, k), &rocm)?, dtype)?;
        let x = Tensor::randn(0f32, 1.0, (tokens, 1, k), &rocm)?.to_dtype(DType::BF16)?;
        let got = gate
            .rocm_indexed_moe_glu_forward(&up, 0, &x, &ids)?
            .expect("fused path");
        assert_eq!(got.dtype(), DType::BF16);
        let g = gate.indexed_moe_forward(&x, &ids)?.to_dtype(DType::F32)?;
        let u = up.indexed_moe_forward(&x, &ids)?.to_dtype(DType::F32)?;
        let expected = (g.silu()? * u)?;
        let diff = max_rel_diff(&expected, &got)?;
        assert!(diff < 0.02, "{dtype:?}: relative diff {diff}");
    }
    Ok(())
}

/// Back-to-back projections of one activation reuse its q8_1 quantization; any write
/// to the stream in between (an in-place update or a new tensor in a recycled buffer)
/// must force a fresh one.
#[test]
fn reused_activation_quantization_tracks_writes() -> Result<()> {
    let rocm = Device::new_rocm(0)?;
    let k = 256;
    let weights = [(64, 1.0), (32, -2.0), (48, 0.5)].map(|(n, scale)| {
        let w = (Tensor::randn(0f32, 1.0, (n, k), &Device::Cpu).unwrap() * scale).unwrap();
        let cpu = QMatMul::from_qtensor(QTensor::quantize(&w, GgmlDType::Q8_0).unwrap()).unwrap();
        let gpu = QMatMul::from_qtensor(
            QTensor::quantize(&w.to_device(&rocm).unwrap(), GgmlDType::Q8_0).unwrap(),
        )
        .unwrap();
        (cpu, gpu)
    });
    let check = |x: &Tensor| -> Result<()> {
        let host = x.to_device(&Device::Cpu)?;
        for (cpu, gpu) in &weights {
            let diff = max_rel_diff(&cpu.forward(&host)?, &gpu.forward(x)?)?;
            assert!(diff < 0.05, "relative diff {diff}");
        }
        Ok(())
    };

    let x = Tensor::randn(0f32, 1.0, (1, k), &rocm)?;
    check(&x)?;
    // Overwrite the same buffer in place.
    x.slice_set(&Tensor::randn(0f32, 1.0, (1, k), &rocm)?, 0, 0)?;
    check(&x)?;
    // Fresh tensors of the same size, likely landing in the dropped buffers.
    for _ in 0..4 {
        let x = Tensor::randn(0f32, 1.0, (1, k), &rocm)?;
        check(&x)?;
    }
    Ok(())
}

