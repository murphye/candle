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
        for m in [1usize, 3, 8, 9, 64] {
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
    indexed_moe_case(16, 64, 512, 160, 10)
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
            let got = qw.indexed_moe_forward(&x.to_device(&rocm)?, &ids_cpu.to_device(&rocm)?)?;
            assert_eq!(got.dims(), &[tokens, topk, n]);
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
            assert!(diff < 0.05, "{dtype:?} x_t={x_t}: relative diff {diff}");
        }
    }
    Ok(())
}
