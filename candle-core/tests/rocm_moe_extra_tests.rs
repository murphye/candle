#![cfg(feature = "rocm")]

use candle_core::quantized::rocm::{
    indexed_moe_forward_with_extra, indexed_moe_forward_with_extra_reduce,
    indexed_moe_glu_forward_with_extra,
};
use candle_core::quantized::{GgmlDType, QTensor};
use candle_core::{DType, Device, Result, Tensor};

fn weights(experts: usize, n: usize, k: usize, salt: f32) -> Result<Tensor> {
    Tensor::from_vec(
        (0..experts * n * k).map(|i| ((i as f32 * 0.0137 + salt).sin()) * 0.3).collect::<Vec<_>>(),
        (experts, n, k),
        &Device::Cpu,
    )
}

#[test]
fn extra_expert_matches_concatenated_table() -> Result<()> {
    let Ok(dev) = Device::new_rocm(0) else { return Ok(()) };
    let (experts, n, k, tokens, topk) = (4usize, 64usize, 512usize, 3usize, 3usize);
    for dtype in [GgmlDType::Q4K, GgmlDType::Q8_0, GgmlDType::Q5_0] {
        let routed = weights(experts, n, k, 0.1)?;
        let shared = weights(1, n, k, 0.7)?;
        let all = Tensor::cat(&[&routed, &shared], 0)?;
        let q = |w: &Tensor| QTensor::quantize_onto(w, dtype, &dev);
        let (q_routed, q_shared, q_all) = (q(&routed)?, q(&shared)?, q(&all)?);
        // The last slot of every token is the shared expert.
        let ids = Tensor::from_vec(
            vec![0u32, 2, 4, 3, 1, 4, 2, 2, 4],
            (tokens, topk),
            &dev,
        )?;
        // Gate/up style: one input row per token.
        let x = Tensor::from_vec(
            (0..tokens * k).map(|i| ((i as f32 * 0.021).cos()) * 0.5).collect::<Vec<_>>(),
            (tokens, 1, k),
            &dev,
        )?
        .to_dtype(DType::BF16)?;
        let got = indexed_moe_forward_with_extra(&q_routed, &q_shared, &x, &ids)?;
        let want = q_all.indexed_moe_forward(&x, &ids)?;
        let err = (got.to_dtype(DType::F32)? - want.to_dtype(DType::F32)?)?.abs()?.max_all()?.to_scalar::<f32>()?;
        assert_eq!(err, 0.0, "{dtype:?} plain: extra-expert output differs");
        // Down style: one input row per (token, slot).
        let xd = Tensor::from_vec(
            (0..tokens * topk * k).map(|i| ((i as f32 * 0.017).sin()) * 0.5).collect::<Vec<_>>(),
            (tokens, topk, k),
            &dev,
        )?
        .to_dtype(DType::BF16)?;
        let got = indexed_moe_forward_with_extra(&q_routed, &q_shared, &xd, &ids)?;
        let want = q_all.indexed_moe_forward(&xd, &ids)?;
        let err = (got.to_dtype(DType::F32)? - want.to_dtype(DType::F32)?)?.abs()?.max_all()?.to_scalar::<f32>()?;
        assert_eq!(err, 0.0, "{dtype:?} per-slot: extra-expert output differs");
        if dtype == GgmlDType::Q4K {
            let up = weights(experts, n, k, 1.3)?;
            let shared_up = weights(1, n, k, 1.9)?;
            let all_up = Tensor::cat(&[&up, &shared_up], 0)?;
            let got = indexed_moe_glu_forward_with_extra(&q_routed, &q(&up)?, 0, &q_shared, &q(&shared_up)?, &x, &ids)?
                .to_dtype(DType::F32)?;
            // Reference from two plain forwards; the fused kernel applies silu in f32.
            let g = q_all.indexed_moe_forward(&x, &ids)?.to_dtype(DType::F32)?;
            let u = q(&all_up)?.indexed_moe_forward(&x, &ids)?.to_dtype(DType::F32)?;
            let want = ((&g / (g.neg()?.exp()? + 1.0)?)? * u)?;
            let err = (got - &want)?.abs()?.max_all()?.to_scalar::<f32>()?;
            let scale = want.abs()?.max_all()?.to_scalar::<f32>()?;
            assert!(err <= scale / 64.0, "glu: extra-expert output differs by {err} at scale {scale}");
        }
    }
    Ok(())
}

/// The slot-reducing down kernel matches the per-slot outputs weighted and summed in f32.
#[test]
fn reducing_down_matches_weighted_sum() -> Result<()> {
    let Ok(dev) = Device::new_rocm(0) else { return Ok(()) };
    let (experts, n, k, tokens, topk) = (4usize, 64usize, 512usize, 3usize, 3usize);
    for dtype in [GgmlDType::Q8_0, GgmlDType::Q5_0, GgmlDType::Q4K] {
        let routed = weights(experts, n, k, 0.1)?;
        let shared = weights(1, n, k, 0.7)?;
        let q = |w: &Tensor| QTensor::quantize_onto(w, dtype, &dev);
        let (q_routed, q_shared) = (q(&routed)?, q(&shared)?);
        let ids = Tensor::from_vec(vec![0u32, 2, 4, 3, 1, 4, 2, 2, 4], (tokens, topk), &dev)?;
        let slot_weights = Tensor::from_vec(
            vec![0.5f32, 0.25, 1.0, 0.1, 0.6, 1.0, 0.3, 0.3, 1.0],
            (tokens, topk),
            &dev,
        )?;
        let xd = Tensor::from_vec(
            (0..tokens * topk * k).map(|i| ((i as f32 * 0.017).sin()) * 0.5).collect::<Vec<_>>(),
            (tokens, topk, k),
            &dev,
        )?
        .to_dtype(DType::BF16)?;
        let got = indexed_moe_forward_with_extra_reduce(&q_routed, &q_shared, &xd, &ids, &slot_weights)?
            .to_dtype(DType::F32)?;
        assert_eq!(got.dims(), [tokens, n]);
        let per_slot = indexed_moe_forward_with_extra(&q_routed, &q_shared, &xd, &ids)?.to_dtype(DType::F32)?;
        let want = per_slot.broadcast_mul(&slot_weights.unsqueeze(2)?)?.sum(1)?;
        let err = (got - &want)?.abs()?.max_all()?.to_scalar::<f32>()?;
        let scale = want.abs()?.max_all()?.to_scalar::<f32>()?.max(1e-3);
        // The per-slot reference rounds each slot to bf16 before the sum.
        assert!(err / scale < 2e-2, "{dtype:?}: rel err {}", err / scale);
    }
    Ok(())
}
