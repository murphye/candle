//! ROCm implementation of quantized tensors, using the same llama.cpp derived
//! kernels as the CUDA backend compiled with hipcc.
use super::{GgmlDType, QStorage};
use crate::quantized::k_quants::GgmlType;
use crate::rocm_backend::hip::{HipSlice, HipView, LaunchConfig};
use crate::rocm_backend::kernels;
use crate::{backend::BackendDevice, rocm_backend::WrapErr};
use crate::{rocm_builder_arg as barg, Result, RocmDevice, RocmStorage};
use half::f16;

#[derive(Clone, Debug)]
struct PaddedHipSlice {
    inner: HipSlice<u8>,
    len: usize,
}

#[derive(Clone, Debug)]
pub struct QRocmStorage {
    data: PaddedHipSlice,
    dtype: GgmlDType,
    device: RocmDevice,
}

pub(crate) static FORCE_DMMV: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

pub fn set_force_dmmv(f: bool) {
    FORCE_DMMV.store(f, std::sync::atomic::Ordering::Relaxed)
}

pub const WARP_SIZE: usize = 32;
pub const MMQ_X_Q4_0_AMPERE: usize = 4;
pub const MMQ_Y_Q4_0_AMPERE: usize = 32;
pub const NWARPS_Q4_0_AMPERE: usize = 4;
pub const GGML_CUDA_MMV_X: usize = 32;
pub const GGML_CUDA_MMV_Y: usize = 1;
pub const CUDA_QUANTIZE_BLOCK_SIZE: usize = 256;
pub const CUDA_DEQUANTIZE_BLOCK_SIZE: usize = 256;
pub const CUDA_GET_ROWS_BLOCK_SIZE: usize = 256;
pub const MATRIX_ROW_PADDING: usize = 512;

fn ceil_div(p: usize, q: usize) -> usize {
    p.div_ceil(q)
}

fn pad(p: usize, q: usize) -> usize {
    ceil_div(p, q) * q
}

fn quantize_q8_1(
    src: &HipView<f32>,
    dst: &mut HipSlice<u8>,
    k: usize,
    ky: usize,
    dev: &RocmDevice,
) -> Result<()> {
    let kx_padded = pad(k, MATRIX_ROW_PADDING);
    let num_blocks = ceil_div(kx_padded, CUDA_QUANTIZE_BLOCK_SIZE);

    let total_rows = ky;
    // Get Q8_1 metadata.
    let q8_1_block_size = GgmlDType::Q8_1.block_size();
    let q8_1_type_size = GgmlDType::Q8_1.type_size();

    // Calculate the size of the output buffer in bytes.
    let num_blocks_per_row = kx_padded / q8_1_block_size;
    let dst_row_size_bytes = num_blocks_per_row * q8_1_type_size;

    const CHUNK_SIZE: usize = 65535; // gridDim.y limit
    let func = dev.get_or_load_func("quantize_q8_1", &kernels::QUANTIZED)?;

    let mut rows_processed = 0;
    while rows_processed < total_rows {
        // --- calculate the number of rows for this chunk ---
        let remaining_rows = total_rows - rows_processed;
        // This is our gridDim.y, now <= 65535
        let rows_in_chunk = std::cmp::min(CHUNK_SIZE, remaining_rows);

        // --- slice the source (f32) tensor by elements ---
        let src_start_elem = rows_processed * k;
        let src_num_elems = rows_in_chunk * k;
        let src_chunk = src.slice(src_start_elem..(src_start_elem + src_num_elems));

        // --- slice the destination (u8) tensor by bytes ---
        let dst_start_byte = rows_processed * dst_row_size_bytes;
        let dst_num_bytes = rows_in_chunk * dst_row_size_bytes;
        let dst_chunk = dst.slice(dst_start_byte..(dst_start_byte + dst_num_bytes));

        let cfg = LaunchConfig {
            grid_dim: (num_blocks as u32, rows_in_chunk as u32, 1),
            block_dim: (CUDA_QUANTIZE_BLOCK_SIZE as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        let mut builder = func.builder();
        builder.arg(&src_chunk);
        builder.arg(&dst_chunk);
        barg!(builder, k as i32, kx_padded as i32);
        unsafe { builder.launch(cfg) }.w()?;

        rows_processed += rows_in_chunk;
    }

    Ok(())
}

fn dequantize_f32(
    data: &PaddedHipSlice,
    dtype: GgmlDType,
    elem_count: usize,
    dev: &RocmDevice,
) -> Result<RocmStorage> {
    let nb = elem_count.div_ceil(256);
    let (kernel_name, is_k, block_dim, num_blocks) = match dtype {
        GgmlDType::Q4_0 => ("dequantize_block_q4_0_f32", false, 32, nb),
        GgmlDType::Q4_1 => ("dequantize_block_q4_1_f32", false, 32, nb),
        GgmlDType::Q5_0 => (
            "dequantize_block_q5_0_f32",
            false,
            CUDA_DEQUANTIZE_BLOCK_SIZE,
            ceil_div(elem_count, 2 * CUDA_DEQUANTIZE_BLOCK_SIZE),
        ),
        GgmlDType::Q5_1 => (
            "dequantize_block_q5_1_f32",
            false,
            CUDA_DEQUANTIZE_BLOCK_SIZE,
            ceil_div(elem_count, 2 * CUDA_DEQUANTIZE_BLOCK_SIZE),
        ),
        GgmlDType::Q8_0 => ("dequantize_block_q8_0_f32", false, 32, nb),
        GgmlDType::Q2K => ("dequantize_block_q2_K_f32", true, 64, nb),
        GgmlDType::Q3K => ("dequantize_block_q3_K_f32", true, 64, nb),
        GgmlDType::Q4K => ("dequantize_block_q4_K_f32", true, 32, nb),
        GgmlDType::Q5K => ("dequantize_block_q5_K_f32", true, 64, nb),
        GgmlDType::Q6K => ("dequantize_block_q6_K_f32", true, 64, nb),
        GgmlDType::Q8K => ("dequantize_block_q8_K_f32", true, 32, nb),
        _ => crate::bail!("unsupported dtype for dequantize {dtype:?}"),
    };
    let func = dev.get_or_load_func(kernel_name, &kernels::QUANTIZED)?;
    let dst = unsafe { dev.alloc::<f32>(elem_count)? };
    // See e.g.
    // https://github.com/ggerganov/llama.cpp/blob/cbbd1efa06f8c09f9dff58ff9d9af509cc4c152b/ggml-cuda.cu#L7270
    let cfg = LaunchConfig {
        grid_dim: (num_blocks as u32, 1, 1),
        block_dim: (block_dim as u32, 1, 1),
        shared_mem_bytes: 0,
    };

    if is_k {
        let mut builder = func.builder();
        builder.arg(&data.inner);
        builder.arg(&dst);
        unsafe { builder.launch(cfg) }.w()?;
    } else {
        let nb32 = match dtype {
            GgmlDType::Q5_0 | GgmlDType::Q5_1 => elem_count,
            _ => elem_count / 32,
        };
        let mut builder = func.builder();
        builder.arg(&data.inner);
        builder.arg(&dst);
        barg!(builder, nb32 as i32);
        unsafe { builder.launch(cfg) }.w()?;
    }
    Ok(RocmStorage::wrap_rocm_slice(dst, dev.clone()))
}

fn dequantize_f16(
    data: &PaddedHipSlice,
    dtype: GgmlDType,
    elem_count: usize,
    dev: &RocmDevice,
) -> Result<RocmStorage> {
    let nb = elem_count.div_ceil(256);
    let (kernel_name, is_k, block_dim, num_blocks) = match dtype {
        GgmlDType::Q4_0 => ("dequantize_block_q4_0_f16", false, 32, nb),
        GgmlDType::Q4_1 => ("dequantize_block_q4_1_f16", false, 32, nb),
        GgmlDType::Q5_0 => (
            "dequantize_block_q5_0_f16",
            false,
            CUDA_DEQUANTIZE_BLOCK_SIZE,
            ceil_div(elem_count, 2 * CUDA_DEQUANTIZE_BLOCK_SIZE),
        ),
        GgmlDType::Q5_1 => (
            "dequantize_block_q5_1_f16",
            false,
            CUDA_DEQUANTIZE_BLOCK_SIZE,
            ceil_div(elem_count, 2 * CUDA_DEQUANTIZE_BLOCK_SIZE),
        ),
        GgmlDType::Q8_0 => ("dequantize_block_q8_0_f16", false, 32, nb),
        GgmlDType::Q2K => ("dequantize_block_q2_K_f16", true, 64, nb),
        GgmlDType::Q3K => ("dequantize_block_q3_K_f16", true, 64, nb),
        GgmlDType::Q4K => ("dequantize_block_q4_K_f16", true, 32, nb),
        GgmlDType::Q5K => ("dequantize_block_q5_K_f16", true, 64, nb),
        GgmlDType::Q6K => ("dequantize_block_q6_K_f16", true, 64, nb),
        GgmlDType::Q8K => ("dequantize_block_q8_K_f16", true, 32, nb),
        _ => crate::bail!("unsupported dtype for dequantize {dtype:?}"),
    };
    let func = dev.get_or_load_func(kernel_name, &kernels::QUANTIZED)?;
    let dst = unsafe { dev.alloc::<f16>(elem_count)? };
    // See e.g.
    // https://github.com/ggerganov/llama.cpp/blob/cbbd1efa06f8c09f9dff58ff9d9af509cc4c152b/ggml-cuda.cu#L7270
    let cfg = LaunchConfig {
        grid_dim: (num_blocks as u32, 1, 1),
        block_dim: (block_dim as u32, 1, 1),
        shared_mem_bytes: 0,
    };

    if is_k {
        let mut builder = func.builder();
        builder.arg(&data.inner);
        builder.arg(&dst);
        unsafe { builder.launch(cfg) }.w()?;
    } else {
        let nb32 = match dtype {
            GgmlDType::Q5_0 | GgmlDType::Q5_1 => elem_count,
            _ => elem_count / 32,
        };
        let mut builder = func.builder();
        builder.arg(&data.inner);
        builder.arg(&dst);
        barg!(builder, nb32 as i32);
        unsafe { builder.launch(cfg) }.w()?;
    }
    Ok(RocmStorage::wrap_rocm_slice(dst, dev.clone()))
}

fn dequantize_bf16(
    data: &PaddedHipSlice,
    dtype: GgmlDType,
    elem_count: usize,
    dev: &RocmDevice,
) -> Result<RocmStorage> {
    // Q8_0 and Q5_0 have vectorized kernels: one thread per eight outputs.
    if matches!(dtype, GgmlDType::Q8_0 | GgmlDType::Q5_0) && elem_count % 32 == 0 {
        let name = match dtype {
            GgmlDType::Q8_0 => "dequantize_v8_q8_0_bf16",
            _ => "dequantize_v8_q5_0_bf16",
        };
        let func = dev.get_or_load_func(name, &kernels::QUANTIZED)?;
        let dst = unsafe { dev.alloc::<half::bf16>(elem_count)? };
        let n_groups = elem_count / 8;
        let cfg = LaunchConfig {
            grid_dim: (n_groups.div_ceil(256) as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = func.builder();
        builder.arg(&data.inner);
        builder.arg(&dst);
        barg!(builder, n_groups as i64);
        unsafe { builder.launch(cfg) }.w()?;
        return Ok(RocmStorage::wrap_rocm_slice(dst, dev.clone()));
    }
    let nb = elem_count.div_ceil(256);
    let (kernel_name, is_k, block_dim, num_blocks) = match dtype {
        GgmlDType::Q4_0 => ("dequantize_block_q4_0_bf16", false, 32, nb),
        GgmlDType::Q4_1 => ("dequantize_block_q4_1_bf16", false, 32, nb),
        GgmlDType::Q5_0 => (
            "dequantize_block_q5_0_bf16",
            false,
            CUDA_DEQUANTIZE_BLOCK_SIZE,
            ceil_div(elem_count, 2 * CUDA_DEQUANTIZE_BLOCK_SIZE),
        ),
        GgmlDType::Q5_1 => (
            "dequantize_block_q5_1_bf16",
            false,
            CUDA_DEQUANTIZE_BLOCK_SIZE,
            ceil_div(elem_count, 2 * CUDA_DEQUANTIZE_BLOCK_SIZE),
        ),
        GgmlDType::Q8_0 => ("dequantize_block_q8_0_bf16", false, 32, nb),
        GgmlDType::Q2K => ("dequantize_block_q2_K_bf16", true, 64, nb),
        GgmlDType::Q3K => ("dequantize_block_q3_K_bf16", true, 64, nb),
        GgmlDType::Q4K => ("dequantize_block_q4_K_bf16", true, 32, nb),
        GgmlDType::Q5K => ("dequantize_block_q5_K_bf16", true, 64, nb),
        GgmlDType::Q6K => ("dequantize_block_q6_K_bf16", true, 64, nb),
        GgmlDType::Q8K => ("dequantize_block_q8_K_bf16", true, 32, nb),
        _ => crate::bail!("unsupported dtype for dequantize {dtype:?}"),
    };
    let func = dev.get_or_load_func(kernel_name, &kernels::QUANTIZED)?;
    let dst = unsafe { dev.alloc::<half::bf16>(elem_count)? };
    // See e.g.
    // https://github.com/ggerganov/llama.cpp/blob/cbbd1efa06f8c09f9dff58ff9d9af509cc4c152b/ggml-cuda.cu#L7270
    let cfg = LaunchConfig {
        grid_dim: (num_blocks as u32, 1, 1),
        block_dim: (block_dim as u32, 1, 1),
        shared_mem_bytes: 0,
    };

    if is_k {
        let mut builder = func.builder();
        builder.arg(&data.inner);
        builder.arg(&dst);
        unsafe { builder.launch(cfg) }.w()?;
    } else {
        let nb32 = match dtype {
            GgmlDType::Q5_0 | GgmlDType::Q5_1 => elem_count,
            _ => elem_count / 32,
        };
        let mut builder = func.builder();
        builder.arg(&data.inner);
        builder.arg(&dst);
        barg!(builder, nb32 as i32);
        unsafe { builder.launch(cfg) }.w()?;
    }
    Ok(RocmStorage::wrap_rocm_slice(dst, dev.clone()))
}

fn get_rows(
    data: &PaddedHipSlice,
    dtype: GgmlDType,
    hidden: usize,
    ids: &HipView<u32>,
    dev: &RocmDevice,
) -> Result<RocmStorage> {
    let (kernel_name, block_dim, block_num_y, can_stride_y) = match dtype {
        GgmlDType::F32 => (
            "get_rows_f32",
            CUDA_GET_ROWS_BLOCK_SIZE,
            ceil_div(hidden, CUDA_GET_ROWS_BLOCK_SIZE),
            true,
        ),
        GgmlDType::F16 => (
            "get_rows_f16",
            CUDA_GET_ROWS_BLOCK_SIZE,
            ceil_div(hidden, CUDA_GET_ROWS_BLOCK_SIZE),
            true,
        ),
        GgmlDType::BF16 => (
            "get_rows_bf16",
            CUDA_GET_ROWS_BLOCK_SIZE,
            ceil_div(hidden, CUDA_GET_ROWS_BLOCK_SIZE),
            true,
        ),
        GgmlDType::Q4_0 => (
            "get_rows_q4_0",
            CUDA_GET_ROWS_BLOCK_SIZE,
            ceil_div(hidden, 2 * CUDA_GET_ROWS_BLOCK_SIZE),
            true,
        ),
        GgmlDType::Q4_1 => (
            "get_rows_q4_1",
            CUDA_GET_ROWS_BLOCK_SIZE,
            ceil_div(hidden, 2 * CUDA_GET_ROWS_BLOCK_SIZE),
            true,
        ),
        GgmlDType::Q5_0 => (
            "get_rows_q5_0",
            CUDA_GET_ROWS_BLOCK_SIZE,
            ceil_div(hidden, 2 * CUDA_GET_ROWS_BLOCK_SIZE),
            true,
        ),
        GgmlDType::Q5_1 => (
            "get_rows_q5_1",
            CUDA_GET_ROWS_BLOCK_SIZE,
            ceil_div(hidden, 2 * CUDA_GET_ROWS_BLOCK_SIZE),
            true,
        ),
        GgmlDType::Q8_0 => (
            "get_rows_q8_0",
            CUDA_GET_ROWS_BLOCK_SIZE,
            ceil_div(hidden, 2 * CUDA_GET_ROWS_BLOCK_SIZE),
            true,
        ),
        GgmlDType::Q2K => ("get_rows_q2_K", 64, hidden / dtype.block_size(), false),
        GgmlDType::Q3K => ("get_rows_q3_K", 64, hidden / dtype.block_size(), false),
        GgmlDType::Q4K => ("get_rows_q4_K", 32, hidden / dtype.block_size(), false),
        GgmlDType::Q5K => ("get_rows_q5_K", 64, hidden / dtype.block_size(), false),
        GgmlDType::Q6K => ("get_rows_q6_K", 64, hidden / dtype.block_size(), false),
        _ => crate::bail!("unsupported dtype for ROCm quantized embedding {dtype:?}"),
    };
    let func = dev.get_or_load_func(kernel_name, &kernels::QUANTIZED)?;
    let ids_len = ids.len();
    let dst = unsafe { dev.alloc::<f32>(ids_len * hidden)? };
    if ids_len == 0 {
        return Ok(RocmStorage::wrap_rocm_slice(dst, dev.clone()));
    }
    if !can_stride_y && block_num_y > u16::MAX as usize {
        crate::bail!("quantized embedding hidden size {hidden} exceeds grid y limit")
    }
    let grid_y = if can_stride_y {
        block_num_y.min(u16::MAX as usize)
    } else {
        block_num_y
    };
    let cfg = LaunchConfig {
        grid_dim: (ids_len as u32, grid_y as u32, 1),
        block_dim: (block_dim as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    let row_stride = hidden * dtype.type_size() / dtype.block_size();

    let mut builder = func.builder();
    builder.arg(&data.inner);
    builder.arg(ids);
    builder.arg(&dst);
    barg!(builder, hidden as i64, row_stride);
    unsafe { builder.launch(cfg) }.w()?;
    Ok(RocmStorage::wrap_rocm_slice(dst, dev.clone()))
}

fn dequantize_mul_mat_vec(
    data: &PaddedHipSlice,
    y: &HipView<f32>,
    dtype: GgmlDType,
    ncols: usize,
    nrows: usize,
    dev: &RocmDevice,
) -> Result<RocmStorage> {
    let data_elems = data.len / dtype.type_size() * dtype.block_size();
    if data_elems < ncols * nrows {
        crate::bail!("unexpected data size {}, ncols {ncols} {nrows}", data_elems)
    }
    if y.len() != ncols {
        crate::bail!("unexpected y size {}, ncols {ncols} {nrows}", y.len())
    }
    let kernel_name = match dtype {
        GgmlDType::Q4_0 => "dequantize_mul_mat_vec_q4_0_cuda",
        GgmlDType::Q4_1 => "dequantize_mul_mat_vec_q4_1_cuda",
        GgmlDType::Q5_0 => "dequantize_mul_mat_vec_q5_0_cuda",
        GgmlDType::Q5_1 => "dequantize_mul_mat_vec_q5_1_cuda",
        GgmlDType::Q8_0 => "dequantize_mul_mat_vec_q8_0_cuda",
        GgmlDType::Q2K => "dequantize_mul_mat_vec_q2_k",
        GgmlDType::Q3K => "dequantize_mul_mat_vec_q3_k",
        GgmlDType::Q4K => "dequantize_mul_mat_vec_q4_k",
        GgmlDType::Q5K => "dequantize_mul_mat_vec_q5_k",
        GgmlDType::Q6K => "dequantize_mul_mat_vec_q6_k",
        _ => crate::bail!("unsupported dtype for quantized matmul {dtype:?}"),
    };
    let func = dev.get_or_load_func(kernel_name, &kernels::QUANTIZED)?;
    let dst = unsafe { dev.alloc::<f32>(nrows)? };
    let block_num_y = ceil_div(nrows, GGML_CUDA_MMV_Y);
    let cfg = LaunchConfig {
        grid_dim: (block_num_y as u32, 1, 1),
        block_dim: (WARP_SIZE as u32, GGML_CUDA_MMV_Y as u32, 1),
        shared_mem_bytes: 0,
    };

    let mut builder = func.builder();
    builder.arg(&data.inner);
    builder.arg(y);
    builder.arg(&dst);
    barg!(builder, ncols as i32, nrows as i32);
    unsafe { builder.launch(cfg) }.w()?;
    Ok(RocmStorage::wrap_rocm_slice(dst, dev.clone()))
}

fn mul_mat_vec_via_q8_1(
    data: &PaddedHipSlice,
    y: &HipView<f32>,
    dtype: GgmlDType,
    ncols: usize,
    nrows: usize,
    b_size: usize,
    dev: &RocmDevice,
) -> Result<RocmStorage> {
    let data_elems = data.len / dtype.type_size() * dtype.block_size();
    if data_elems < ncols * nrows {
        crate::bail!("unexpected data size {}, ncols {ncols} {nrows}", data_elems)
    }
    if y.len() != ncols * b_size {
        crate::bail!("unexpected y size {}, ncols {ncols} {nrows}", y.len())
    }
    if b_size == 0 || b_size > 8 {
        crate::bail!("only bsize between 1 and 8 are supported, got {b_size}")
    }
    // Start by quantizing y
    let ncols_padded = pad(ncols, MATRIX_ROW_PADDING);
    let y_size_in_bytes =
        b_size * ncols_padded * GgmlDType::Q8_1.type_size() / GgmlDType::Q8_1.block_size();
    let mut y_q8_1 = dev.alloc_zeros::<u8>(y_size_in_bytes)?;
    quantize_q8_1(y, &mut y_q8_1, ncols, b_size, dev)?;

    let kernel_name = match dtype {
        GgmlDType::Q4_0 => "mul_mat_vec_q4_0_q8_1_cuda",
        GgmlDType::Q4_1 => "mul_mat_vec_q4_1_q8_1_cuda",
        GgmlDType::Q5_0 => "mul_mat_vec_q5_0_q8_1_cuda",
        GgmlDType::Q5_1 => "mul_mat_vec_q5_1_q8_1_cuda",
        GgmlDType::Q8_0 => "mul_mat_vec_q8_0_q8_1_cuda",
        GgmlDType::Q2K => "mul_mat_vec_q2_K_q8_1_cuda",
        GgmlDType::Q3K => "mul_mat_vec_q3_K_q8_1_cuda",
        GgmlDType::Q4K => "mul_mat_vec_q4_K_q8_1_cuda",
        GgmlDType::Q5K => "mul_mat_vec_q5_K_q8_1_cuda",
        GgmlDType::Q6K => "mul_mat_vec_q6_K_q8_1_cuda",
        _ => crate::bail!("unsupported dtype for quantized matmul {dtype:?}"),
    };
    let kernel_name = format!("{kernel_name}{b_size}");
    let func = dev.get_or_load_func(&kernel_name, &kernels::QUANTIZED)?;
    let dst = dev.alloc_zeros::<f32>(nrows * b_size)?;
    // https://github.com/ggerganov/llama.cpp/blob/facb8b56f8fd3bb10a693bf0943ae9d69d0828ef/ggml-cuda/mmvq.cu#L98
    let (nblocks, nwarps) = match b_size {
        1 => (nrows as u32, 4),
        2..=4 => ((nrows as u32).div_ceil(2), 4),
        5..=8 => ((nrows as u32).div_ceil(2), 2),
        _ => crate::bail!("unexpected bsize {b_size}"),
    };
    let cfg = LaunchConfig {
        grid_dim: (nblocks, 1, 1),
        block_dim: (WARP_SIZE as u32, nwarps, 1),
        shared_mem_bytes: 0,
    };

    let mut builder = func.builder();
    builder.arg(&data.inner);
    builder.arg(&y_q8_1);
    builder.arg(&dst);
    barg!(
        builder,
        /* ncols_x */ ncols as i32,
        /* nrows_x */ nrows as i32,
        /* nrows_y */ ncols_padded as i32,
        /* nrows_dst */ nrows as i32
    );
    unsafe { builder.launch(cfg) }.w()?;
    Ok(RocmStorage::wrap_rocm_slice(dst, dev.clone()))
}

#[allow(clippy::too_many_arguments)]
fn mul_mat_via_q8_1(
    data: &PaddedHipSlice,
    y: &HipView<f32>,
    dtype: GgmlDType,
    x_rows: usize,
    x_cols: usize,
    y_rows: usize,
    y_cols: usize,
    dev: &RocmDevice,
) -> Result<RocmStorage> {
    let data_elems = data.len / dtype.type_size() * dtype.block_size();
    if data_elems < x_rows * x_cols {
        crate::bail!("unexpected lhs size {}, {x_rows} {x_cols}", data_elems)
    }
    if y.len() != y_rows * y_cols {
        crate::bail!("unexpected y size {}, {y_rows} {y_cols}", y.len())
    }
    if x_cols != y_rows {
        crate::bail!("unexpected x/y size {x_rows} {x_cols} {y_rows} {y_cols}")
    }
    let k = x_cols;
    // Start by quantizing y
    let k_padded = pad(k, MATRIX_ROW_PADDING);
    let y_size_in_bytes =
        k_padded * y_cols * GgmlDType::Q8_1.type_size() / GgmlDType::Q8_1.block_size();
    let mut y_q8_1 = dev.alloc_zeros::<u8>(y_size_in_bytes)?;
    quantize_q8_1(y, &mut y_q8_1, k, y_cols, dev)?;

    let (kernel_name, mmq_x, mmq_y) = match dtype {
        GgmlDType::Q4_0 => ("mul_mat_q4_0", 64, 128),
        GgmlDType::Q4_1 => ("mul_mat_q4_1", 64, 128),
        GgmlDType::Q5_0 => ("mul_mat_q5_0", 128, 64),
        GgmlDType::Q5_1 => ("mul_mat_q5_1", 128, 64),
        GgmlDType::Q8_0 => ("mul_mat_q8_0", 128, 64),
        GgmlDType::Q2K => ("mul_mat_q2_K", 64, 128),
        GgmlDType::Q3K => ("mul_mat_q3_K", 128, 128),
        GgmlDType::Q4K => ("mul_mat_q4_K", 64, 128),
        GgmlDType::Q5K => ("mul_mat_q5_K", 64, 128),
        GgmlDType::Q6K => ("mul_mat_q6_K", 64, 64),
        _ => crate::bail!("unsupported dtype for quantized matmul {dtype:?}"),
    };
    let func = dev.get_or_load_func(kernel_name, &kernels::QUANTIZED)?;
    let dst = dev.alloc_zeros::<f32>(x_rows * y_cols)?;
    let cfg = LaunchConfig {
        grid_dim: (
            ceil_div(x_rows, mmq_y) as u32,
            ceil_div(y_cols, mmq_x) as u32,
            1,
        ),
        block_dim: (WARP_SIZE as u32, 4, 1),
        shared_mem_bytes: 0,
    };

    let mut builder = func.builder();
    builder.arg(/* vx */ &data.inner);
    builder.arg(/* vy */ &y_q8_1);
    builder.arg(/* dst */ &dst);
    barg!(
        builder,
        /* ncols_x */ x_cols as i32,
        /* nrows_x */ x_rows as i32,
        /* ncols_y */ y_cols as i32,
        /* nrows_y */ k_padded as i32,
        /* nrows_dst */ x_rows as i32
    );
    unsafe { builder.launch(cfg) }.w()?;
    Ok(RocmStorage::wrap_rocm_slice(dst, dev.clone()))
}

/// Kernel name suffix of the indexed MoE kernels for a weight type.
fn indexed_moe_tag(dtype: GgmlDType) -> Option<&'static str> {
    Some(match dtype {
        GgmlDType::Q4_0 => "q4_0",
        GgmlDType::Q4_1 => "q4_1",
        GgmlDType::Q5_0 => "q5_0",
        GgmlDType::Q5_1 => "q5_1",
        GgmlDType::Q8_0 => "q8_0",
        GgmlDType::Q2K => "q2k",
        GgmlDType::Q3K => "q3k",
        GgmlDType::Q4K => "q4k",
        GgmlDType::Q5K => "q5k",
        GgmlDType::Q6K => "q6k",
        _ => return None,
    })
}

/// Whether `dev`'s code object carries the WMMA grouped MoE kernels (gfx11 and gfx12).
fn moe_gemm_available(dev: &RocmDevice) -> Result<bool> {
    moe_gemm_probe(dev, "moe_gemm_available")
}

/// Whether `dev`'s code object carries the int8 WMMA grouped MoE kernels (gfx12).
fn moe_gemm_i8_available(dev: &RocmDevice) -> Result<bool> {
    moe_gemm_probe(dev, "moe_gemm_i8_available")
}

fn moe_gemm_probe(dev: &RocmDevice, probe: &'static str) -> Result<bool> {
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};
    static AVAILABLE: OnceLock<Mutex<HashMap<(usize, &'static str), bool>>> = OnceLock::new();
    let cache = AVAILABLE.get_or_init(Default::default);
    if let Some(&known) = cache.lock().unwrap().get(&(dev.ordinal(), probe)) {
        return Ok(known);
    }
    let flag = dev.alloc_zeros::<i32>(1)?;
    let func = dev.get_or_load_func(probe, &kernels::MOE_GEMM)?;
    let mut builder = func.builder();
    builder.arg(&flag);
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe { builder.launch(cfg) }.w()?;
    let known = dev.clone_dtoh(&flag)?[0] == 1;
    cache.lock().unwrap().insert((dev.ordinal(), probe), known);
    Ok(known)
}

/// Whether the grouped MoE GEMM multiplies int8-quantized activations on the int8 WMMA for
/// this weight type. Off by default: once the bf16 tile's f32 -> bf16 rounding stopped going
/// through `__float2bfloat16` (a branchy NaN check per weight on gfx1201), the bf16 path
/// beat the int8 tile for Q8_0 as well (0.89 vs 1.06 ms for the Flash Next down projection at
/// 2048 tokens). `CANDLE_ROCM_MOE_INT8=q8` keeps the int8 tile for Q8_0, `all` adds Q4_K.
fn moe_gemm_int8_enabled(dtype: GgmlDType) -> bool {
    match std::env::var("CANDLE_ROCM_MOE_INT8").as_deref() {
        Ok("q8") => dtype == GgmlDType::Q8_0,
        Ok("all") => matches!(dtype, GgmlDType::Q4K | GgmlDType::Q8_0),
        _ => false,
    }
}

impl QRocmStorage {
    /// `x @ w[ids]^T` for weights `[num_experts, n, k]`, input `[batch, topk or 1, k]`
    /// and ids `[batch, topk]`, returning `[batch, topk, n]` in the input dtype.
    pub fn indexed_moe_forward(
        &self,
        self_shape: &crate::Shape,
        input: &RocmStorage,
        input_l: &crate::Layout,
        ids: &RocmStorage,
        ids_l: &crate::Layout,
    ) -> Result<(RocmStorage, crate::Shape)> {
        self.indexed_moe(None, None, None, self_shape, input, input_l, ids, ids_l)
    }

    /// `act(x @ self[ids]^T) * (x @ up[ids]^T)` in one kernel, for GLU experts whose
    /// gate (`self`) and up weights share shape and type. `act`: 0 silu, 1 gelu (tanh).
    #[allow(clippy::too_many_arguments)]
    pub fn indexed_moe_glu_forward(
        &self,
        up: &QRocmStorage,
        act: i32,
        self_shape: &crate::Shape,
        input: &RocmStorage,
        input_l: &crate::Layout,
        ids: &RocmStorage,
        ids_l: &crate::Layout,
    ) -> Result<(RocmStorage, crate::Shape)> {
        if up.dtype() != self.dtype() || up.data.len != self.data.len {
            crate::bail!("indexed_moe_glu_forward: gate and up weights differ");
        }
        self.indexed_moe(Some((up, act)), None, None, self_shape, input, input_l, ids, ids_l)
    }

    /// Grouped GEMM over the experts for prefill-sized batches: `x: [tokens, 1 | topk, k]`,
    /// `ids: [tokens, topk]` u32, output `[tokens, topk, n]` in x's dtype. Pairs are sorted by
    /// expert on the device and every expert's weights are dequantized tile by tile into LDS,
    /// so nothing waits on the host and no dequantized copy of the experts is made.
    /// `None` when the weight type or device has no kernel.
    pub fn moe_gemm_forward(
        &self,
        self_shape: &crate::Shape,
        input: &RocmStorage,
        input_l: &crate::Layout,
        ids: &RocmStorage,
        ids_l: &crate::Layout,
    ) -> Result<Option<(RocmStorage, crate::Shape)>> {
        use crate::backend::BackendStorage;
        use crate::rocm_backend::RocmStorageSlice as S;
        const BK: usize = 64;
        const MAX_EXPERTS: usize = 1024;
        let tag = match self.dtype() {
            GgmlDType::Q4_0 => "q4_0",
            GgmlDType::Q5_0 => "q5_0",
            GgmlDType::Q8_0 => "q8_0",
            GgmlDType::Q3K => "q3k",
            GgmlDType::Q4K => "q4k",
            GgmlDType::Q6K => "q6k",
            _ => return Ok(None),
        };
        let (experts, n, k) = self_shape.dims3()?;
        let (batch, input_dim1, in_k) = input_l.shape().dims3()?;
        let (id_batch, topk) = ids_l.shape().dims2()?;
        if in_k != k
            || id_batch != batch
            || (input_dim1 != 1 && input_dim1 != topk)
            || k % BK != 0
            || experts > MAX_EXPERTS
            || ids.dtype() != crate::DType::U32
        {
            return Ok(None);
        }
        let (Some((x1, x2)), Some((i1, i2))) =
            (input_l.contiguous_offsets(), ids_l.contiguous_offsets())
        else {
            return Ok(None);
        };
        let dev = &self.device;
        if !moe_gemm_available(dev)? {
            return Ok(None);
        }
        let (dst_tag, x_ptr) = match &input.slice {
            S::BF16(s) => ("bf16", s.slice(x1..x2).device_ptr()),
            S::F16(s) => ("f16", s.slice(x1..x2).device_ptr()),
            S::F32(s) => ("f32", s.slice(x1..x2).device_ptr()),
            _ => return Ok(None),
        };
        let ids_ptr = ids.as_rocm_slice::<u32>()?.slice(i1..i2).device_ptr();
        let pairs = batch * topk;
        // Tall tiles once experts average 64+ rows: each weight is then dequantized
        // about once; below that, wide tiles waste less on padding rows.
        let (shape, bm, bn) = if pairs >= 64 * experts {
            ("tall", 128, 64)
        } else {
            ("wide", 32, 128)
        };
        let max_tiles = pairs.div_ceil(bm) + experts;

        let sorted = unsafe { dev.alloc::<u32>(pairs)? };
        let offsets = unsafe { dev.alloc::<u32>(experts + 1)? };
        let tiles = unsafe { dev.alloc::<u32>(2 * max_tiles)? };
        let n_tiles = unsafe { dev.alloc::<u32>(1)? };
        let route = dev.get_or_load_func("moe_route", &kernels::MOE_GEMM)?;
        let mut builder = route.builder();
        builder.arg(&ids_ptr);
        barg!(builder, pairs as i32, experts as i32, bm as i32);
        builder.arg(&sorted);
        builder.arg(&offsets);
        builder.arg(&tiles);
        builder.arg(&n_tiles);
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (1024, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { builder.launch(cfg) }.w()?;

        let out_len = pairs * n;
        let out = match input.dtype() {
            crate::DType::BF16 => S::BF16(unsafe { dev.alloc::<half::bf16>(out_len)? }),
            crate::DType::F16 => S::F16(unsafe { dev.alloc::<f16>(out_len)? }),
            _ => S::F32(unsafe { dev.alloc::<f32>(out_len)? }),
        };
        let out_ptr = match &out {
            S::BF16(s) => s.device_ptr(),
            S::F16(s) => s.device_ptr(),
            S::F32(s) => s.device_ptr(),
            _ => unreachable!(),
        };
        let cfg = LaunchConfig {
            grid_dim: (n.div_ceil(bn) as u32, max_tiles as u32, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let int8 = k % 32 == 0 && moe_gemm_int8_enabled(self.dtype()) && moe_gemm_i8_available(dev)?;
        if int8 {
            // The activation rows quantized to int8 per 32-element block, with {d, d * sum(q)}.
            let x_rows = batch * input_dim1;
            let xq = unsafe { dev.alloc::<u8>(x_rows * k)? };
            let xm = unsafe { dev.alloc::<f32>(x_rows * (k / 32) * 2)? };
            let quant = dev.get_or_load_func(&format!("moe_quant_x_q8_{dst_tag}"), &kernels::MOE_GEMM)?;
            let mut builder = quant.builder();
            builder.arg(&x_ptr);
            builder.arg(&xq);
            builder.arg(&xm);
            barg!(builder, k as i32);
            let qcfg = LaunchConfig {
                grid_dim: (x_rows as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe { builder.launch(qcfg) }.w()?;
            let func = dev
                .get_or_load_func(&format!("moe_gemm_i8_{tag}_{dst_tag}_{shape}"), &kernels::MOE_GEMM)?;
            let mut builder = func.builder();
            builder.arg(&self.data.inner);
            builder.arg(&xq);
            builder.arg(&xm);
            builder.arg(&out_ptr);
            builder.arg(&sorted);
            builder.arg(&offsets);
            builder.arg(&tiles);
            builder.arg(&n_tiles);
            barg!(builder, n as i32, k as i32, topk as i32, (input_dim1 != 1) as i32);
            unsafe { builder.launch(cfg) }.w()?;
        } else {
            let func =
                dev.get_or_load_func(&format!("moe_gemm_{tag}_{dst_tag}_{shape}"), &kernels::MOE_GEMM)?;
            let mut builder = func.builder();
            builder.arg(&self.data.inner);
            builder.arg(&x_ptr);
            builder.arg(&out_ptr);
            builder.arg(&sorted);
            builder.arg(&offsets);
            builder.arg(&tiles);
            builder.arg(&n_tiles);
            barg!(builder, n as i32, k as i32, topk as i32, (input_dim1 != 1) as i32);
            unsafe { builder.launch(cfg) }.w()?;
        }
        Ok(Some((
            RocmStorage {
                slice: out,
                device: dev.clone(),
            },
            crate::Shape::from_dims(&[batch, topk, n]),
        )))
    }

    /// `reduce`: the f32 `[batch, topk]` slot weights' device address; the launch then sums
    /// each token's slots in the kernel and returns `[batch, 1, n]` (needs `extra`, no `glu`).
    fn indexed_moe(
        &self,
        glu: Option<(&QRocmStorage, i32)>,
        extra: Option<(&QRocmStorage, Option<&QRocmStorage>, usize)>,
        reduce: Option<u64>,
        self_shape: &crate::Shape,
        input: &RocmStorage,
        input_l: &crate::Layout,
        ids: &RocmStorage,
        ids_l: &crate::Layout,
    ) -> Result<(RocmStorage, crate::Shape)> {
        use crate::backend::BackendStorage;
        use crate::rocm_backend::RocmStorageSlice as S;
        let Some(tag) = indexed_moe_tag(self.dtype()) else {
            crate::bail!(
                "The given quantized dtype {:?} is not supported for indexed_moe_forward!",
                self.dtype()
            );
        };
        let (_, n, k) = self_shape.dims3()?;
        let (batch, input_dim1, in_k) = input_l.shape().dims3()?;
        let (id_batch, topk) = ids_l.shape().dims2()?;
        if reduce.is_some() && (glu.is_some() || extra.is_none() || input_dim1 != topk) {
            crate::bail!("indexed_moe_forward: the reducing launch needs per-slot inputs and an extra table");
        }
        if in_k != k || id_batch != batch || (input_dim1 != 1 && input_dim1 != topk) {
            crate::bail!(
                "indexed_moe_forward shape mismatch: w {self_shape:?}, x {:?}, ids {:?}",
                input_l.shape(),
                ids_l.shape()
            );
        }
        let (Some((x1, x2)), Some((i1, i2))) =
            (input_l.contiguous_offsets(), ids_l.contiguous_offsets())
        else {
            crate::bail!("indexed_moe_forward needs contiguous inputs and ids");
        };
        let dev = &self.device;
        let (dst_tag, x_ptr) = match &input.slice {
            S::BF16(s) => ("bf16", s.slice(x1..x2).device_ptr()),
            S::F16(s) => ("f16", s.slice(x1..x2).device_ptr()),
            S::F32(s) => ("f32", s.slice(x1..x2).device_ptr()),
            _ => crate::bail!("indexed_moe_forward: unsupported input dtype {:?}", input.dtype()),
        };
        let ids_ptr = ids.as_rocm_slice::<u32>()?.slice(i1..i2).device_ptr();

        // Quantize the activation rows to q8_1, zero padded to MATRIX_ROW_PADDING. One input
        // row per token may already be quantized as the stream's shared activation (seeded by
        // its producer, or by a matmul over the same rows just before).
        let total_rows = batch * input_dim1;
        let k_padded = pad(k, MATRIX_ROW_PADDING);
        let row_bytes = k_padded / GgmlDType::Q8_1.block_size() * GgmlDType::Q8_1.type_size();
        let shared = Some(SharedQ8Activation::new(dev, x_ptr, dst_tag, k, total_rows)).transpose()?;
        let mut own_q8_1: Option<HipSlice<u8>> = None;
        let x_q8_1: u64 = match &shared {
            Some(shared) => shared.ptr,
            None => {
                let y = unsafe { dev.alloc::<u8>(total_rows * row_bytes)? };
                let func = dev.get_or_load_func(
                    &format!("mmvq_gguf_quantize_q8_1_{dst_tag}"),
                    &kernels::MMVQ,
                )?;
                let cfg = LaunchConfig {
                    grid_dim: (
                        ceil_div(k_padded, CUDA_QUANTIZE_BLOCK_SIZE) as u32,
                        total_rows as u32,
                        1,
                    ),
                    block_dim: (CUDA_QUANTIZE_BLOCK_SIZE as u32, 1, 1),
                    shared_mem_bytes: 0,
                };
                let mut builder = func.builder();
                builder.arg(&x_ptr);
                builder.arg(&y);
                barg!(builder, k as i32, k_padded as i32);
                unsafe { builder.launch(cfg) }.w()?;
                let ptr = y.device_ptr();
                own_q8_1 = Some(y);
                ptr
            }
        };

        // Every output element is written, so the buffer needs no zeroing.
        let out_slots = if reduce.is_some() { 1 } else { topk };
        let out_len = batch * out_slots * n;
        let out = match input.dtype() {
            crate::DType::BF16 => S::BF16(unsafe { dev.alloc::<half::bf16>(out_len)? }),
            crate::DType::F16 => S::F16(unsafe { dev.alloc::<f16>(out_len)? }),
            _ => S::F32(unsafe { dev.alloc::<f32>(out_len)? }),
        };
        let out_ptr = match &out {
            S::BF16(s) => s.device_ptr(),
            S::F16(s) => s.device_ptr(),
            S::F32(s) => s.device_ptr(),
            _ => unreachable!(),
        };
        // The GLU kernel with an extra table also writes its output's q8_1 form (32 rows per
        // block), left as the shared activation for the down projection that follows.
        let glu_q8 = glu.is_some() && extra.is_some() && reduce.is_none() && n % 32 == 0;
        // The down projection reads rows of pad(n, MATRIX_ROW_PADDING) / 32 blocks.
        let q8_row_blocks = pad(n, MATRIX_ROW_PADDING) / GgmlDType::Q8_1.block_size();
        let out_q8 = if glu_q8 {
            Some(unsafe {
                dev.alloc::<u8>(batch * topk * q8_row_blocks * GgmlDType::Q8_1.type_size())?
            })
        } else {
            None
        };
        let name = match (glu, extra, reduce) {
            (None, Some(_), Some(_)) => format!("indexed_moe_xr_{tag}_{dst_tag}"),
            (Some(_), Some(_), _) if glu_q8 => format!("indexed_moe_glu_xq8_{tag}_{dst_tag}"),
            (Some(_), Some(_), _) => format!("indexed_moe_glu_x_{tag}_{dst_tag}"),
            (Some(_), None, _) => format!("indexed_moe_glu_{tag}_{dst_tag}"),
            (None, Some(_), None) => format!("indexed_moe_x_{tag}_{dst_tag}"),
            (None, None, _) => format!("indexed_moe_{tag}_{dst_tag}"),
        };
        let func = dev.get_or_load_func(&name, &kernels::QUANTIZED)?;
        // One output row per warp.
        let rows_per_block: usize = if glu_q8 { 32 } else { 4 };
        let cfg = LaunchConfig {
            grid_dim: (n.div_ceil(rows_per_block) as u32, batch as u32, out_slots as u32),
            block_dim: (WARP_SIZE as u32, rows_per_block as u32, 1),
            shared_mem_bytes: 0,
        };
        let act = glu.map_or(0i32, |(_, act)| act);
        let mut builder = func.builder();
        builder.arg(&self.data.inner);
        if let Some((up, _)) = glu {
            builder.arg(&up.data.inner);
        }
        builder.arg(&x_q8_1);
        builder.arg(&ids_ptr);
        if let Some(slot_weights) = &reduce {
            builder.arg(slot_weights);
        }
        builder.arg(&out_ptr);
        let q8_row_blocks_i32 = q8_row_blocks as i32;
        if let Some(out_q8) = &out_q8 {
            builder.arg(out_q8);
            builder.arg(&q8_row_blocks_i32);
        }
        barg!(
            builder,
            n as i32,
            k as i32,
            k_padded as i32,
            input_dim1 as i32
        );
        if glu.is_some() {
            builder.arg(&act);
        }
        let n_experts = extra.map_or(0, |(_, _, n)| n as i32);
        if let Some((extra_w, extra_up, _)) = extra {
            builder.arg(&extra_w.data.inner);
            if let Some(extra_up) = extra_up {
                builder.arg(&extra_up.data.inner);
            }
            builder.arg(&n_experts);
        }
        unsafe { builder.launch(cfg) }.w()?;
        drop(own_q8_1);
        if let Some(shared) = shared {
            shared.keep();
        }
        if let Some(out_q8) = out_q8 {
            // The key SharedQ8Activation::new computes for the down projection's rows.
            let code = match dst_tag {
                "bf16" => 0,
                "f16" => 1,
                _ => 2,
            };
            dev.hip_stream()
                .set_derived((out_ptr, n, batch * topk * 4 + code), out_q8);
        }
        Ok((
            RocmStorage {
                slice: out,
                device: dev.clone(),
            },
            crate::Shape::from((batch, out_slots, n)),
        ))
    }

    pub fn zeros(device: &RocmDevice, el_count: usize, dtype: GgmlDType) -> Result<Self> {
        let size_in_bytes = ceil_div(el_count, dtype.block_size()) * dtype.type_size();
        let padded_size_in_bytes =
            ceil_div(el_count + MATRIX_ROW_PADDING, dtype.block_size()) * dtype.type_size();
        let inner = device.alloc_zeros::<u8>(padded_size_in_bytes)?;
        Ok(QRocmStorage {
            data: PaddedHipSlice {
                inner,
                len: size_in_bytes,
            },
            device: device.clone(),
            dtype,
        })
    }

    pub fn dtype(&self) -> GgmlDType {
        self.dtype
    }

    pub fn device(&self) -> &RocmDevice {
        &self.device
    }

    pub fn dequantize(&self, elem_count: usize) -> Result<RocmStorage> {
        fn deq<T: GgmlType>(buffer: &[u8], n: usize, dst: &mut [f32]) {
            let slice = unsafe { std::slice::from_raw_parts(buffer.as_ptr() as *const T, n) };
            let vec = slice.to_vec();
            T::to_float(&vec, dst)
        }

        let fast_kernel = matches!(
            self.dtype,
            GgmlDType::Q4_0
                | GgmlDType::Q4_1
                | GgmlDType::Q5_0
                | GgmlDType::Q5_1
                | GgmlDType::Q8_0
                | GgmlDType::Q2K
                | GgmlDType::Q3K
                | GgmlDType::Q4K
                | GgmlDType::Q5K
                | GgmlDType::Q6K
                | GgmlDType::Q8K
        );
        if fast_kernel {
            return dequantize_f32(&self.data, self.dtype, elem_count, self.device());
        }
        // Run the dequantization on cpu.

        let buffer = self
            .device
            .clone_dtoh(&self.data.inner.slice(..self.data.len))?;
        let mut out = vec![0.0; elem_count];
        let block_len = elem_count / self.dtype.block_size();
        match self.dtype {
            GgmlDType::F32 => deq::<f32>(&buffer, block_len, &mut out),
            GgmlDType::F16 => deq::<half::f16>(&buffer, block_len, &mut out),
            GgmlDType::BF16 => deq::<half::bf16>(&buffer, block_len, &mut out),
            GgmlDType::Q4_0 => deq::<crate::quantized::BlockQ4_0>(&buffer, block_len, &mut out),
            GgmlDType::Q4_1 => deq::<crate::quantized::BlockQ4_1>(&buffer, block_len, &mut out),
            GgmlDType::Q5_0 => deq::<crate::quantized::BlockQ5_0>(&buffer, block_len, &mut out),
            GgmlDType::Q5_1 => deq::<crate::quantized::BlockQ5_1>(&buffer, block_len, &mut out),
            GgmlDType::Q8_0 => deq::<crate::quantized::BlockQ8_0>(&buffer, block_len, &mut out),
            GgmlDType::Q8_1 => deq::<crate::quantized::BlockQ8_1>(&buffer, block_len, &mut out),
            GgmlDType::Q2K => deq::<crate::quantized::BlockQ2K>(&buffer, block_len, &mut out),
            GgmlDType::Q3K => deq::<crate::quantized::BlockQ3K>(&buffer, block_len, &mut out),
            GgmlDType::Q4K => deq::<crate::quantized::BlockQ4K>(&buffer, block_len, &mut out),
            GgmlDType::Q5K => deq::<crate::quantized::BlockQ5K>(&buffer, block_len, &mut out),
            GgmlDType::Q6K => deq::<crate::quantized::BlockQ6K>(&buffer, block_len, &mut out),
            GgmlDType::Q8K => deq::<crate::quantized::BlockQ8K>(&buffer, block_len, &mut out),
        }

        self.device
            .storage_from_cpu_storage(&crate::CpuStorage::F32(out))
    }

    pub fn dequantize_f16(&self, elem_count: usize) -> Result<RocmStorage> {
        dequantize_f16(&self.data, self.dtype, elem_count, self.device())
    }

    pub fn dequantize_bf16(&self, elem_count: usize) -> Result<RocmStorage> {
        dequantize_bf16(&self.data, self.dtype, elem_count, self.device())
    }

    pub fn quantize(&mut self, src: &RocmStorage) -> Result<()> {
        // Run the quantization on cpu.
        let src = match &src.slice {
            crate::rocm_backend::RocmStorageSlice::F32(data) => self.device.clone_dtoh(data)?,
            _ => crate::bail!("only f32 can be quantized"),
        };
        let src_len = src.len();
        let src = crate::Storage::Cpu(crate::CpuStorage::F32(src));
        let mut qcpu_storage = crate::Device::Cpu.qzeros(src_len, self.dtype)?;
        qcpu_storage.quantize(&src)?;
        let data = qcpu_storage.data()?;
        let padded_len =
            data.len() + MATRIX_ROW_PADDING * self.dtype.type_size() / self.dtype.block_size();
        let mut inner = unsafe { self.device.alloc::<u8>(padded_len)? };
        self.device
            .memcpy_htod(&data, &mut inner.slice_mut(..data.len()))?;
        self.data = PaddedHipSlice {
            inner,
            len: data.len(),
        };
        Ok(())
    }

    pub fn quantize_imatrix(
        &mut self,
        src: &RocmStorage,
        imatrix_weights: &[f32],
        n_per_row: usize,
    ) -> Result<()> {
        // Run the quantization on cpu.
        let src = match &src.slice {
            crate::rocm_backend::RocmStorageSlice::F32(data) => self.device.clone_dtoh(data)?,
            _ => crate::bail!("only f32 can be quantized"),
        };
        let src_len = src.len();
        let src = crate::Storage::Cpu(crate::CpuStorage::F32(src));
        let mut qcpu_storage = crate::Device::Cpu.qzeros(src_len, self.dtype)?;
        qcpu_storage.quantize_imatrix(&src, imatrix_weights, n_per_row)?;
        let data = qcpu_storage.data()?;
        let padded_len =
            data.len() + MATRIX_ROW_PADDING * self.dtype.type_size() / self.dtype.block_size();
        let mut inner = unsafe { self.device.alloc::<u8>(padded_len)? };
        self.device
            .memcpy_htod(&data, &mut inner.slice_mut(..data.len()))?;
        self.data = PaddedHipSlice {
            inner,
            len: data.len(),
        };
        Ok(())
    }

    pub fn quantize_imatrix_onto(
        &mut self,
        src: &crate::CpuStorage,
        imatrix_weights: &[f32],
        n_per_row: usize,
    ) -> Result<()> {
        // Run the quantization on cpu.
        let src_len = src.as_slice::<f32>()?.len();
        let mut qcpu_storage = crate::Device::Cpu.qzeros(src_len, self.dtype)?;

        if let QStorage::Cpu(storage) = &mut qcpu_storage {
            storage.from_float_imatrix(src.as_slice::<f32>()?, imatrix_weights, n_per_row);
        } else {
            unreachable!()
        }

        let data = qcpu_storage.data()?;
        let padded_len =
            data.len() + MATRIX_ROW_PADDING * self.dtype.type_size() / self.dtype.block_size();
        let mut inner = unsafe { self.device.alloc::<u8>(padded_len)? };
        self.device
            .memcpy_htod(&data, &mut inner.slice_mut(..data.len()))?;
        self.data = PaddedHipSlice {
            inner,
            len: data.len(),
        };
        Ok(())
    }

    pub fn quantize_onto(&mut self, src: &crate::CpuStorage) -> Result<()> {
        // Run the quantization on cpu.
        let src_len = src.as_slice::<f32>()?.len();
        let mut qcpu_storage = crate::Device::Cpu.qzeros(src_len, self.dtype)?;

        if let QStorage::Cpu(storage) = &mut qcpu_storage {
            storage.from_float(src.as_slice::<f32>()?);
        } else {
            unreachable!()
        }

        let data = qcpu_storage.data()?;
        let padded_len =
            data.len() + MATRIX_ROW_PADDING * self.dtype.type_size() / self.dtype.block_size();
        let mut inner = unsafe { self.device.alloc::<u8>(padded_len)? };
        self.device
            .memcpy_htod(&data, &mut inner.slice_mut(..data.len()))?;
        self.data = PaddedHipSlice {
            inner,
            len: data.len(),
        };
        Ok(())
    }

    pub fn storage_size_in_bytes(&self) -> usize {
        self.data.len
    }

    pub fn embedding(
        &self,
        rows: usize,
        hidden: usize,
        ids: &RocmStorage,
        ids_l: &crate::Layout,
    ) -> Result<RocmStorage> {
        if !ids_l.is_contiguous() {
            crate::bail!("quantized embedding requires contiguous ids")
        }
        if !hidden.is_multiple_of(self.dtype.block_size()) {
            crate::bail!(
                "quantized embedding hidden size {hidden} is not divisible by block size {}",
                self.dtype.block_size()
            )
        }
        let expected_size = rows * hidden * self.dtype.type_size() / self.dtype.block_size();
        if self.storage_size_in_bytes() != expected_size {
            crate::bail!(
                "quantized tensor has {} bytes, expected {expected_size}",
                self.storage_size_in_bytes()
            )
        }
        let ids = ids.as_rocm_slice::<u32>()?;
        let ids = match ids_l.contiguous_offsets() {
            Some((o1, o2)) => ids.slice(o1..o2),
            None => Err(crate::Error::RequiresContiguous {
                op: "quantized-embedding",
            }
            .bt())?,
        };
        get_rows(&self.data, self.dtype, hidden, &ids, self.device())
    }

    pub fn fwd(
        &self,
        self_shape: &crate::Shape,
        storage: &RocmStorage,
        layout: &crate::Layout,
    ) -> Result<(RocmStorage, crate::Shape)> {
        let in_dtype = crate::backend::BackendStorage::dtype(storage);
        if !FORCE_DMMV.load(std::sync::atomic::Ordering::Relaxed) {
            if let Some(result) = self.fast_mmvq(self_shape, storage, layout)? {
                return Ok(result);
            }
        }
        // Beyond the batch sizes handled by the vector kernels, dequantizing the
        // weights and running a rocBLAS GEMM is several times faster than MMQ for
        // half precision activations. rocBLAS f32 GEMMs are slower than MMQ on
        // RDNA, so f32 activations keep the MMQ path.
        let rows = match layout.shape().dims() {
            [b, m, _] => b * m,
            [m, _] => *m,
            _ => 0,
        };
        if rows > MMVQ_MAX_ROWS
            && matches!(in_dtype, crate::DType::BF16 | crate::DType::F16)
            && layout.is_contiguous()
            && dequantize_kernel_available(self.dtype)
            && !FORCE_DMMV.load(std::sync::atomic::Ordering::Relaxed)
        {
            log_dequantize_gemm_shape(self.dtype, self_shape, rows);
            return self.dequantize_gemm(self_shape, storage, layout, in_dtype);
        }

        // The kernels below take f32 activations: run other dtypes through f32.
        if in_dtype != crate::DType::F32 {
            use crate::backend::BackendStorage;
            let storage = storage.to_dtype(layout, crate::DType::F32)?;
            let layout = crate::Layout::contiguous(layout.shape());
            let (out, shape) = self.fwd(self_shape, &storage, &layout)?;
            let out = out.to_dtype(&crate::Layout::contiguous(&shape), in_dtype)?;
            return Ok((out, shape));
        }

        // Fallback
        let max_bm = if FORCE_DMMV.load(std::sync::atomic::Ordering::Relaxed) {
            1
        } else {
            8
        };
        let use_vec_kernel = match layout.shape().dims() {
            [b, m, _k] => b * m <= max_bm,
            [b, _k] => *b <= max_bm,
            _ => false,
        };
        if use_vec_kernel {
            self.dequantize_matmul_vec(self_shape, storage, layout)
        } else {
            self.dequantize_matmul(self_shape, storage, layout)
        }
    }

    pub fn data(&self) -> Result<Vec<u8>> {
        let mut out = vec![0u8; self.data.len];
        self.device
            .memcpy_dtoh(&self.data.inner.slice(..self.data.len), &mut out)?;
        Ok(out)
    }

    pub fn device_ptr(&self) -> Result<*const u8> {
        Ok(self.data.inner.device_ptr() as *const u8)
    }
}

/// Batch sizes up to this use the q8_1 matrix-vector kernels.
const MMVQ_MAX_ROWS: usize = 8;
/// Larger batches are zero-padded to one of these instantiated widths and still use them;
/// beyond the last, the weights are dequantized for a dense GEMM.
const MMVQ_PADDED_ROWS: [usize; 4] = [12, 16, 24, 32];
const MMVQ_PADDED_MAX_ROWS: usize = 32;
/// Waves per row in the split-K kernels (`MMVQ_SPLITK_WAVES` in mmvq_rocm.cu).
const MMVQ_SPLITK_WAVES: u32 = 4;

/// `CANDLE_ROCM_MMVQ_SPLITK=0` sends padded batches on few-row weights back to dequantize + GEMM.
fn mmvq_splitk_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("CANDLE_ROCM_MMVQ_SPLITK").map_or(true, |v| v != "0"))
}

/// With `CANDLE_ROCM_LOG_DEQUANT_GEMM` set, print each (dtype, weight shape, rows) that falls
/// back to dequantize + dense GEMM, once: the shapes a missing vector kernel would cover.
fn log_dequantize_gemm_shape(dtype: GgmlDType, shape: &crate::Shape, rows: usize) {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*ON.get_or_init(|| std::env::var_os("CANDLE_ROCM_LOG_DEQUANT_GEMM").is_some()) {
        return;
    }
    static SEEN: std::sync::Mutex<Vec<(GgmlDType, Vec<usize>, usize)>> =
        std::sync::Mutex::new(Vec::new());
    let key = (dtype, shape.dims().to_vec(), rows);
    let mut seen = SEEN.lock().unwrap();
    if !seen.contains(&key) {
        eprintln!("CANDLE_ROCM_DEQUANT_GEMM {dtype:?} weight {:?} rows {rows}", shape.dims());
        seen.push(key);
    }
}

fn dequantize_kernel_available(dtype: GgmlDType) -> bool {
    matches!(
        dtype,
        GgmlDType::Q4_0
            | GgmlDType::Q4_1
            | GgmlDType::Q5_0
            | GgmlDType::Q5_1
            | GgmlDType::Q8_0
            | GgmlDType::Q2K
            | GgmlDType::Q3K
            | GgmlDType::Q4K
            | GgmlDType::Q5K
            | GgmlDType::Q6K
            | GgmlDType::Q8K
    )
}

/// The q8_1 quantization of a decode activation, padded to MATRIX_ROW_PADDING. Projections of
/// the same activations back to back (q/k/v, gate/up) reuse the previous quantization through
/// the stream's derived-buffer slot; [`Self::keep`] records this one for the next matmul.
struct SharedQ8Activation {
    ptr: u64,
    fresh: Option<HipSlice<u8>>,
    key: (u64, usize, usize),
    stream: std::sync::Arc<crate::rocm_backend::hip::HipStream>,
}

impl SharedQ8Activation {
    fn new(dev: &RocmDevice, rhs_ptr: u64, dst_tag: &'static str, k: usize, b_size: usize) -> Result<Self> {
        let k_padded = pad(k, MATRIX_ROW_PADDING);
        let stream = dev.hip_stream();
        let dtype_code = match dst_tag {
            "bf16" => 0,
            "f16" => 1,
            _ => 2,
        };
        let key = (rhs_ptr, k, b_size * 4 + dtype_code);
        let (ptr, fresh) = match stream.derived(key) {
            Some(ptr) => (ptr, None),
            None => {
                let row_bytes =
                    k_padded / GgmlDType::Q8_1.block_size() * GgmlDType::Q8_1.type_size();
                let y = unsafe { dev.alloc::<u8>(b_size * row_bytes)? };
                let func = dev.get_or_load_func_keyed(
                    ("quantize_q8_1", dst_tag, ""),
                    0,
                    &kernels::MMVQ,
                    || format!("mmvq_gguf_quantize_q8_1_{dst_tag}"),
                )?;
                let cfg = LaunchConfig {
                    grid_dim: (
                        ceil_div(k_padded, CUDA_QUANTIZE_BLOCK_SIZE) as u32,
                        b_size as u32,
                        1,
                    ),
                    block_dim: (CUDA_QUANTIZE_BLOCK_SIZE as u32, 1, 1),
                    shared_mem_bytes: 0,
                };
                let mut builder = func.builder();
                builder.arg(&rhs_ptr);
                builder.arg(&y);
                barg!(builder, k as i32, k_padded as i32);
                unsafe { builder.launch(cfg) }.w()?;
                (y.device_ptr(), Some(y))
            }
        };
        Ok(Self {
            ptr,
            fresh,
            key,
            stream,
        })
    }

    /// Only the matmul's own output was written since the quantization: keep it for the next.
    fn keep(self) {
        match self.fresh {
            Some(y) => {
                self.stream.set_derived(self.key, y);
            }
            None => self.stream.renew_derived(),
        }
    }
}

/// Declares that nothing launched or allocated since the shared activation quantization
/// `seed` (from [`qgemv_gate_mean_seeded`]) was recorded has touched that activation, so the
/// next quantized matmul over it may still reuse it. A later quantization that replaced the
/// seed is left alone.
pub fn renew_shared_activation(device: &crate::Device, seed: u64) -> bool {
    match device {
        crate::Device::Rocm(device) => device.hip_stream().renew_derived_id(seed),
        _ => false,
    }
}

/// Mirrors `MmvqMultiSegments` in mmvq_rocm.cu.
#[repr(C)]
#[derive(Clone, Copy)]
struct MmvqMultiSegments {
    vx: [u64; 4],
    dst: [u64; 4],
    nrows: [i32; 4],
    block_start: [i32; 5],
    tag: [i32; 4],
    x2: [i32; 4],
    count: i32,
}
unsafe impl crate::rocm_backend::hip::DeviceRepr for MmvqMultiSegments {}

fn mmvq_multi_tag(dtype: GgmlDType) -> Option<i32> {
    Some(match dtype {
        GgmlDType::Q4_0 => 0,
        GgmlDType::Q5_0 => 1,
        GgmlDType::Q8_0 => 2,
        GgmlDType::Q4K => 3,
        GgmlDType::Q5K => 4,
        GgmlDType::Q6K => 5,
        _ => return None,
    })
}

/// `x @ w^T` for up to four GGUF weights `[n_i, k]` over the same decode activations
/// `x: [.., k]` (at most 8 rows) in one launch: one quantization of `x`, one grid with a block
/// range per weight. Returns one `[.., n_i]` tensor per weight, or `None` when a weight's
/// format, the shapes or the device don't fit (the caller then runs them one by one).
pub fn fast_mmvq_multi(weights: &[&super::QTensor], x: &crate::Tensor) -> Result<Option<Vec<crate::Tensor>>> {
    use crate::rocm_backend::RocmStorageSlice as S;
    if weights.is_empty() || weights.len() > 4 {
        return Ok(None);
    }
    let crate::Device::Rocm(dev) = x.device() else {
        return Ok(None);
    };
    let k = x.dim(crate::D::Minus1)?;
    let b_size = x.elem_count() / k;
    if b_size == 0 || b_size > MMVQ_MAX_ROWS || k % 256 != 0 {
        return Ok(None);
    }
    let mut storages = Vec::with_capacity(weights.len());
    for w in weights {
        let super::QStorage::Rocm(s) = &w.storage else {
            return Ok(None);
        };
        let (n, wk) = w.shape().dims2()?;
        if wk != k || n == 0 || mmvq_multi_tag(s.dtype).is_none() || !s.device().same_device(dev) {
            return Ok(None);
        }
        storages.push((s, n));
    }
    let (x_storage, x_layout) = x.storage_and_layout();
    let crate::Storage::Rocm(xs) = &*x_storage else {
        return Ok(None);
    };
    let Some((o1, o2)) = x_layout.contiguous_offsets() else {
        return Ok(None);
    };
    let (dst_tag, x_ptr) = match &xs.slice {
        S::BF16(s) => ("bf16", s.slice(o1..o2).device_ptr()),
        S::F16(s) => ("f16", s.slice(o1..o2).device_ptr()),
        S::F32(s) => ("f32", s.slice(o1..o2).device_ptr()),
        _ => return Ok(None),
    };
    let shared = SharedQ8Activation::new(dev, x_ptr, dst_tag, k, b_size)?;
    let k_padded = pad(k, MATRIX_ROW_PADDING);
    const ROWS_PER_BLOCK: usize = 4;
    let mut segments = MmvqMultiSegments {
        vx: [0; 4],
        dst: [0; 4],
        nrows: [0; 4],
        block_start: [0; 5],
        tag: [0; 4],
        x2: [0; 4],
        count: weights.len() as i32,
    };
    let mut outs = Vec::with_capacity(weights.len());
    let mut blocks = 0usize;
    for (i, (s, n)) in storages.iter().enumerate() {
        let x2 = b_size > 1 && s.dtype == GgmlDType::Q4K && n % 2 == 0 && *n >= 2048;
        let out = match x.dtype() {
            crate::DType::BF16 => S::BF16(unsafe { dev.alloc::<half::bf16>(n * b_size)? }),
            crate::DType::F16 => S::F16(unsafe { dev.alloc::<f16>(n * b_size)? }),
            _ => S::F32(unsafe { dev.alloc::<f32>(n * b_size)? }),
        };
        segments.vx[i] = s.data.inner.device_ptr();
        segments.dst[i] = match &out {
            S::BF16(o) => o.device_ptr(),
            S::F16(o) => o.device_ptr(),
            S::F32(o) => o.device_ptr(),
            _ => unreachable!(),
        };
        segments.nrows[i] = *n as i32;
        segments.tag[i] = mmvq_multi_tag(s.dtype).expect("checked above");
        segments.x2[i] = x2 as i32;
        segments.block_start[i] = blocks as i32;
        blocks += n.div_ceil(ROWS_PER_BLOCK * if x2 { 2 } else { 1 });
        outs.push(out);
    }
    segments.block_start[weights.len()] = blocks as i32;
    let func = dev.get_or_load_func_keyed(
        ("mmvq_multi", dst_tag, ""),
        b_size,
        &kernels::MMVQ,
        || format!("mmvq_gguf_multi_{dst_tag}_seg{b_size}"),
    )?;
    let cfg = LaunchConfig {
        grid_dim: (blocks as u32, 1, 1),
        block_dim: (WARP_SIZE as u32, ROWS_PER_BLOCK as u32, 1),
        shared_mem_bytes: 0,
    };
    let stride_col_y = (k_padded / GgmlDType::Q8_1.block_size()) as i32;
    let y_q8_1 = shared.ptr;
    let mut builder = func.builder();
    builder.arg(&segments);
    builder.arg(&y_q8_1);
    barg!(builder, k as i32, stride_col_y);
    // SAFETY: ffi; every segment's shape, format and device were checked above.
    unsafe { builder.launch(cfg) }.w()?;
    shared.keep();
    let mut lead = x.dims().to_vec();
    lead.pop();
    Ok(Some(
        outs.into_iter()
            .zip(storages.iter())
            .map(|(slice, (_, n))| {
                let mut dims = lead.clone();
                dims.push(*n);
                crate::Tensor::from_storage(
                    crate::Storage::Rocm(RocmStorage {
                        slice,
                        device: dev.clone(),
                    }),
                    dims,
                    crate::op::BackpropOp::none(),
                    false,
                )
            })
            .collect(),
    ))
}

fn mmvq_tag(dtype: GgmlDType) -> Option<&'static str> {
    Some(match dtype {
        GgmlDType::Q4_0 => "q4_0",
        GgmlDType::Q4_1 => "q4_1",
        GgmlDType::Q5_0 => "q5_0",
        GgmlDType::Q5_1 => "q5_1",
        GgmlDType::Q8_0 => "q8_0",
        GgmlDType::Q2K => "q2_k",
        GgmlDType::Q3K => "q3_k",
        GgmlDType::Q4K => "q4_k",
        GgmlDType::Q5K => "q5_k",
        GgmlDType::Q6K => "q6_k",
        _ => return None,
    })
}

impl QRocmStorage {
    /// Matrix-vector product for batches of up to `MMVQ_MAX_ROWS` rows with the
    /// mmvq_gguf kernels, which read and write the activation dtype directly.
    /// Returns `None` when the inputs are not supported.
    fn fast_mmvq(
        &self,
        self_shape: &crate::Shape,
        rhs: &RocmStorage,
        rhs_l: &crate::Layout,
    ) -> Result<Option<(RocmStorage, crate::Shape)>> {
        use crate::backend::BackendStorage;
        use crate::rocm_backend::RocmStorageSlice as S;
        let Some(tag) = mmvq_tag(self.dtype) else {
            return Ok(None);
        };
        let (nrows, ncols) = self_shape.dims2()?;
        let (b_size, k) = match rhs_l.shape().dims() {
            [b, m, k] => (b * m, *k),
            [b, k] => (*b, *k),
            _ => return Ok(None),
        };
        if ncols != k || b_size == 0 || b_size > MMVQ_PADDED_MAX_ROWS {
            return Ok(None);
        }
        let Some((o1, o2)) = rhs_l.contiguous_offsets() else {
            return Ok(None);
        };
        let dev = self.device();
        // Past the one-launch row limit (speculative verify steps carry 1 + depth rows per
        // sequence), the rows are zero-padded to the next instantiated kernel width: the
        // weights are still read once, where dequantizing them for a dense GEMM read them
        // once more and ran the skinny GEMM at a tenth of the bandwidth.
        let b_launch = if b_size <= MMVQ_MAX_ROWS {
            b_size
        } else {
            *MMVQ_PADDED_ROWS
                .iter()
                .find(|&&rows| rows >= b_size)
                .expect("b_size <= MMVQ_PADDED_MAX_ROWS")
        };
        // Weights with fewer than k / 8 rows get a block of waves per row splitting K at the
        // padded widths; one wave per row would leave the GPU mostly idle. Measured against the
        // dequantized GEMM it wins at twelve columns for every such weight, and at sixteen and
        // more only for the tiny ones (gates of a few dozen rows); the 256- and 320-row weights
        // are faster through the GEMM there.
        let splitk = b_size > MMVQ_MAX_ROWS && k > 8 * nrows;
        if splitk && (!mmvq_splitk_enabled() || (b_launch > 12 && nrows > 64)) {
            return Ok(None);
        }
        let mut padded: Option<S> = None;
        let (dst_tag, rhs_ptr) = if b_launch == b_size {
            match &rhs.slice {
                S::BF16(s) => ("bf16", s.slice(o1..o2).device_ptr()),
                S::F16(s) => ("f16", s.slice(o1..o2).device_ptr()),
                S::F32(s) => ("f32", s.slice(o1..o2).device_ptr()),
                _ => return Ok(None),
            }
        } else {
            let (tag, slice, ptr) = match &rhs.slice {
                S::BF16(s) => {
                    let mut buf = dev.alloc_zeros::<half::bf16>(b_launch * k)?;
                    dev.memcpy_dtod(&s.slice(o1..o2), &mut buf.slice_mut(0..b_size * k))?;
                    let ptr = buf.device_ptr();
                    ("bf16", S::BF16(buf), ptr)
                }
                S::F16(s) => {
                    let mut buf = dev.alloc_zeros::<f16>(b_launch * k)?;
                    dev.memcpy_dtod(&s.slice(o1..o2), &mut buf.slice_mut(0..b_size * k))?;
                    let ptr = buf.device_ptr();
                    ("f16", S::F16(buf), ptr)
                }
                S::F32(s) => {
                    let mut buf = dev.alloc_zeros::<f32>(b_launch * k)?;
                    dev.memcpy_dtod(&s.slice(o1..o2), &mut buf.slice_mut(0..b_size * k))?;
                    let ptr = buf.device_ptr();
                    ("f32", S::F32(buf), ptr)
                }
                _ => return Ok(None),
            };
            padded = Some(slice);
            (tag, ptr)
        };
        let _keep_padded = &padded;
        let b_size = b_launch;

        let k_padded = pad(k, MATRIX_ROW_PADDING);
        let shared = SharedQ8Activation::new(dev, rhs_ptr, dst_tag, k, b_size)?;
        let y_q8_1 = shared.ptr;
        let keep_q8_1 = move || shared.keep();

        // Same launch shape as the CUDA launcher: two rows per block past batch 1.
        let weights = &self.data.inner;
        let out = match rhs.dtype() {
            crate::DType::BF16 => S::BF16(unsafe { dev.alloc::<half::bf16>(nrows * b_size)? }),
            crate::DType::F16 => S::F16(unsafe { dev.alloc::<f16>(nrows * b_size)? }),
            _ => S::F32(unsafe { dev.alloc::<f32>(nrows * b_size)? }),
        };
        let out_ptr = match &out {
            S::BF16(s) => s.device_ptr(),
            S::F16(s) => s.device_ptr(),
            S::F32(s) => s.device_ptr(),
            _ => unreachable!(),
        };
        if splitk {
            let cfg = LaunchConfig {
                grid_dim: (nrows as u32, 1, 1),
                block_dim: (WARP_SIZE as u32, MMVQ_SPLITK_WAVES, 1),
                shared_mem_bytes: 0,
            };
            let func = dev.get_or_load_func_keyed(
                (tag, dst_tag, "splitk"),
                b_size,
                &kernels::MMVQ,
                || format!("mmvq_gguf_{tag}_{dst_tag}_splitk{b_size}"),
            )?;
            let stride_col_y = (k_padded / GgmlDType::Q8_1.block_size()) as i32;
            let mut builder = func.builder();
            builder.arg(weights);
            builder.arg(&y_q8_1);
            builder.arg(&out_ptr);
            barg!(builder, k as i32, nrows as i32, stride_col_y);
            unsafe { builder.launch(cfg) }.w()?;
            keep_q8_1();
            let mut out_shape = rhs_l.shape().dims().to_vec();
            out_shape.pop();
            out_shape.push(nrows);
            return Ok(Some((
                RocmStorage {
                    slice: out,
                    device: dev.clone(),
                },
                out_shape.into(),
            )));
        }
        // A single token gives each row its own warp unless there are too few rows
        // to fill the GPU, where splitting long rows across warps wins.
        if b_size == 1 && k <= 8 * nrows {
            const ROWS_PER_BLOCK: usize = 4;
            let cfg = LaunchConfig {
                grid_dim: (nrows.div_ceil(ROWS_PER_BLOCK) as u32, 1, 1),
                block_dim: (WARP_SIZE as u32, ROWS_PER_BLOCK as u32, 1),
                shared_mem_bytes: 0,
            };
            let func =
                dev.get_or_load_func_keyed((tag, dst_tag, "warp_row"), 1, &kernels::MMVQ, || {
                    format!("mmvq_gguf_{tag}_{dst_tag}_warp_row")
                })?;
            let mut builder = func.builder();
            builder.arg(weights);
            builder.arg(&y_q8_1);
            builder.arg(&out_ptr);
            barg!(builder, k as i32, nrows as i32);
            unsafe { builder.launch(cfg) }.w()?;
            keep_q8_1();
            let mut out_shape = rhs_l.shape().dims().to_vec();
            out_shape.pop();
            out_shape.push(nrows);
            return Ok(Some((
                RocmStorage {
                    slice: out,
                    device: dev.clone(),
                },
                out_shape.into(),
            )));
        }
        // Decode batches also give each row its own warp, reading it once for every column.
        if b_size > 1 && k <= 8 * nrows {
            const ROWS_PER_BLOCK: usize = 4;
            // Q4_K waves take two rows once there are enough rows to fill the GPU anyway:
            // the rows share each activation load, which is what bounds multi-column dots.
            let x2 = tag == "q4_k" && nrows % 2 == 0 && nrows >= 2048;
            let rows_per_wave = if x2 { 2 } else { 1 };
            let cfg = LaunchConfig {
                grid_dim: (nrows.div_ceil(ROWS_PER_BLOCK * rows_per_wave) as u32, 1, 1),
                block_dim: (WARP_SIZE as u32, ROWS_PER_BLOCK as u32, 1),
                shared_mem_bytes: 0,
            };
            let func = dev.get_or_load_func_keyed(
                (tag, dst_tag, if x2 { "warp_row_x2" } else { "warp_row" }),
                b_size,
                &kernels::MMVQ,
                || format!("mmvq_gguf_{tag}_{dst_tag}_warp_row{b_size}{}", if x2 { "x2" } else { "" }),
            )?;
            let stride_col_y = (k_padded / GgmlDType::Q8_1.block_size()) as i32;
            let mut builder = func.builder();
            builder.arg(weights);
            builder.arg(&y_q8_1);
            builder.arg(&out_ptr);
            barg!(builder, k as i32, nrows as i32, stride_col_y);
            unsafe { builder.launch(cfg) }.w()?;
            keep_q8_1();
            let mut out_shape = rhs_l.shape().dims().to_vec();
            out_shape.pop();
            out_shape.push(nrows);
            return Ok(Some((
                RocmStorage {
                    slice: out,
                    device: dev.clone(),
                },
                out_shape.into(),
            )));
        }
        let rows_per_block = if b_size <= 1 { 1 } else { 2 };
        let nwarps = if b_size <= 4 { 4 } else { 2 };
        let cfg = LaunchConfig {
            grid_dim: (nrows.div_ceil(rows_per_block) as u32, 1, 1),
            block_dim: (WARP_SIZE as u32, nwarps, 1),
            shared_mem_bytes: 0,
        };
        let func = dev.get_or_load_func_keyed(
            (tag, dst_tag, "plain_cuda"),
            b_size,
            &kernels::MMVQ,
            || format!("mmvq_gguf_{tag}_{dst_tag}_plain_cuda{b_size}"),
        )?;
        let stride_col_y = (k_padded / GgmlDType::Q8_1.block_size()) as i32;
        let mut builder = func.builder();
        builder.arg(weights);
        builder.arg(&y_q8_1);
        builder.arg(&out_ptr);
        barg!(builder, k as i32, nrows as i32, stride_col_y, nrows as i32);
        unsafe { builder.launch(cfg) }.w()?;
        keep_q8_1();

        let mut out_shape = rhs_l.shape().dims().to_vec();
        out_shape.pop();
        out_shape.push(nrows);
        let out = RocmStorage {
            slice: out,
            device: dev.clone(),
        };
        Ok(Some((out, out_shape.into())))
    }

    /// Dequantizes the weights to the activation dtype and multiplies with a
    /// rocBLAS GEMM. The dequantized copy is freed on the stream once used.
    fn dequantize_gemm(
        &self,
        self_shape: &crate::Shape,
        storage: &RocmStorage,
        layout: &crate::Layout,
        in_dtype: crate::DType,
    ) -> Result<(RocmStorage, crate::Shape)> {
        use crate::backend::BackendStorage;
        let (n, k) = self_shape.dims2()?;
        let (b, m, k2) = match layout.shape().dims() {
            &[b, m, k2] => (b, m, k2),
            &[m, k2] => (1, m, k2),
            s => crate::bail!("unexpected shape for input {s:?}"),
        };
        if k2 != k {
            crate::bail!("mismatch on matmul dim {self_shape:?} {:?}", layout.shape())
        }
        let weights = match in_dtype {
            crate::DType::BF16 => self.dequantize_bf16(n * k)?,
            crate::DType::F16 => self.dequantize_f16(n * k)?,
            dtype => crate::bail!("unsupported activation dtype {dtype:?} for quantized matmul"),
        };
        let rhs_l = crate::Layout::new((k, n).into(), vec![1, k], 0).broadcast_as((b, k, n))?;
        let out = storage.matmul(&weights, (b, m, n, k), layout, &rhs_l)?;
        let mut out_shape = layout.shape().dims().to_vec();
        out_shape.pop();
        out_shape.push(n);
        Ok((out, out_shape.into()))
    }

    fn dequantize_matmul_vec(
        &self,
        self_shape: &crate::Shape,
        rhs: &RocmStorage,
        rhs_l: &crate::Layout,
    ) -> Result<(RocmStorage, crate::Shape)> {
        let (nrows, ncols) = self_shape.dims2()?;
        let rhs = rhs.as_rocm_slice::<f32>()?;
        let rhs = match rhs_l.contiguous_offsets() {
            Some((o1, o2)) => rhs.slice(o1..o2),
            None => Err(crate::Error::RequiresContiguous { op: "dmmv" }.bt())?,
        };
        let (b_size, k) = match rhs_l.shape().dims() {
            [b, m, k] => (b * m, *k),
            [b, k] => (*b, *k),
            _ => crate::bail!("unexpected rhs shape in dmmv {:?}", rhs_l.shape()),
        };
        if ncols != k {
            crate::bail!("mismatch on matmul dim {self_shape:?} {:?}", rhs_l.shape())
        }

        let out = if FORCE_DMMV.load(std::sync::atomic::Ordering::Relaxed) {
            dequantize_mul_mat_vec(&self.data, &rhs, self.dtype, ncols, nrows, self.device())?
        } else {
            mul_mat_vec_via_q8_1(
                &self.data,
                &rhs,
                self.dtype,
                ncols,
                nrows,
                b_size,
                self.device(),
            )?
        };
        let mut out_shape = rhs_l.shape().dims().to_vec();
        out_shape.pop();
        out_shape.push(nrows);
        Ok((out, out_shape.into()))
    }

    fn dequantize_matmul(
        &self,
        self_shape: &crate::Shape,
        storage: &RocmStorage,
        layout: &crate::Layout,
    ) -> Result<(RocmStorage, crate::Shape)> {
        use crate::backend::BackendStorage;
        let (n, k) = self_shape.dims2()?;
        let (b, m, k2) = match layout.shape().dims() {
            &[b, m, k2] => (b, m, k2),
            &[m, k2] => (1, m, k2),
            s => crate::bail!("unexpected shape for input {s:?}"),
        };
        if k2 != k {
            crate::bail!("mismatch on matmul dim {self_shape:?} {:?}", layout.shape())
        }

        let out = if FORCE_DMMV.load(std::sync::atomic::Ordering::Relaxed) {
            let data_f32 = self.dequantize(n * k)?;
            let rhs_l = crate::Layout::new((k, n).into(), vec![1, k], 0).broadcast_as((b, k, n))?;
            storage.matmul(&data_f32, (b, m, n, k), layout, &rhs_l)?
        } else {
            let storage = storage.as_rocm_slice::<f32>()?;
            let storage = match layout.contiguous_offsets() {
                Some((o1, o2)) => storage.slice(o1..o2),
                None => Err(crate::Error::RequiresContiguous {
                    op: "quantized-matmul",
                }
                .bt())?,
            };
            mul_mat_via_q8_1(
                &self.data,
                &storage,
                self.dtype,
                /* x_rows */ n,
                /* x_cols */ k,
                /* y_rows */ k,
                /* y_cols */ b * m,
                self.device(),
            )?
        };
        let mut out_shape = layout.shape().dims().to_vec();
        out_shape.pop();
        out_shape.push(n);
        Ok((out, out_shape.into()))
    }
}

pub fn load_quantized<T: super::GgmlType + Send + Sync + 'static>(
    device: &RocmDevice,
    data: &[T],
) -> Result<super::QStorage> {
    let data = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, core::mem::size_of_val(data))
    };
    let dtype = T::DTYPE;
    let padded_len = data.len() + MATRIX_ROW_PADDING * dtype.type_size() / dtype.block_size();
    let mut inner = device.alloc_zeros::<u8>(padded_len)?;
    device.memcpy_htod(data, &mut inner.slice_mut(..data.len()))?;
    Ok(QStorage::Rocm(QRocmStorage {
        data: PaddedHipSlice {
            inner,
            len: data.len(),
        },
        device: device.clone(),
        dtype,
    }))
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn rocm_quantize_q8_1() -> Result<()> {
        let dev = RocmDevice::new(0)?;
        let el = 256;
        let el_padded = pad(el, MATRIX_ROW_PADDING);
        let y_size_in_bytes =
            el_padded * GgmlDType::Q8_1.type_size() / GgmlDType::Q8_1.block_size();
        let mut y_q8_1 = unsafe { dev.alloc::<u8>(y_size_in_bytes)? };
        let vs: Vec<f32> = (0..el).map(|v| v as f32).collect();
        let y = dev.clone_htod(&vs)?;
        quantize_q8_1(&y.as_view(), &mut y_q8_1, el, 1, &dev)?;
        Ok(())
    }

    #[test]
    fn rocm_mmv_q8_1() -> Result<()> {
        let dev = RocmDevice::new(0)?;
        let ncols = 256;
        let vs: Vec<f32> = (0..ncols).map(|v| v as f32).collect();
        let y = dev.clone_htod(&vs)?;
        let mut xs = QRocmStorage::zeros(&dev, ncols, GgmlDType::Q4_0)?;
        xs.quantize(&RocmStorage::wrap_rocm_slice(y.clone(), dev.clone()))?;
        let rocm_storage = mul_mat_vec_via_q8_1(
            &xs.data,
            &y.as_view(),
            /* dtype */ GgmlDType::Q4_0,
            /* ncols */ ncols,
            /* nrows */ 1,
            /* b_size */ 1,
            &dev,
        )?;
        let vs = rocm_storage.as_rocm_slice::<f32>()?;
        let vs = dev.clone_dtoh(&vs.as_view())?;
        assert_eq!(vs.len(), 1);
        // for n = 255, n.(n+1).(2n+1) / 6 = 5559680
        // Q8 means 1/256 precision.
        assert_eq!(vs[0], 5561664.5);

        let rocm_storage = dequantize_mul_mat_vec(
            &xs.data,
            &y.as_view(),
            /* dtype */ GgmlDType::Q4_0,
            /* ncols */ ncols,
            /* nrows */ 1,
            &dev,
        )?;
        let vs = rocm_storage.as_rocm_slice::<f32>()?;
        let vs = dev.clone_dtoh(&vs.as_view())?;
        assert_eq!(vs.len(), 1);
        assert_eq!(vs[0], 5561851.0);
        Ok(())
    }

    #[test]
    fn rocm_mm_q8_1() -> Result<()> {
        let dev = RocmDevice::new(0)?;
        let ncols = 256;
        let vs: Vec<f32> = (0..ncols * 4).map(|v| v as f32 / 4.).collect();
        let y = dev.clone_htod(&vs)?;
        let mut xs = QRocmStorage::zeros(&dev, ncols * 4, GgmlDType::Q4_0)?;
        xs.quantize(&RocmStorage::wrap_rocm_slice(y.clone(), dev.clone()))?;
        let rocm_storage = mul_mat_via_q8_1(
            &xs.data,
            &y.as_view(),
            /* dtype */ GgmlDType::Q4_0,
            /* x_rows */ 4,
            /* x_cols */ ncols,
            /* y_rows */ ncols,
            /* y_cols */ 4,
            &dev,
        )?;
        let vs = rocm_storage.as_rocm_slice::<f32>()?;
        let vs = dev.clone_dtoh(&vs.as_view())?;

        /*
           x = torch.tensor([float(v) for v in range(1024)]).reshape(4, 256)
           x @ x.t() / 16
        tensor([[  347480.0000,   869720.0000,  1391960.0000,  1914200.0000],
                [  869720.0000,  2440536.0000,  4011352.0000,  5582166.5000],
                [ 1391960.0000,  4011352.0000,  6630742.0000,  9250132.0000],
                [ 1914200.0000,  5582166.5000,  9250132.0000, 12918099.0000]])
                */
        assert_eq!(vs.len(), 16);
        assert_eq!(vs[0], 347604.0);
        assert_eq!(vs[1], 888153.06);
        assert_eq!(vs[4], 869780.7);
        assert_eq!(vs[5], 2483145.0);
        assert_eq!(vs[11], 9407368.0);
        assert_eq!(vs[14], 9470856.0);
        assert_eq!(vs[15], 13138824.0);
        Ok(())
    }

    // The following test used to fail under compute-sanitizer until #2526.
    #[test]
    fn rocm_mm_q8_1_pad() -> Result<()> {
        let dev = RocmDevice::new(0)?;
        let (x_rows, ncols, y_cols) = (4, 16, 2048);
        let vs: Vec<f32> = (0..ncols * y_cols).map(|v| v as f32 / 256.).collect();
        let y = dev.clone_htod(&vs)?;
        let mut xs = QRocmStorage::zeros(&dev, ncols * x_rows, GgmlDType::Q4_0)?;
        xs.quantize(&RocmStorage::wrap_rocm_slice(y.clone(), dev.clone()))?;
        let rocm_storage = mul_mat_via_q8_1(
            &xs.data,
            &y.as_view(),
            /* dtype */ GgmlDType::Q4_0,
            /* x_rows */ x_rows,
            /* x_cols */ ncols,
            /* y_rows */ ncols,
            /* y_cols */ y_cols,
            &dev,
        )?;
        let vs = rocm_storage.as_rocm_slice::<f32>()?;
        let _vs = dev.clone_dtoh(&vs.as_view())?;
        Ok(())
    }
}

/// Format code of a weight `qgemv_fused.cu` dequantizes on the fly.
fn fused_weight_code(dtype: GgmlDType) -> Option<i32> {
    match dtype {
        GgmlDType::Q4_0 => Some(0),
        GgmlDType::Q5_0 => Some(1),
        GgmlDType::Q8_0 => Some(2),
        GgmlDType::Q4K => Some(3),
        GgmlDType::Q6K => Some(4),
        _ => None,
    }
}

const FUSED_MAX_ROWS: usize = 8;

/// `[rows, k]` of a contiguous bf16 activation matrix on `device`, and its device address, when
/// the fused kernels take it.
fn fused_activation(x: &crate::Tensor, device: &RocmDevice) -> Result<Option<(usize, usize, u64)>> {
    if x.dtype() != crate::DType::BF16 || !x.is_contiguous() {
        return Ok(None);
    }
    let k = x.dim(crate::D::Minus1)?;
    let rows = x.elem_count() / k.max(1);
    if rows == 0 || rows > FUSED_MAX_ROWS {
        return Ok(None);
    }
    let (storage, layout) = x.storage_and_layout();
    let crate::Storage::Rocm(storage) = &*storage else {
        return Ok(None);
    };
    if storage.device.id() != device.id() {
        return Ok(None);
    }
    let ptr = storage
        .as_rocm_slice::<half::bf16>()?
        .slice(layout.start_offset()..)
        .device_ptr();
    Ok((ptr % 16 == 0).then_some((rows, k, ptr)))
}

/// `(act(x . w1^T), x . w2^T)` for bf16 `x: [.., k]` with at most 8 rows, in one launch: two
/// GGUF matrices `[r1, k]` and `[r2, k]` over the same activations, where `act` is
/// `silu(v * scale)` when `silu_scale1` is set. `None` when the shapes or formats don't fit.
pub fn qgemv_pair(
    x: &crate::Tensor,
    w1: &super::QTensor,
    silu_scale1: Option<f32>,
    w2: &super::QTensor,
) -> Result<Option<(crate::Tensor, crate::Tensor)>> {
    let crate::Device::Rocm(device) = x.device() else {
        return Ok(None);
    };
    let (Some(t1), Some(t2)) = (fused_weight_code(w1.dtype()), fused_weight_code(w2.dtype())) else {
        return Ok(None);
    };
    let Some((rows, k, x_ptr)) = fused_activation(x, device)? else {
        return Ok(None);
    };
    let (Ok((r1, k1)), Ok((r2, k2))) = (w1.shape().dims2(), w2.shape().dims2()) else {
        return Ok(None);
    };
    if k1 != k
        || k2 != k
        || k % w1.dtype().block_size() != 0
        || k % w2.dtype().block_size() != 0
        || k % 8 != 0
    {
        return Ok(None);
    }
    let y1 = unsafe { device.alloc::<half::bf16>(rows * r1)? };
    let y2 = unsafe { device.alloc::<half::bf16>(rows * r2)? };
    let pair_rows = 1;
    // Shapes with a specialized kernel (Qwen3.8-Flash-Next's hyper-connection mixers) issue
    // every weight load up front; other shapes take the generic kernel.
    let name = match (k, w1.dtype(), w2.dtype()) {
        (10240, GgmlDType::Q4K, GgmlDType::Q4K) => "qgemv_pair_bf16_k10240_q4k_q4k",
        (10240, GgmlDType::Q6K, GgmlDType::Q4K) => "qgemv_pair_bf16_k10240_q6k_q4k",
        _ => "qgemv_pair_bf16_r1",
    };
    let func = device.get_or_load_func(name, &kernels::QGEMV_FUSED)?;
    let (w1_ptr, w2_ptr) = (w1.device_ptr()? as u64, w2.device_ptr()? as u64);
    let (m, k_i, r1_i, r2_i) = (rows as i32, k as i32, r1 as i32, r2 as i32);
    let act1 = silu_scale1.is_some() as i32;
    let scale1 = silu_scale1.unwrap_or(1.0);
    let mut builder = func.builder();
    builder.arg(&x_ptr);
    builder.arg(&m);
    builder.arg(&k_i);
    builder.arg(&w1_ptr);
    builder.arg(&t1);
    builder.arg(&r1_i);
    builder.arg(&act1);
    builder.arg(&scale1);
    builder.arg(&y1);
    builder.arg(&w2_ptr);
    builder.arg(&t2);
    builder.arg(&r2_i);
    builder.arg(&y2);
    let cfg = LaunchConfig {
        grid_dim: ((r1 + r2).div_ceil(pair_rows) as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    // SAFETY: ffi; shapes, formats and alignment were checked above.
    unsafe { builder.launch(cfg) }.w()?;
    let mut lead = x.dims().to_vec();
    lead.pop();
    let wrap = |slice, n: usize| {
        let mut dims = lead.clone();
        dims.push(n);
        crate::Tensor::from_storage(
            crate::Storage::Rocm(RocmStorage::wrap_rocm_slice(slice, device.clone())),
            dims,
            crate::op::BackpropOp::none(),
            false,
        )
    };
    Ok(Some((wrap(y1, r1), wrap(y2, r2))))
}

/// `mean_s xn[.., s * hidden + h] * sigmoid(lo . w[s * hidden + h])` over `groups` streams,
/// for up to 4 streams:
/// the gate projection of bf16 `lo: [.., k]` through GGUF `w: [groups * hidden, k]` and the gated
/// mean of `xn: [.., groups * hidden]`, in one launch. `None` when the shapes or formats don't fit.
pub fn qgemv_gate_mean(
    lo: &crate::Tensor,
    w: &super::QTensor,
    xn: &crate::Tensor,
    groups: usize,
) -> Result<Option<crate::Tensor>> {
    Ok(qgemv_gate_mean_impl(lo, w, xn, groups, false)?.map(|(out, _)| out))
}

/// [`qgemv_gate_mean`] whose kernel also quantizes its output to q8_1 and leaves that as the
/// stream's shared activation quantization: the next quantized matmul over the result (with
/// nothing launched or allocated in between) skips its own quantize launch. Shapes without a
/// specialized kernel take the plain path.
/// Also returns the seed's identity for [`renew_shared_activation`] when one was left.
pub fn qgemv_gate_mean_seeded(
    lo: &crate::Tensor,
    w: &super::QTensor,
    xn: &crate::Tensor,
    groups: usize,
) -> Result<Option<(crate::Tensor, Option<u64>)>> {
    qgemv_gate_mean_impl(lo, w, xn, groups, true)
}

fn qgemv_gate_mean_impl(
    lo: &crate::Tensor,
    w: &super::QTensor,
    xn: &crate::Tensor,
    groups: usize,
    seed_q8: bool,
) -> Result<Option<(crate::Tensor, Option<u64>)>> {
    let crate::Device::Rocm(device) = lo.device() else {
        return Ok(None);
    };
    let Some(t) = fused_weight_code(w.dtype()) else {
        return Ok(None);
    };
    let (Some((rows, k, lo_ptr)), Some((xn_rows, width, xn_ptr))) =
        (fused_activation(lo, device)?, fused_activation(xn, device)?)
    else {
        return Ok(None);
    };
    let Ok((w_rows, w_k)) = w.shape().dims2() else {
        return Ok(None);
    };
    let block = w.dtype().block_size();
    if w_k != k || k % block != 0 || k % 8 != 0 || xn_rows != rows || w_rows != width
        || width % groups != 0
        || groups > 4
        || 32 % groups != 0
    {
        return Ok(None);
    }
    let hidden = width / groups;
    let out = unsafe { device.alloc::<half::bf16>(rows * hidden)? };
    let specialized = match (k, groups, w.dtype()) {
        (320, 4, GgmlDType::Q5_0) => Some("qgemv_gate_mean_bf16_k320_g4_q5_0"),
        _ => None,
    };
    // The q8 variant needs the padded q8_1 row (MATRIX_ROW_PADDING) to equal the plain one.
    let seed_q8 = seed_q8 && specialized.is_some() && hidden % MATRIX_ROW_PADDING == 0;
    let q8 = if seed_q8 {
        let row_bytes = hidden / GgmlDType::Q8_1.block_size() * GgmlDType::Q8_1.type_size();
        Some(unsafe { device.alloc::<u8>(rows * row_bytes)? })
    } else {
        None
    };
    let name = match (specialized, seed_q8) {
        (Some(name), true) => format!("{name}_q8"),
        (Some(name), false) => name.to_string(),
        (None, _) => "qgemv_gate_mean_bf16".to_string(),
    };
    let func = device.get_or_load_func(&name, &kernels::QGEMV_FUSED)?;
    let w_ptr = w.device_ptr()? as u64;
    let (m, k_i, g, h) = (rows as i32, k as i32, groups as i32, hidden as i32);
    let mut builder = func.builder();
    builder.arg(&lo_ptr);
    builder.arg(&m);
    builder.arg(&k_i);
    builder.arg(&w_ptr);
    builder.arg(&t);
    builder.arg(&xn_ptr);
    builder.arg(&g);
    builder.arg(&h);
    builder.arg(&out);
    if let Some(q8) = &q8 {
        builder.arg(q8);
    }
    let warps = if seed_q8 { 32 } else { 8 };
    let cfg = LaunchConfig {
        grid_dim: (hidden.div_ceil(warps) as u32, 1, 1),
        block_dim: ((warps * 32) as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    // SAFETY: ffi; shapes, formats and alignment were checked above.
    unsafe { builder.launch(cfg) }.w()?;
    let seed = q8.map(|q8| {
        // The key SharedQ8Activation::new computes for bf16 activations of this shape.
        device
            .hip_stream()
            .set_derived((out.device_ptr(), hidden, rows * 4), q8)
    });
    let mut dims = xn.dims().to_vec();
    *dims.last_mut().expect("xn has a feature axis") = hidden;
    Ok(Some((
        crate::Tensor::from_storage(
            crate::Storage::Rocm(RocmStorage::wrap_rocm_slice(out, device.clone())),
            dims,
            crate::op::BackpropOp::none(),
            false,
        ),
        seed,
    )))
}


/// Rocm storage of a quantized tensor, or an error.
fn rocm_qstorage(t: &super::QTensor) -> Result<&QRocmStorage> {
    match &t.storage {
        super::QStorage::Rocm(s) => Ok(s),
        _ => crate::bail!("indexed MoE with an extra expert needs rocm quantized tensors"),
    }
}

fn rocm_tensor_storage<'a, G: std::ops::Deref<Target = crate::Storage>>(
    guard: &'a G,
    name: &str,
) -> Result<&'a RocmStorage> {
    match &**guard {
        crate::Storage::Rocm(s) => Ok(s),
        _ => crate::bail!("indexed MoE with an extra expert: {name} must be a rocm tensor"),
    }
}

/// `x @ experts[ids]^T` where ids at or past `experts.dim(0)` select rows of `extra`
/// (`[extra_count, n, k]` or `[n, k]`, same format as `experts`): the shared expert of an MoE
/// block runs in the same launch as the routed ones.
pub fn indexed_moe_forward_with_extra(
    experts: &super::QTensor,
    extra: &super::QTensor,
    x: &crate::Tensor,
    ids: &crate::Tensor,
) -> Result<crate::Tensor> {
    let (n_experts, out_shape) = extra_table_shapes(experts, extra, None)?;
    let (w, e) = (rocm_qstorage(experts)?, rocm_qstorage(extra)?);
    let (xs, is) = (x.storage(), ids.storage());
    let (storage, shape) = w.indexed_moe(
        None,
        Some((e, None, n_experts)),
        None,
        &out_shape,
        rocm_tensor_storage(&xs, "x")?,
        x.layout(),
        rocm_tensor_storage(&is, "ids")?,
        ids.layout(),
    )?;
    Ok(crate::Tensor::from_storage(
        crate::Storage::Rocm(storage),
        shape,
        crate::op::BackpropOp::none(),
        false,
    ))
}

/// [`indexed_moe_forward_with_extra`] summed over each token's slots in the kernel:
/// `sum_j slot_weights[t, j] * (x[t, j] @ experts[ids[t, j]]^T)`, returning `[batch, n]`.
/// `x: [batch, topk, k]`, `ids: [batch, topk]` u32, `slot_weights: [batch, topk]` f32.
pub fn indexed_moe_forward_with_extra_reduce(
    experts: &super::QTensor,
    extra: &super::QTensor,
    x: &crate::Tensor,
    ids: &crate::Tensor,
    slot_weights: &crate::Tensor,
) -> Result<crate::Tensor> {
    let (n_experts, out_shape) = extra_table_shapes(experts, extra, None)?;
    let (w, e) = (rocm_qstorage(experts)?, rocm_qstorage(extra)?);
    if slot_weights.dtype() != crate::DType::F32 || slot_weights.dims() != ids.dims() {
        crate::bail!(
            "indexed_moe_forward_with_extra_reduce: slot weights {:?} must be f32 and shaped like ids {:?}",
            slot_weights.shape(),
            ids.shape()
        );
    }
    let slot_weights = slot_weights.contiguous()?;
    let (xs, is, ws) = (x.storage(), ids.storage(), slot_weights.storage());
    let weights_ptr = rocm_tensor_storage(&ws, "slot_weights")?
        .as_rocm_slice::<f32>()?
        .slice(slot_weights.layout().start_offset()..)
        .device_ptr();
    let (storage, shape) = w.indexed_moe(
        None,
        Some((e, None, n_experts)),
        Some(weights_ptr),
        &out_shape,
        rocm_tensor_storage(&xs, "x")?,
        x.layout(),
        rocm_tensor_storage(&is, "ids")?,
        ids.layout(),
    )?;
    let (batch, _, n) = shape.dims3()?;
    Ok(crate::Tensor::from_storage(
        crate::Storage::Rocm(storage),
        (batch, n),
        crate::op::BackpropOp::none(),
        false,
    ))
}

/// GLU form of [`indexed_moe_forward_with_extra`]: `act(x @ gate[ids]^T) * (x @ up[ids]^T)` with
/// `extra_gate` / `extra_up` for ids at or past the expert count. `act`: 0 silu, 1 gelu.
pub fn indexed_moe_glu_forward_with_extra(
    gate: &super::QTensor,
    up: &super::QTensor,
    act: i32,
    extra_gate: &super::QTensor,
    extra_up: &super::QTensor,
    x: &crate::Tensor,
    ids: &crate::Tensor,
) -> Result<crate::Tensor> {
    let (n_experts, out_shape) = extra_table_shapes(gate, extra_gate, Some((up, extra_up)))?;
    let (g, u) = (rocm_qstorage(gate)?, rocm_qstorage(up)?);
    let (eg, eu) = (rocm_qstorage(extra_gate)?, rocm_qstorage(extra_up)?);
    let (xs, is) = (x.storage(), ids.storage());
    let (storage, shape) = g.indexed_moe(
        Some((u, act)),
        Some((eg, Some(eu), n_experts)),
        None,
        &out_shape,
        rocm_tensor_storage(&xs, "x")?,
        x.layout(),
        rocm_tensor_storage(&is, "ids")?,
        ids.layout(),
    )?;
    Ok(crate::Tensor::from_storage(
        crate::Storage::Rocm(storage),
        shape,
        crate::op::BackpropOp::none(),
        false,
    ))
}

/// Checks that `extra` (and `up`s) match `experts`' format and `[n, k]`, returning the routed
/// expert count and `experts`' shape.
fn extra_table_shapes(
    experts: &super::QTensor,
    extra: &super::QTensor,
    ups: Option<(&super::QTensor, &super::QTensor)>,
) -> Result<(usize, crate::Shape)> {
    let (n_experts, n, k) = experts.shape().dims3()?;
    let extra_nk = match extra.shape().dims() {
        [_, en, ek] | [en, ek] => (*en, *ek),
        dims => crate::bail!("extra expert table has shape {dims:?}"),
    };
    let mut ok = extra_nk == (n, k) && extra.dtype() == experts.dtype();
    if let Some((up, extra_up)) = ups {
        ok &= up.shape() == experts.shape()
            && up.dtype() == experts.dtype()
            && extra_up.shape() == extra.shape()
            && extra_up.dtype() == experts.dtype();
    }
    if !ok {
        crate::bail!(
            "extra expert {:?} {:?} does not match experts {:?} {:?}",
            extra.dtype(),
            extra.shape(),
            experts.dtype(),
            experts.shape()
        );
    }
    Ok((n_experts, experts.shape().clone()))
}
