# candle-rocm-kernels

ROCm/HIP kernels and runtime bindings for candle's `rocm` backend.

The kernels are the CUDA sources from `candle-kernels`, compiled unchanged with
`hipcc`. The headers in `hip_compat/` stand in for the CUDA toolkit headers they
include and map the CUDA-specific names onto HIP. Each module becomes a clang
offload bundle with one code object per GPU architecture, loaded at runtime with
`hipModuleLoadData`. `src/rocm/` holds ROCm-only kernels (random number
generation, which replaces curand).

`src/ffi.rs` declares the small subset of the HIP runtime and rocBLAS that the
backend uses, avoiding bindgen and version-dependent structs.

## Build configuration

- `ROCM_PATH`: ROCm installation, default `/opt/rocm`.
- `CANDLE_ROCM_ARCHS`: comma separated GPU targets, e.g. `gfx1201,gfx1100`. When
  unset, the GPUs on the build machine are detected with `amdgpu-arch`, falling
  back to a list of common RDNA and CDNA targets. Every target adds compile time.

## Development container

`docker/rocm/run.sh` at the repository root runs commands in a ROCm container
with the GPUs attached, so ROCm does not need to be installed on the host:

```bash
docker/rocm/run.sh cargo test -p candle-core --features rocm -- rocm
docker/rocm/run.sh cargo run -p candle-core --features rocm --example rocm_basics
```
