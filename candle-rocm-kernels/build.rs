//! Compiles candle's CUDA kernels for AMD GPUs with `hipcc`.
//!
//! The `.cu` sources are shared with `candle-kernels`; the headers in `hip_compat/`
//! stand in for the CUDA toolkit headers they include. Each source is compiled to a
//! clang offload bundle holding one code object per target architecture, which the
//! HIP runtime loads with `hipModuleLoadData`.
//!
//! Environment variables:
//! - `ROCM_PATH`: ROCm installation (default `/opt/rocm`).
//! - `CANDLE_ROCM_ARCHS`: comma separated gfx targets. When unset, the GPUs present
//!   on the build machine are used, falling back to a list of common targets.
use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Kernel modules: (constant name, source file relative to the kernels directory).
const MODULES: &[(&str, &str)] = &[
    ("AFFINE", "candle-kernels/affine.cu"),
    ("BINARY", "candle-kernels/binary.cu"),
    ("CAST", "candle-kernels/cast.cu"),
    ("CONV", "candle-kernels/conv.cu"),
    ("FILL", "candle-kernels/fill.cu"),
    ("GEMV", "rocm/gemv.cu"),
    ("INDEXING", "candle-kernels/indexing.cu"),
    ("MMVQ", "rocm/mmvq_rocm.cu"),
    ("QUANTIZED", "rocm/quantized_rocm.cu"),
    ("REDUCE", "candle-kernels/reduce.cu"),
    ("SORT", "candle-kernels/sort.cu"),
    ("TERNARY", "candle-kernels/ternary.cu"),
    ("UNARY", "candle-kernels/unary.cu"),
    ("RANDOM", "rocm/random.cu"),
];

const FALLBACK_ARCHS: &[&str] = &[
    "gfx1030", "gfx1100", "gfx1101", "gfx1102", "gfx1200", "gfx1201", "gfx90a", "gfx942",
];

fn main() {
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-changed=src/rocm");
    println!("cargo::rerun-if-changed=hip_compat");
    println!("cargo::rerun-if-env-changed=ROCM_PATH");
    println!("cargo::rerun-if-env-changed=CANDLE_ROCM_ARCHS");

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let rocm_path = PathBuf::from(env::var("ROCM_PATH").unwrap_or_else(|_| "/opt/rocm".into()));

    println!(
        "cargo::rustc-link-search=native={}",
        rocm_path.join("lib").display()
    );
    println!("cargo::rustc-link-lib=dylib=amdhip64");
    println!("cargo::rustc-link-lib=dylib=rocblas");

    // When packaged, the CUDA sources are not next to this crate. Fall back to the
    // sibling `candle-kernels` checkout used during workspace development.
    let cuda_src = manifest_dir.join("../candle-kernels/src");
    println!("cargo::rerun-if-changed={}", cuda_src.display());
    let rocm_src = manifest_dir.join("src/rocm");
    let compat = manifest_dir.join("hip_compat");

    let hipcc = rocm_path.join("bin/hipcc");
    let archs = target_archs(&rocm_path);
    println!(
        "cargo::warning=candle-rocm-kernels: building for {}",
        archs.join(",")
    );
    // Exported to dependents that compile their own kernels against the same
    // compatibility headers and targets: DEP_CANDLE_ROCM_KERNELS_{HIP_COMPAT,ARCHS}.
    println!("cargo::metadata=hip_compat={}", compat.display());
    println!("cargo::metadata=archs={}", archs.join(","));

    let jobs: Vec<(String, PathBuf, PathBuf)> = MODULES
        .iter()
        .map(|(name, src)| {
            let src = match src.split_once('/') {
                Some(("candle-kernels", file)) => cuda_src.join(file),
                Some(("rocm", file)) => rocm_src.join(file),
                _ => unreachable!(),
            };
            let out = out_dir.join(format!("{}.hsaco", name.to_lowercase()));
            (name.to_string(), src, out)
        })
        .collect();

    std::thread::scope(|s| {
        let handles: Vec<_> = jobs
            .iter()
            .map(|(name, src, out)| {
                let (hipcc, archs, compat, cuda_src) = (&hipcc, &archs, &compat, &cuda_src);
                s.spawn(move || compile(hipcc, archs, compat, cuda_src, src, out, name))
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
    });

    let mut code = String::new();
    for (name, _, out) in jobs.iter() {
        code.push_str(&format!(
            "pub const {name}: &[u8] = include_bytes!({:?});\n",
            out.display().to_string()
        ));
    }
    std::fs::write(out_dir.join("code_objects.rs"), code).unwrap();
}

fn compile(
    hipcc: &Path,
    archs: &[String],
    compat: &Path,
    cuda_src: &Path,
    src: &Path,
    out: &Path,
    name: &str,
) {
    let mut cmd = Command::new(hipcc);
    cmd.arg("--genco")
        .arg("-O3")
        .arg("-std=c++17")
        .arg("-munsafe-fp-atomics")
        .arg("-Wno-unused-result")
        .arg("-include")
        .arg(compat.join("candle_hip_prelude.h"))
        .arg("-I")
        .arg(compat)
        .arg("-I")
        .arg(cuda_src);
    for arch in archs {
        cmd.arg(format!("--offload-arch={arch}"));
    }
    cmd.arg("-o").arg(out).arg(src);
    let output = cmd
        .output()
        .unwrap_or_else(|e| panic!("failed to run {}: {e}", hipcc.display()));
    if !output.status.success() {
        panic!(
            "hipcc failed for {name} ({}):\n{}\n{}",
            src.display(),
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

fn target_archs(rocm_path: &Path) -> Vec<String> {
    if let Ok(archs) = env::var("CANDLE_ROCM_ARCHS") {
        let archs: Vec<String> = archs
            .split([',', ';', ' '])
            .filter(|s| !s.is_empty())
            .map(String::from)
            .collect();
        if !archs.is_empty() {
            return archs;
        }
    }
    // `amdgpu-arch` lists the architectures of the GPUs on this machine.
    let detected = [
        rocm_path.join("llvm/bin/amdgpu-arch"),
        rocm_path.join("bin/amdgpu-arch"),
    ]
    .iter()
    .find_map(|tool| Command::new(tool).output().ok())
    .filter(|o| o.status.success())
    .map(|o| {
        let mut archs: Vec<String> = String::from_utf8_lossy(&o.stdout)
            .lines()
            .map(|l| l.trim().to_string())
            .filter(|l| l.starts_with("gfx"))
            .collect();
        archs.sort();
        archs.dedup();
        archs
    })
    .unwrap_or_default();
    if !detected.is_empty() {
        return detected;
    }
    FALLBACK_ARCHS.iter().map(|s| s.to_string()).collect()
}
