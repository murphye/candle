# Development image for building and testing candle's ROCm backend.
#
# ROCm and the Rust toolchain live in the image; sources and the cargo cache are
# mounted at runtime by run.sh, so nothing needs to be installed on the host.
ARG ROCM_IMAGE=rocm/dev-ubuntu-24.04:10.1.0-full
FROM ${ROCM_IMAGE}

ARG RUST_TOOLCHAIN=stable
ENV RUSTUP_HOME=/opt/rustup \
    PATH=/opt/cargo/bin:/opt/rocm/bin:/opt/rocm/llvm/bin:${PATH} \
    ROCM_PATH=/opt/rocm \
    LD_LIBRARY_PATH=/opt/rocm/lib

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential ca-certificates curl git pkg-config libssl-dev \
    && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
        | CARGO_HOME=/opt/cargo sh -s -- -y --no-modify-path --profile minimal \
            --default-toolchain ${RUST_TOOLCHAIN} --component rustfmt,clippy \
    && chmod -R a+rX /opt/rustup /opt/cargo

# Overridden by run.sh to point at a writable, persistent cache volume.
ENV CARGO_HOME=/cache/cargo
