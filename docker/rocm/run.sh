#!/usr/bin/env bash
# Run a command inside the ROCm development container with the GPUs attached.
#
#   docker/rocm/run.sh                      # interactive shell
#   docker/rocm/run.sh cargo test -p candle-core --features rocm
#
# The parent directory of this repository is mounted at the same path, so
# sibling checkouts (e.g. ../mistral.rs) are visible inside the container.
set -euo pipefail

IMAGE="${CANDLE_ROCM_IMAGE:-candle-rocm-dev:latest}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WORKSPACE="$(dirname "${REPO_ROOT}")"
CACHE_DIR="${CANDLE_ROCM_CACHE:-${HOME}/.cache/candle-rocm}"
# Share the host's Hugging Face cache so downloaded models persist.
HF_CACHE="${CANDLE_ROCM_HF_CACHE:-${HOME}/.cache/huggingface}"

if ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
    docker build -t "${IMAGE}" "${SCRIPT_DIR}"
fi

mkdir -p "${CACHE_DIR}/cargo" "${HF_CACHE}"

tty_flags=()
if [ -t 0 ] && [ -t 1 ]; then
    tty_flags=(-it)
fi

if [ $# -eq 0 ]; then
    set -- bash
fi

# Forward selected host variables only when they are set.
env_flags=()
for var in CARGO_TARGET_DIR CANDLE_ROCM_ARCHS HIP_VISIBLE_DEVICES RUST_BACKTRACE RUST_LOG HF_TOKEN; do
    if [ -n "${!var:-}" ]; then
        env_flags+=(-e "${var}=${!var}")
    fi
done

exec docker run --rm "${tty_flags[@]}" \
    --device /dev/kfd --device /dev/dri \
    --group-add "$(getent group video | cut -d: -f3)" \
    --group-add "$(getent group render | cut -d: -f3)" \
    --security-opt seccomp=unconfined \
    --security-opt label=disable \
    --ipc host \
    --user "$(id -u):$(id -g)" \
    -e HOME=/cache \
    -e CARGO_HOME=/cache/cargo \
    -e HF_HOME=/cache/.cache/huggingface \
    "${env_flags[@]}" \
    -v "${CACHE_DIR}:/cache" \
    -v "${HF_CACHE}:/cache/.cache/huggingface" \
    -v "${WORKSPACE}:${WORKSPACE}" \
    -w "${PWD}" \
    "${IMAGE}" "$@"
