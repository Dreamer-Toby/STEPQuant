#!/usr/bin/env bash
# Reuse installed packages without changing the source environment or downloading wheels.
set -euo pipefail
runtime=${1:?Usage: bash scripts/reuse_sglang_env.sh /path/to/existing/sglang/environment}
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
"$runtime/bin/python" - <<'PY'
from importlib.metadata import version
assert version('sglang') == '0.5.12', 'SGLang 0.5.12 is required'
PY
site_path=$("$runtime/bin/python" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
"$runtime/bin/python" -m venv "$repo/.venv-serving"
local_site=$("$repo/.venv-serving/bin/python" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
printf '%s\n' "$site_path" > "$local_site/reused-sglang-runtime.pth"
"$repo/.venv-serving/bin/python" -m pip install -e "$repo" --no-deps --no-build-isolation --no-index --disable-pip-version-check
if [[ -x "$runtime/bin/ninja" ]]; then
    ln -sfn "$(cd -- "$runtime" && pwd)/bin/ninja" "$repo/.venv-serving/bin/ninja"
fi
# The CUDA wheel uses lib/, while SGLang's JIT expects a toolkit with lib64/.
if [[ -x "$site_path/nvidia/cu13/bin/nvcc" ]]; then
    mkdir -p "$repo/artifacts/cuda13"
    for part in bin include nvvm; do
        ln -sfn "$site_path/nvidia/cu13/$part" "$repo/artifacts/cuda13/$part"
    done
    if [[ -L "$repo/artifacts/cuda13/lib64" ]]; then
        unlink "$repo/artifacts/cuda13/lib64"
    fi
    mkdir -p "$repo/artifacts/cuda13/lib64"
    for library in "$site_path/nvidia/cu13/lib/"*; do
        ln -sfn "$library" "$repo/artifacts/cuda13/lib64/$(basename -- "$library")"
    done
    ln -sfn libcudart.so.13 "$repo/artifacts/cuda13/lib64/libcudart.so"
    printf 'export CUDA_HOME=%q\n' "$repo/artifacts/cuda13"
fi
printf 'source %q\n' "$repo/.venv-serving/bin/activate"
