#!/usr/bin/env bash
# Reuse an existing model-capable environment; never install a second torch stack.
set -euo pipefail
runtime=${1:?Usage: bash scripts/reuse_calibration_env.sh /existing/environment qwen-or-kimi}
model=${2:?Specify qwen or kimi}
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
case "$model" in
  qwen) target="$repo/.venv-runtime"; expected=5.8.1 ;;
  kimi) target="$repo/.venv-kimi"; expected=4.57.1 ;;
  *) echo 'Expected qwen or kimi' >&2; exit 1 ;;
esac
"$runtime/bin/python" - "$expected" <<'PY'
import sys
from importlib.metadata import version
import torch, accelerate, safetensors
assert version('transformers') == sys.argv[1], 'Source environment must have the pinned Transformers version'
if sys.argv[1]=='4.57.1':
    import fla, einops, tiktoken
PY
if [[ -e "$target" ]]; then
  echo "$target already exists; reuse it directly or choose calibration_python in configs/models.json" >&2
  exit 1
fi
"$runtime/bin/python" -m venv "$target"
local_site=$("$target/bin/python" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
"$runtime/bin/python" -c 'import sys; print("\n".join(p for p in sys.path if p.endswith("site-packages")))' > "$local_site/reused-calibration-runtime.pth"
"$target/bin/python" -m pip install -e "$repo" --no-deps --no-build-isolation --no-index --disable-pip-version-check
printf 'Calibration Python: %s\n' "$target/bin/python"
