#!/usr/bin/env bash
set -euo pipefail
runtime=${1:-.venv-serving}
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
"$runtime/bin/python" -m venv "$repo/.venv-eval"
local_site=$("$repo/.venv-eval/bin/python" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
"$runtime/bin/python" -c 'import sys; print("\n".join(p for p in sys.path if p.endswith("site-packages")))' > "$local_site/reused-runtime.pth"
printf '%s\n' "$repo" >> "$local_site/reused-runtime.pth"
# Only small evaluation dependencies. --no-deps prevents any torch/CUDA/vLLM install.
"$repo/.venv-eval/bin/python" -m pip install --no-deps \
  datasets==3.6.0 dill==0.3.8 multiprocess==0.70.16 fsspec==2025.3.0 \
  fire==0.7.1 absl-py==2.3.1 tree-sitter==0.24.0 tree-sitter-python==0.23.6 \
  wget==3.2 appdirs==1.4.4 tempdir==0.7.1 termcolor==3.1.0 \
  langdetect==1.0.9 immutabledict==4.2.2 emoji==2.14.1 syllapy==0.7.2 \
  math-verify==0.8.0 loguru==0.7.3 nltk==3.9.1 antlr4-python3-runtime==4.13.2
"$repo/.venv-eval/bin/python" -m stepquant.evaluation.upstream
