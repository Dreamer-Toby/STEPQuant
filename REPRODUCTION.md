# Running STEPQuant

[← Back to README](README.md)

Run the commands below from the repository root. Model checkpoints and datasets are not included. The examples use four GPUs; change `0,1,2,3` to match your machine.

## 1. Prepare environments

The tested serving stack is **SGLang 0.5.12**, **PyTorch 2.11.0**, and **Triton 3.7.1** with a working CUDA toolkit. Calibration needs Transformers **5.8.1** for Qwen or **4.57.1** with `fla-core==0.4.1` for Kimi.

```bash
bash scripts/reuse_sglang_env.sh /path/to/sglang-env
bash scripts/reuse_eval_env.sh .venv-serving
bash scripts/reuse_calibration_env.sh /path/to/qwen-env qwen
bash scripts/reuse_calibration_env.sh /path/to/kimi-env kimi
```

The scripts reuse existing environments. Run only the calibration command(s) for your model. If the serving script prints an `export CUDA_HOME=...` command, run it before serving.

## 2. Calibrate

Set the checkpoint paths and calibration Python executables in [`configs/models.json`](configs/models.json), then generate the STEPQuant plans:

```bash
.venv-eval/bin/python -m stepquant.reproduction \
  --models qwen kimi --formats stepquant4 stepquant6 --devices 0,1,2,3
```

Plans are saved in `artifacts/reproduction/plans/`. Use `--models qwen` or `--models kimi` to run one model.

## 3. Serve

Qwen example with STEPQuant@6:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 .venv-serving/bin/python -m stepquant.sglang \
  --model-path /path/to/Qwen3.8-27B --tp-size 4 \
  --state-format stepquant6 --plan artifacts/reproduction/plans/qwen-stepquant6.pt \
  --attention-backend triton --reasoning-parser qwen3 \
  --context-length 131072 --chunked-prefill-size 2048 \
  --mem-fraction-static 0.85 --served-model-name stepquant \
  --host 127.0.0.1 --port 31080
```

For Kimi, use its checkpoint and `kimi-stepquant6.pt`, add `--trust-remote-code`, and omit `--reasoning-parser qwen3`. For STEPQuant@4, change both the format and plan to `stepquant4`.

## 4. Evaluate

Stop a manually launched server first; the evaluation suite launches its own.

```bash
.venv-eval/bin/python -m stepquant.evaluation.suite \
  --models qwen kimi --benchmarks all --formats fp32 stepquant4 stepquant6 \
  --server-python .venv-serving/bin/python --devices 0,1,2,3 --tp 4
```

Results are saved under `artifacts/evaluation/`. Use `--benchmarks long` or `--benchmarks short` for a subset, and `--dry-run` to inspect the launch commands.

Short tasks use zero-shot greedy generation, at most **2048 output tokens**, and **thinking disabled**. Long-generation task settings and official grader revisions are recorded in `configs/evaluation/`. Both HumanEval+ and MBPP+ are provided through EvalPlus.

When generation settings change, frozen task files receive a separate protocol snapshot and results use a protocol-specific directory. Previous task files and results remain intact.

## 5. Decode throughput

Use the BF16 checkpoints and @6 plans configured in `configs/models.json`. Each job runs matching native FP32-state and STEPQuant servers with **128 prompt + 1024 decode tokens**, three rounds, and batches **32, 64, 128, 256, 512**:

```bash
.venv-serving/bin/python benchmarks/run_preset.py \
  --preset configs/benchmarks/stepquant6.json \
  --models qwen kimi --devices 0,1,2,3 \
  --output artifacts/benchmarks
```

Use `--batches 32 64` for a subset or `--dry-run` to inspect commands. The same runner accepts `configs/benchmarks/stepquant4.json` for @4. Results are written below the format-specific directory under `artifacts/benchmarks/`.

For a per-layer kernel benchmark, supply a calibrated plan. The format and architecture are selected from that plan, and FP32 is included as the comparison:

```bash
.venv-serving/bin/python benchmarks/bench_formats.py \
  --plan artifacts/reproduction/plans/qwen-stepquant6.pt \
  --async-writeback --raw-inputs --batch-sizes 32 64 128 256 512 \
  --output artifacts/benchmarks/qwen-stepquant6-kernel.json
```

Use `--layer` to select a calibration layer and `--tp-size` / `--tp-rank` to select its shard. These per-layer timings are separate from full-model throughput. Without a plan, the microbenchmark runs only the paper baselines.

Public MoE tuning templates use capacity-neutral device names. The launcher creates SGLang matching files using the actual runtime device name under the ignored `artifacts/runtime/moe/` directory. Tuning values are preserved; benchmark identities reference the public templates.

Figure 3(c) uses aggregate serving-memory accounting with **W4/AWQ weights and five prefix-state slots per request**. This is separate from the BF16 decode-throughput protocol above. Task evaluation disables radix caching.
