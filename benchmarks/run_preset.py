"""Run a paper benchmark preset as matched native-FP32/STEPQuant pairs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def commands(preset, models, batches, models_config, output, devices=None):
    common = list(preset['common_arguments'])
    if devices is not None:
        common[common.index('--devices')+1] = devices
    fmt = common[common.index('--state-format') + 1]
    if fmt not in ('stepquant4', 'stepquant6'):
        raise ValueError('paper presets require symmetric STEPQuant')
    jobs = {}
    for name in models:
        model = models_config[name]
        if model['weights'] != 'bf16':
            raise ValueError('decode throughput presets require BF16 checkpoints')
        for batch in batches:
            args = common + ['--model', model['model_path'], '--plan', model['plans'][fmt]]
            args += preset['models'][name]['batches'][str(batch)]
            jobs[f'{name}-b{batch}'] = [sys.executable, str(Path(__file__).with_name('run_comparison.py')),
                                      *args, '--output-dir', str(Path(output) / fmt / f'{name}-b{batch}')]
    return jobs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--preset', default='configs/benchmarks/stepquant6.json')
    p.add_argument('--models-config', default='configs/models.json')
    p.add_argument('--models', nargs='+', choices=('qwen', 'kimi'), default=['qwen', 'kimi'])
    p.add_argument('--batches', nargs='+', type=int, choices=(32, 64, 128, 256, 512), default=[32, 64, 128, 256, 512])
    p.add_argument('--devices', help='override visible CUDA devices for both modes')
    p.add_argument('--output', default='artifacts/benchmarks')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    preset = json.loads(Path(a.preset).read_text())
    models = json.loads(Path(a.models_config).read_text())
    jobs = commands(preset, a.models, a.batches, models, a.output, a.devices)
    if a.dry_run:
        print(json.dumps(jobs, indent=2))
        return
    env = dict(os.environ, **preset.get('environment', {}))
    if os.environ.get('CUDA_HOME'):
        env['CUDA_HOME'] = os.environ['CUDA_HOME']
    for name, cmd in jobs.items():
        print(f'Running {name}', flush=True)
        subprocess.run(cmd, env=env, check=True)


if __name__ == '__main__':
    main()
