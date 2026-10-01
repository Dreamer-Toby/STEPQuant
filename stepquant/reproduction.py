"""Collect once per checkpoint, then fit reproducible STEPQuant plans.

Example: python -m stepquant.reproduction --models qwen kimi --devices 0,1,2,3
Use --dry-run to print commands. Calibration uses the existing per-model Python
from configs/models.json; fitting reuses its saved statistics. No environments
or model weights are downloaded.
Evaluation: python -m stepquant.evaluation.suite --benchmarks all --formats all
            --devices 0,1,2,3 --tp 4
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
from .reproducibility import source_identity, file_sha256


def commands(model_name, model, root, formats, smoke=False):
    python=model['calibration_python']
    protocol=json.loads(Path('configs/calibration/paper.json').read_text())
    plans=root/'plans';stats=root/f'{model_name}-statistics.pt'
    shared=['--model',model['model_path'],'--horizon',str(protocol['horizon'])]
    selected=list(formats)
    if any(f not in ('stepquant4','stepquant6') for f in selected):
        raise ValueError('only STEPQuant formats require calibration')
    def options(fmt):
        return ['--bits','6' if fmt=='stepquant6' else '4','--state-format',
                'stepquant']
    jobs=[]
    if selected:
        first=selected[0]
        jobs.append([python,'-m','stepquant','calibrate',*shared,*options(first),
            '--segments',str(1 if smoke else protocol['segments']),
            '--sequence-length',str(64 if smoke else protocol['sequence_length']),
            '--sample-every',str(protocol['sample_every']),'--max-snapshots',str(2 if smoke else protocol['max_snapshots']),
            '--seed',str(protocol['seed']),'--data-revision',protocol['revision'],
            '--statistics-output',str(stats),'--output',str(plans/f'{model_name}-{first}.pt')])
        for fmt in selected[1:]:
            jobs.append([python,'-m','stepquant','fit',*shared,*options(fmt),
                         '--statistics',str(stats),'--output',str(plans/f'{model_name}-{fmt}.pt')])
    return jobs


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--models-config',default='configs/models.json')
    p.add_argument('--models',nargs='+',default=['qwen','kimi'])
    p.add_argument('--formats',nargs='+',choices=['stepquant4','stepquant6'],
                   default=['stepquant4','stepquant6'])
    p.add_argument('--devices',help='explicit visible GPUs used by calibration device_map=auto')
    p.add_argument('--output',default='artifacts/reproduction')
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--smoke',action='store_true',help='64-token integration only; never paper calibration')
    a=p.parse_args()
    if not a.dry_run and not a.devices:p.error('--devices is required')
    if a.smoke and a.output=='artifacts/reproduction':p.error('--smoke requires a separate --output directory')
    models=json.loads(Path(a.models_config).read_text());root=Path(a.output)
    matrix={name:commands(name,models[name],root,a.formats,smoke=a.smoke) for name in a.models}
    if a.dry_run:print(json.dumps(matrix,indent=2));return
    root.mkdir(parents=True,exist_ok=True)
    manifest=root/'calibration-run.json'
    identity=dict(commands=matrix,devices=a.devices,smoke=a.smoke,source_sha256=source_identity())
    completed={}
    if manifest.exists():
        previous=json.loads(manifest.read_text())
        completed=previous.pop('completed',{})
        if previous!=identity:raise ValueError('calibration resume identity mismatch; use a new output directory')
    def persist():
        temporary=manifest.with_suffix('.tmp')
        temporary.write_text(json.dumps(dict(identity,completed=completed),indent=2)+'\n')
        temporary.replace(manifest)
    persist()
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=a.devices)
    for name,jobs in matrix.items():
        for i,cmd in enumerate(jobs):
            key=f'{name}-{i}'
            if key in completed:
                if any(file_sha256(path)!=sha for path,sha in completed[key].items()):
                    raise ValueError(f'calibration outputs changed: {key}')
                print(f'Reused completed calibration: {key}',flush=True)
                continue
            with (root/f'{name}-{i}.log').open('w') as log:
                subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            outputs=[cmd[cmd.index('--output')+1]]
            if '--statistics-output' in cmd:outputs.append(cmd[cmd.index('--statistics-output')+1])
            completed[key]={path:file_sha256(path) for path in outputs}
            persist()
            print(f'Completed {name}: {cmd[-1]}',flush=True)


if __name__=='__main__':main()
