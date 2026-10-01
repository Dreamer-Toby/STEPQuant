"""Prepare, generate, resume and grade the frozen evaluation matrix.

Use --benchmarks all/long/short and --formats all for the full matrix.
--dry-run validates plans and emits commands without starting a server.
--limit-tasks/--max-tokens/--samples explicitly label an integration smoke run.
"""
import argparse
import itertools
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from .pipeline import request, run
from .data import FrozenTasks, prepare_protocol, protocol_identity
from .grade import grade
from stepquant.formats import get_format
from stepquant.sglang.config import request_limit, graph_batches


def validate_plan(path, model, fmt, allow_smoke=False):
    import torch
    from stepquant.cli import fingerprint
    value=torch.load(path,map_location='cpu',weights_only=True)
    if value.get('checkpoint_fingerprint')!=fingerprint(model['model_path']):
        raise ValueError(f'checkpoint/plan mismatch: {path}')
    if value.get('format_version')!=2:
        raise ValueError(f'unsupported plan: {path}')
    from stepquant.reproducibility import checkpoint_identity
    if value.get('checkpoint_identity') and value['checkpoint_identity']!=checkpoint_identity(model['model_path']):
        raise ValueError(f'checkpoint files changed since calibration: {path}')
    if fmt in ('stepquant4','stepquant6'):
        if value.get('state_format')!='stepquant' or value.get('settings',{}).get('nominal_bits')!=get_format(fmt).bits:
            raise ValueError(f'plan bit budget mismatch: {path}')
    elif value.get('state_format')!=fmt:
        raise ValueError(f'plan format mismatch: {path}')
    data=value.get('data',{})
    if not allow_smoke:
        expected=(32,2048)
        if (data.get('segments'),data.get('sequence_length'))!=expected or not data.get('token_ids_sha256'):
            raise ValueError(f'unverified/smoke calibration: {path}; calibrate with the reproduction entrypoint')
        if not value.get('checkpoint_identity'):
            raise ValueError(f'missing calibration checkpoint identity: {path}')
        protocol=json.loads(Path('configs/calibration/paper.json').read_text())
        if (data.get('protocol')!='wikitext' or data.get('source')!=protocol['source']
                or data.get('revision')!=protocol['revision']
                or any(data.get(k)!=protocol[k] for k in ('seed','sample_every','max_snapshots'))):
            raise ValueError(f'calibration protocol mismatch: {path}')
    return value


def server_command(a, model, fmt, workload):
    profile=json.loads(Path(f'configs/serving/{workload}.json').read_text())
    cap=request_limit(workload,a.max_running_requests or profile['max_running_requests'])
    cmd=[a.server_python,'-m','stepquant.sglang','--state-format',fmt,'--workload',workload,
         '--max-running-requests',str(cap),'--model-path',model['model_path'],'--tp-size',str(a.tp),
         '--attention-backend',a.attention_backend,'--context-length',str(a.context_length or profile['context_length']),
         '--chunked-prefill-size',str(profile['chunked_prefill_size']),'--mem-fraction-static',str(a.mem_fraction_static),
         '--disable-piecewise-cuda-graph','--host','127.0.0.1','--port',str(a.port)]
    if a.disable_cuda_graph:cmd+=['--disable-cuda-graph']
    else:
        graph_cap=min(a.cuda_graph_max_bs or cap,cap)
        cmd+=['--cuda-graph-bs',*[str(n) for n in graph_batches(graph_cap)],'--cuda-graph-max-bs',str(graph_cap)]
    if fmt=='fp32':cmd+=['--kernel-backend','native']
    if get_format(fmt).kind in ('stepquant',):
        cmd+=['--state-storage',a.stepquant_storage,'--fit-mode','requantize']
    if get_format(fmt).calibrated:
        path=model['plans'][fmt]
        validate_plan(path,model,fmt,a.allow_smoke_plan)
        cmd+=['--plan',path]
    if model.get('reasoning_parser'):cmd+=['--reasoning-parser',model['reasoning_parser']]
    if model.get('trust_remote_code'):cmd+=['--trust-remote-code']
    if model['weights']=='awq_int4':cmd+=['--quantization','awq']
    elif model['weights']!='bf16':raise ValueError('weights must be bf16 or awq_int4')
    return cmd,cap


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models-config',default='configs/models.json')
    p.add_argument('--models',nargs='+',default=['qwen','kimi'])
    p.add_argument('--formats',nargs='+',default=['stepquant4','stepquant6'])
    p.add_argument('--benchmarks',nargs='+',required=True)
    p.add_argument('--stepquant-storage',choices=('packed','byte'),default='packed')
    p.add_argument('--server-python',default='.venv-serving-fast/bin/python')
    p.add_argument('--devices',help='explicit visible CUDA devices, e.g. 0,1,2,3')
    p.add_argument('--max-running-requests',type=int)
    p.add_argument('--tp',type=int,default=4)
    p.add_argument('--port',type=int,default=31080)
    p.add_argument('--output',default='artifacts/evaluation')
    p.add_argument('--data-dir',default='artifacts/eval-data')
    p.add_argument('--upstream',default='artifacts/upstream')
    p.add_argument('--processes',type=int,default=4)
    stage=p.add_mutually_exclusive_group()
    stage.add_argument('--prepare-only',action='store_true')
    stage.add_argument('--grade-only',action='store_true')
    stage.add_argument('--dry-run',action='store_true')
    p.add_argument('--disable-cuda-graph',action='store_true')
    p.add_argument('--cuda-graph-max-bs',type=int)
    p.add_argument('--attention-backend',choices=('triton','flashinfer'),default='triton')
    p.add_argument('--mem-fraction-static',type=float,default=.85)
    p.add_argument('--context-length',type=int)
    p.add_argument('--allow-smoke-plan',action='store_true',help='development only; never paper calibration')
    p.add_argument('--limit-tasks',type=int)
    p.add_argument('--max-tokens',type=int)
    p.add_argument('--samples',type=int)
    a=p.parse_args()
    for key in ('tp','processes','limit_tasks','max_tokens','samples','context_length','cuda_graph_max_bs'):
        if getattr(a,key) is not None and getattr(a,key)<1:p.error(f'{key} must be positive')
    if not 0<a.mem_fraction_static<1:p.error('memory fraction must be in (0,1)')
    if a.max_running_requests is not None:request_limit('long',a.max_running_requests)
    if not (a.prepare_only or a.grade_only or a.dry_run):
        if not a.devices:p.error('--devices is required to launch servers')
        devices=a.devices.split(',')
        if len(devices)!=a.tp or len(set(devices))!=a.tp or any(not d.strip() for d in devices):
            p.error('--devices must contain exactly --tp distinct devices')
    models=json.loads(Path(a.models_config).read_text())
    available={p.stem:json.loads(p.read_text()) for p in Path('configs/evaluation').glob('*.json') if p.name!='upstream.lock.json'}
    names=[]
    for name in a.benchmarks:
        names.extend([n for n,c in available.items() if name=='all' or c['workload']==name] if name in ('all','long','short') else [name])
    configs=[available[n] for n in dict.fromkeys(names)]
    formats=sorted(p.stem for p in Path('configs/formats').glob('*.json')) if a.formats==['all'] else a.formats
    root=Path(a.output);root.mkdir(parents=True,exist_ok=True)
    launches={}
    if not (a.prepare_only or a.grade_only):
        if not Path(a.server_python).is_file():raise FileNotFoundError(a.server_python)
        for model_name in a.models:
            if not (Path(models[model_name]['model_path'])/'config.json').is_file():
                raise FileNotFoundError(models[model_name]['model_path'])
            for fmt in formats:
                for workload in sorted({c['workload'] for c in configs}):
                    launches[model_name,fmt,workload]=server_command(a,models[model_name],fmt,workload)
    tasks={}
    if not a.dry_run:
        for c in configs:
            path=Path(a.data_dir)/f"{c['name']}.jsonl"
            path=prepare_protocol(c,path,a.upstream)
            tasks[c['name']]=FrozenTasks(path,c)
            print(f"Verified {c['name']}: {len(tasks[c['name']])} tasks",flush=True)
    if a.prepare_only:return
    for model_name in a.models:
        model=models[model_name]
        for fmt in formats:
            for workload in sorted({c['workload'] for c in configs}):
                dest=root/model_name/fmt/workload
                dest.mkdir(parents=True,exist_ok=True)
                if not a.grade_only:
                    cmd,cap=launches[model_name,fmt,workload]
                    if a.dry_run:
                        print(json.dumps(dict(model=model_name,format=fmt,workload=workload,command=cmd)));continue
                else:cap=request_limit(workload,a.max_running_requests)
                def evaluate():
                    for base in configs:
                        if base['workload']!=workload:continue
                        out=dest/base['name']/protocol_identity(base);c=dict(base,max_running_requests=cap)
                        selected=tasks[c['name']]
                        if a.limit_tasks:selected=list(itertools.islice(selected,a.limit_tasks))
                        if a.max_tokens:c['sampling']=dict(c['sampling'],max_tokens=a.max_tokens)
                        if a.samples:c['samples']=a.samples
                        if a.limit_tasks or a.max_tokens or a.samples or a.allow_smoke_plan:
                            c.update(protocol_scope='integration_smoke',selected_tasks=len(selected))
                        if a.grade_only:
                            # Exact generation settings, including smoke overrides, come from the manifest.
                            c=json.loads((out/'manifest.json').read_text())['config']
                            if c.get('selected_tasks'):selected=list(itertools.islice(tasks[c['name']],c['selected_tasks']))
                        else:run(c,selected,out,url)
                        grade(c,selected,out,a.upstream,a.processes)
                if a.grade_only:evaluate();continue
                (dest/'server-command.json').write_text(json.dumps(cmd,indent=2)+'\n')
                env=dict(os.environ,SGLANG_PLUGINS='stepquant',CUDA_VISIBLE_DEVICES=a.devices,CUDA_DEVICE_MAX_CONNECTIONS='8')
                env.setdefault('CUDA_HOME',str(Path('artifacts/cuda13').resolve()))
                env['PATH']=str(Path(a.server_python).absolute().parent)+os.pathsep+env['PATH']
                url=f'http://127.0.0.1:{a.port}'
                with socket.socket() as probe:
                    if probe.connect_ex(('127.0.0.1',a.port))==0:raise RuntimeError(f'port {a.port} is already in use')
                with (dest/'server.log').open('w') as log:
                    server=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
                    try:
                        deadline=time.monotonic()+1200
                        while time.monotonic()<deadline:
                            if server.poll() is not None:raise RuntimeError(f'server failed: {dest}/server.log')
                            try:request(url+'/health_generate',timeout=5);break
                            except Exception:time.sleep(1)
                        else:raise TimeoutError('server startup timed out')
                        evaluate()
                    finally:
                        if server.poll() is None:
                            os.killpg(server.pid,signal.SIGINT)
                            try:server.wait(timeout=30)
                            except subprocess.TimeoutExpired:os.killpg(server.pid,signal.SIGKILL);server.wait()


if __name__=='__main__':main()
