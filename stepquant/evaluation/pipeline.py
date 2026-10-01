"""Continuously replenish independent samples; persist every completed response.

The server, not this client, controls KV admission. No static n-way sample batch.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import hashlib
from stepquant.sglang.config import request_limit
import json
import os
from pathlib import Path
import time
import urllib.request
from .data import tasks_digest
from stepquant.reproducibility import source_identity
from stepquant.device import public_metadata


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def request(url, body=None, timeout=86400):
    payload=None if body is None else json.dumps(body).encode()
    req=urllib.request.Request(url,data=payload,headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=timeout) as response:
        data=response.read()
        return json.loads(data) if data else None


def seed_for(base, task, sample):
    return int(digest([base,task,sample])[:8],16) & 0x7fffffff


def read_results(path):
    """Recover only an interrupted final JSONL write; never swallow interior damage."""
    if not path.exists():
        return []
    data=path.read_bytes()
    if data and not data.endswith(b'\n'):
        # Every committed record is newline terminated and fsynced.
        end=data.rfind(b'\n')+1
        path.write_bytes(data[:end])
        data=data[:end]
    return [json.loads(line) for line in data.splitlines()]


def run(config, tasks, output, url, *, workers=None, timeout=86400):
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    task_ids={str(t['id']) for t in tasks}
    if len(task_ids)!=len(tasks):
        raise ValueError('duplicate task identifiers')
    cap=request_limit(config['workload'],config.get('max_running_requests'))
    workers=workers or cap*2
    if workers<1:raise ValueError('workers must be positive')
    if not tasks or config['samples']<1:raise ValueError('nonempty tasks and positive samples required')
    server=public_metadata(request(url+'/get_server_info',timeout=60))
    initial_path=output/'server-initial.json'
    if not initial_path.exists():initial_path.write_text(json.dumps(server,indent=2)+'\n')
    state=server.get('internal_states',[server])[0]
    admission=state.get('stepquant_admission')
    if admission is None or abs(admission['watermark']-.6)>1e-9:
        raise ValueError('server must enable --workload and --kv-watermark 0.6')
    if state.get('max_running_requests',cap)>cap:
        raise ValueError('server batch cap exceeds evaluation workload')
    identity={k:state.get(k) for k in ('model_path','dtype','quantization','tp_size','max_running_requests','context_length','max_total_tokens')}
    identity['versions']=state.get('stepquant_versions')
    identity['state_format']=state.get('stepquant_state_format')
    identity['plan_sha256']=state.get('stepquant_plan_sha256')
    identity['runtime']=state.get('stepquant_runtime_identity')
    if not identity['state_format']:
        raise ValueError('server does not report state format identity')
    if not identity['runtime']:
        raise ValueError('server does not report runtime/checkpoint identity; update the STEPQuant plugin')
    manifest=dict(config=config,tasks_sha256=tasks_digest(tasks),server_identity=identity,
                  source_sha256=source_identity(),workers=workers)
    manifest['run_hash']=digest(manifest)
    manifest_path=output/'manifest.json'
    if manifest_path.exists():
        old=json.loads(manifest_path.read_text())
        if old['run_hash']!=manifest['run_hash']:
            raise ValueError('resume configuration/data/model mismatch; choose a new output directory')
    else:
        manifest_path.write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
    results=output/'responses.jsonl'
    records=read_results(results)
    for r in records:
        if (r.get('run_hash')!=manifest['run_hash'] or str(r.get('task_id')) not in task_ids
                or not isinstance(r.get('sample'),int) or not 0<=r['sample']<config['samples']
                or r.get('sample_id')!=f"{r['task_id']}:{r['sample']}"):
            raise ValueError('response identity mismatch; refusing to resume')
    completed={r['sample_id'] for r in records if 'error' not in r}
    if len(completed)!=sum('error' not in r for r in records):
        raise ValueError('duplicate successful samples')
    jobs=(({k:v for k,v in task.items() if k!='source'},sample) for task in tasks for sample in range(config['samples'])
          if f"{task['id']}:{sample}" not in completed)
    def generate(job):
        task,sample=job
        sample_id=f"{task['id']}:{sample}"
        params=dict(config['sampling'])
        params['seed']=seed_for(config['seed'],str(task['id']),sample)
        body=dict(model='default',messages=task['messages'],**params)
        if 'chat_template_kwargs' in config:
            body['chat_template_kwargs']=config['chat_template_kwargs']
        start=time.monotonic()
        record=dict(sample_id=sample_id,task_id=str(task['id']),sample=sample,seed=params['seed'],run_hash=manifest['run_hash'])
        try:
            result=request(url+'/v1/chat/completions',body,timeout)
            choice=result['choices'][0]
            record.update(response=result,answer=choice['message'].get('content') or '',
                          reasoning=choice['message'].get('reasoning_content') or '',
                          finish_reason=choice['finish_reason'])
        except Exception as exc:
            record['error']=str(exc)
        record['seconds']=time.monotonic()-start
        return record
    failures=0
    with ThreadPoolExecutor(max_workers=workers) as pool, results.open('a') as stream:
        pending=set()
        def fill():
            while len(pending)<workers:
                job=next(jobs,None)
                if job is None:
                    break
                pending.add(pool.submit(generate,job))
        fill()
        while pending:
            done,pending=wait(pending,return_when=FIRST_COMPLETED)
            for future in done:
                record=future.result()
                failures+='error' in record
                stream.write(json.dumps(record,ensure_ascii=False)+'\n')
                stream.flush()
                os.fsync(stream.fileno())
                print(json.dumps({k:record[k] for k in ('sample_id','seconds','error') if k in record}),flush=True)
            fill()
    final=public_metadata(request(url+'/get_server_info',timeout=60))
    (output/'server-final.json').write_text(json.dumps(final,indent=2)+'\n')
    if failures:
        raise RuntimeError(f'{failures} requests failed; rerun to retry only failed samples')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True)
    p.add_argument('--tasks',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--url',default='http://127.0.0.1:31080')
    p.add_argument('--workers',type=int)
    p.add_argument('--timeout',type=int,default=86400)
    a=p.parse_args()
    config=json.loads(Path(a.config).read_text())
    tasks=[json.loads(line) for line in Path(a.tasks).read_text().splitlines() if line.strip()]
    run(config,tasks,a.output,a.url,workers=a.workers,timeout=a.timeout)


if __name__=='__main__':
    main()
