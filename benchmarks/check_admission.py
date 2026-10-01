"""Real server check: full generation budget, continuous cap, pause and resume."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
from stepquant.evaluation.pipeline import request

p=argparse.ArgumentParser()
p.add_argument('--url',default='http://127.0.0.1:31080')
p.add_argument('--output',required=True)
p.add_argument('--cap',type=int,default=64)
p.add_argument('--proof-budget',type=int,default=65536)
p.add_argument('--pressure-prompt-repeats',type=int,default=90)
p.add_argument('--pressure-requests',type=int,default=48)
p.add_argument('--pressure-tokens',type=int,default=24)
a=p.parse_args()
start=time.monotonic()
# A 65k budget must not reserve 65k KV tokens before a short answer can run.
proof=request(a.url+'/generate',dict(text='The capital of France is',sampling_params=dict(temperature=0,max_new_tokens=a.proof_budget,stop=['.','\n'])))
assert proof['meta_info']['completion_tokens']>0

def generate(i):
    return request(a.url+'/generate',dict(text='Count: 1, 2, 3,',sampling_params=dict(temperature=0,max_new_tokens=48,ignore_eos=True)))
with ThreadPoolExecutor(max_workers=a.cap*2) as pool:
    responses=list(pool.map(generate,range(a.cap*2)))
assert all(r['meta_info']['completion_tokens']==48 for r in responses)
# Larger prompts fill the small validation pool past the watermark.
def pressured(i):
    return request(a.url+'/generate',dict(text=('The sky is blue. '*a.pressure_prompt_repeats)+'The capital of France is',sampling_params=dict(temperature=0,max_new_tokens=a.pressure_tokens,ignore_eos=True)))
with ThreadPoolExecutor(max_workers=a.cap*2) as pool:
    pressure=list(pool.map(pressured,range(a.pressure_requests)))
assert all(r['meta_info']['completion_tokens']==a.pressure_tokens for r in pressure)
info=request(a.url+'/get_server_info')
metrics=info['internal_states'][0]['stepquant_admission']
assert metrics['paused']>0,metrics
assert metrics['running_requests']==0 and metrics['kv_used_tokens']==0,metrics
assert metrics['peak_running_requests']<=a.cap,metrics
Path(a.output).write_text(json.dumps(dict(metrics=metrics,proof=proof,requests=len(responses)+len(pressure),seconds=time.monotonic()-start),indent=2)+'\n')
print(json.dumps(metrics))
