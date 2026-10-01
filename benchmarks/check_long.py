"""Independent 65k-budget requests at a chosen batch, then sustained decode."""
import argparse
import json
from pathlib import Path
import time
from stepquant.evaluation.pipeline import request

p=argparse.ArgumentParser()
p.add_argument('--url',default='http://127.0.0.1:31080')
p.add_argument('--batch',type=int,default=64)
p.add_argument('--tokens',type=int,default=8192)
p.add_argument('--output',required=True)
a=p.parse_args()
if not 1<=a.batch<=512:p.error('batch must be 1..512')
start=time.monotonic()
r=request(a.url+'/generate',dict(text=['The capital of France is']*a.batch,sampling_params=dict(temperature=0,max_new_tokens=65536,stop=['.','\n'])))
assert all(x['meta_info']['completion_tokens']>0 for x in r)
metrics=request(a.url+'/get_server_info')['internal_states'][0]['stepquant_admission']
assert metrics['peak_running_requests']==a.batch,metrics
print(f'{a.batch}-slot full-budget admission passed',flush=True)
long_start=time.monotonic()
long=request(a.url+'/generate',dict(text='Write out a careful mathematical derivation, explaining every step. Start by counting the positive integers: 1, 2, 3,',sampling_params=dict(temperature=0,max_new_tokens=a.tokens,ignore_eos=True)))
assert long['meta_info']['completion_tokens']==a.tokens,long['meta_info']
# A new request must observe a clean recycled slot, not the long sample's state.
reused=request(a.url+'/generate',dict(text='The capital of France is',sampling_params=dict(temperature=0,max_new_tokens=65536,stop=['.','\n'])))
assert reused['text']==r[0]['text']
info=request(a.url+'/get_server_info')['internal_states'][0]
metrics=info['stepquant_admission']
assert metrics['kv_used_tokens']==0 and metrics['running_requests']==0,metrics
assert metrics.get('retracted_requests',0)==0 and metrics.get('aborted_requests',0)==0,metrics
report=dict(metrics=metrics,long_meta=long['meta_info'],long_seconds=time.monotonic()-long_start,
            total_seconds=time.monotonic()-start,text_prefix=long['text'][:500],reused_text=reused['text'],
            state_format=info['stepquant_state_format'],plan_sha256=info['stepquant_plan_sha256'])
Path(a.output).write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
print(json.dumps(report))
