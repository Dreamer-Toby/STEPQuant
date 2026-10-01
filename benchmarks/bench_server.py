"""Fixed-token real-model requests with separate HTTP and per-rank CUDA timings.

Set STEPQUANT_TIMING_DIR when launching the server to collect CUDA decode times.
These synthetic prompts measure performance, not reasoning accuracy.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
from stepquant.device import public_metadata
import argparse
import json
import statistics
import time
import urllib.request
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url',default='http://127.0.0.1:31080')
    parser.add_argument('--batch',type=int,default=64)
    parser.add_argument('--prompt-tokens',type=int,default=128)
    parser.add_argument('--prompt-mode',choices=('varied_tokens','matrix'),default='varied_tokens')
    parser.add_argument('--tokenizer')
    parser.add_argument('--output-tokens',type=int,default=1024)
    parser.add_argument('--rounds',type=int,default=3)
    parser.add_argument('--output',required=True)
    parser.add_argument('--profile-at-token',type=int,default=0)
    parser.add_argument('--profile-dir')
    parser.add_argument('--profile-steps',type=int,default=8)
    args=parser.parse_args()
    if min(args.batch,args.prompt_tokens,args.output_tokens,args.rounds)<1:parser.error('counts must be positive')
    def request(path,payload=None):
        data=json.dumps(payload).encode() if payload is not None else None
        req=urllib.request.Request(args.url+path,data=data,headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(req,timeout=7200) as response:
            return response.read().decode() if path=='/start_profile' else json.load(response)
    if args.prompt_mode=='matrix':
        if not args.tokenizer:parser.error('--tokenizer is required for matrix prompts')
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(args.tokenizer,local_files_only=True,trust_remote_code=True)
        text=tokenizer.encode('Explain how matrix multiplication works, with examples and detailed steps. ',add_special_tokens=False)
        prompt=(text*((args.prompt_tokens+len(text)-1)//len(text)))[:args.prompt_tokens]
        prompts=[prompt for _ in range(args.batch)]
    else:
        prompts=[[42]*(args.prompt_tokens-1)+[100+i] for i in range(args.batch)]
    prompt_sha256=hashlib.sha256(json.dumps(prompts,separators=(',',':')).encode()).hexdigest()
    def generate(tokens):
        started=time.perf_counter()
        payload={'input_ids':prompts,'sampling_params':{
            'temperature':0,'max_new_tokens':tokens,'ignore_eos':True}}
        if args.profile_at_token and tokens>args.profile_at_token:
            if not args.profile_dir:parser.error('--profile-dir is required for delayed profiling')
            payload['stream']=True
            # Large batches otherwise emit a full-text SSE update for every
            # token and can finish decoding before the client reaches the trigger.
            payload['sampling_params']['stream_interval']=16
            req=urllib.request.Request(args.url+'/generate',data=json.dumps(payload).encode(),
                                       headers={'Content-Type':'application/json'})
            result=[None]*args.batch
            future=None
            with ThreadPoolExecutor(max_workers=1) as executor, urllib.request.urlopen(req,timeout=7200) as response:
                for line in response:
                    if not line.startswith(b'data: ') or line.strip()==b'data: [DONE]':continue
                    event=json.loads(line[6:]);result[event.get('index',0)]=event
                    if future is None and event['meta_info']['completion_tokens']>=args.profile_at_token:
                        future=executor.submit(request,'/start_profile',dict(output_dir=args.profile_dir,
                            num_steps=args.profile_steps,activities=['GPU'],profile_by_stage=True,
                            profile_stages=['decode']))
                if future is None:raise RuntimeError('generation ended before the requested profile token')
                future.result()
        else:
            result=request('/generate',payload)
        elapsed=time.perf_counter()-started
        assert len(result)==args.batch
        assert all(r['meta_info']['completion_tokens']==tokens for r in result)
        report=dict(wall_seconds=elapsed,output_tokens=args.batch*tokens,
                    output_tokens_per_second=args.batch*tokens/elapsed,
                    output_text_sha256=hashlib.sha256(json.dumps([r.get('text','') for r in result],ensure_ascii=False).encode()).hexdigest(),
                    unique_output_texts=len({r.get('text','') for r in result}),
                    sample_text=result[0].get('text','')[:2000])
        if args.profile_at_token:
            report['sample_texts']=[r.get('text','')[:2000] for r in result[:3]]
        return report
    generate(16)
    info=request('/server_info')
    before=info['internal_states']
    capacities=[int(state['memory_usage']['token_capacity']) for state in before]
    capacity=min(capacities)
    if capacity<args.batch*(args.prompt_tokens+args.output_tokens):
        raise RuntimeError(f'fixed batch needs {args.batch*(args.prompt_tokens+args.output_tokens)} KV tokens, but only {capacity} are allocated')
    keys=('model_path','tp_size','dtype','context_length','max_total_tokens','max_running_requests',
          'attention_backend','linear_attn_backend','disable_cuda_graph','cuda_graph_max_bs',
          'mem_fraction_static','chunked_prefill_size')
    configuration={key:info.get(key) for key in keys}
    configuration['actual_token_capacity']=capacity
    rounds=[generate(args.output_tokens) for _ in range(args.rounds)]
    after=request('/server_info')['internal_states']
    report=dict(batch=args.batch,prompt_tokens=args.prompt_tokens,output_tokens=args.output_tokens,
                prompt_mode=args.prompt_mode,prompt_sha256=prompt_sha256,
                scope='synthetic fixed-token real-model performance, not accuracy',rounds=rounds,
                median_wall_seconds=statistics.median(r['wall_seconds'] for r in rounds),
                configuration=configuration,before=before,after=after)
    Path(args.output).write_text(json.dumps(public_metadata(report),indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('batch','median_wall_seconds')}),flush=True)


if __name__=='__main__':main()
