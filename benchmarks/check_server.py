"""HTTP smoke check: single/batched/concurrent requests, slot reuse, chunked prefill."""
import argparse
import concurrent.futures
import json
import time
import urllib.request
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url',default='http://127.0.0.1:31080')
    p.add_argument('--output',required=True)
    args=p.parse_args()
    def request(text):
        body=json.dumps({'text':text,'sampling_params':{'temperature':0,'max_new_tokens':16}}).encode()
        req=urllib.request.Request(args.url+'/generate',data=body,headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(req,timeout=600) as response:
            result=json.load(response)
        for item in result if isinstance(result,list) else [result]:
            assert item['meta_info']['completion_tokens']>0
            assert item['text']
        return result
    started=time.monotonic()
    prompt='The capital of France is'
    single=request(prompt)
    batched=request([prompt,'Two plus two equals'])
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        concurrent_results=list(executor.map(request,[prompt]*4))
    reused=request(prompt)
    assert reused['text']==single['text'],'request-slot reuse changed deterministic output'
    long=request(('A recurrent state summarizes previous tokens. '*60)+prompt)
    report=dict(single=single,batched=batched,concurrent=concurrent_results,reused=reused,long_prefill=long,
                elapsed_seconds=time.monotonic()-started,
                scope='HTTP functionality only, not accuracy or throughput benchmark')
    Path(args.output).write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'status':'passed','output':args.output,'text':single['text']},ensure_ascii=False))


if __name__=='__main__':
    main()
