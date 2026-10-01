"""Opt-in per-rank CUDA start-to-start decode timing; no synchronization per token."""
import json
import hashlib
import os
from pathlib import Path
import torch
from importlib.metadata import version

_kernel_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted((Path(__file__).parents[1]/'kernels').glob('*.py'))}
_moe_root=Path(os.environ['SGLANG_MOE_CONFIG_DIR']) if os.environ.get('SGLANG_MOE_CONFIG_DIR') else None
from stepquant.device import tuning_identity, public_metadata
_moe_sha256=json.loads(os.environ['STEPQUANT_MOE_TEMPLATE_SHA256']) if os.environ.get('STEPQUANT_MOE_TEMPLATE_SHA256') else (tuning_identity(_moe_root) if _moe_root else {})
_versions={name:version(name) for name in ('torch','triton','sglang')}
_events=[]
_segment=0
_rank=0
_dropped=0


def forward(original,self,forward_batch,*args,**kwargs):
    global _segment,_rank,_dropped
    if forward_batch.forward_mode.is_decode():
        if len(_events)<int(os.environ.get('STEPQUANT_TIMING_LIMIT','16384')):
            event=torch.cuda.Event(enable_timing=True)
            event.record()
            _rank=self.tp_rank
            _events.append((event,_segment,int(forward_batch.batch_size)))
        else:_dropped+=1
    else:
        _segment+=1
    return original(self,forward_batch,*args,**kwargs)


def internal_state(original,self,request):
    result=original(self,request)
    # The benchmark explicitly requests this snapshot after generation finishes.
    # Events include scheduler/CPU gaps between decode starts, but exclude prefill.
    torch.cuda.synchronize()
    rows=[]
    for (first,segment,batch),(second,next_segment,next_batch) in zip(_events,_events[1:]):
        if segment==next_segment and batch==next_batch:
            rows.append(dict(segment=segment,batch=batch,start_to_start_ms=first.elapsed_time(second)))
    directory=Path(os.environ['STEPQUANT_TIMING_DIR'])
    directory.mkdir(parents=True,exist_ok=True)
    path=directory/f'rank-{_rank}.json'
    from .runtime import _writeback_profiles
    chunk=os.environ.get('STEPQUANT_WRITEBACK_CHUNK','0')
    automatic_sms=os.environ.get('STEPQUANT_WRITEBACK_SMS')=='auto'
    report=dict(rank=_rank,kernel_sha256=_kernel_sha256,moe_config_sha256=_moe_sha256,versions=_versions,scope='CUDA decode start-to-start intervals; excludes prefill, includes step gaps',
                state_format=os.environ.get('STEPQUANT_STATE_FORMAT'),
                fit_mode='requantize',
                state_storage='native' if os.environ.get('STEPQUANT_KERNEL_BACKEND')=='native' else os.environ.get('STEPQUANT_STATE_STORAGE','packed'),
                kernel_backend=os.environ.get('STEPQUANT_KERNEL_BACKEND','auto'),
                async_writeback=os.environ.get('STEPQUANT_ASYNC_WRITEBACK')=='1',
                writeback_sms=None if automatic_sms else int(os.environ.get('STEPQUANT_WRITEBACK_ACTUAL_SMS','0')),
                writeback_chunk=None if chunk=='auto' else int(chunk),
                writeback_profiles=dict(_writeback_profiles),
                dropped_events=_dropped,intervals=rows)
    path.write_text(json.dumps(public_metadata(report),indent=2)+'\n')
    result.internal_state['stepquant_decode_timing']=dict(path=str(path),last_segment=_segment,event_count=len(_events),dropped_events=_dropped)
    return result
