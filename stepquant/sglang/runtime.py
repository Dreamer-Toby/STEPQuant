"""Packed Mamba pool and GDN/KDA dispatch adapters for the pinned SGLang version.

The upstream model, convolution, full-attention backend and scheduler remain native.
Prefill only reconstructs active slots; decode updates compressed pages directly.
"""
import dataclasses
import functools
import inspect
import logging
import os
import re
import weakref
import torch
from stepquant.quantization import QuantizationPlan
from stepquant.kernels.pool import PackedStatePool
from stepquant.formats import get_format, FormatPlan
from stepquant.kernels.format_pool import FormatStatePool, format_layout

log=logging.getLogger(__name__)
_temporal_pools=weakref.WeakSet()
_writeback_profiles={}


@functools.lru_cache(maxsize=1)
def artifact():
    value=torch.load(os.environ['STEPQUANT_PLAN'],map_location='cpu',weights_only=True)
    if value.get('format_version')!=2:
        raise ValueError('unsupported STEPQuant plan format')
    return value


def state_format():
    return get_format(os.environ.get("STEPQUANT_STATE_FORMAT", "stepquant"))


def writeback_settings(architecture, batch):
    """Resolve launch overrides independently from the STEPQuant@6 defaults."""
    from stepquant.kernels.profiles import stepquant6_profile
    nominal_six = state_format().name == 'stepquant6'
    profile = stepquant6_profile(architecture, batch) if nominal_six else None
    sms = os.environ.get('STEPQUANT_WRITEBACK_SMS', 'auto' if nominal_six else '32')
    chunk = os.environ.get('STEPQUANT_WRITEBACK_CHUNK', 'auto' if nominal_six else '64')
    return (profile.sms if sms == 'auto' else int(sms),
            profile.chunk if chunk == 'auto' else int(chunk))


def local_plans(layer_ids,heads,device,keys=128):
    from sglang.srt.distributed import get_tensor_model_parallel_rank,get_tensor_model_parallel_world_size
    rank,world=get_tensor_model_parallel_rank(),get_tensor_model_parallel_world_size()
    spec=state_format()
    if not spec.calibrated:
        architecture=os.environ["STEPQUANT_ARCHITECTURE"]
        return [FormatPlan(architecture,torch.full((heads,keys),spec.bits,dtype=torch.long),
                           torch.ones(heads,keys)).to(device) for _ in layer_ids]
    expected=artifact().get("state_format", "stepquant")
    if spec.name in ('stepquant4','stepquant6') and artifact().get('settings',{}).get('nominal_bits')!=spec.bits:
        raise ValueError('plan nominal bits do not match the requested STEPQuant format')
    if spec.kind=='stepquant' and expected!='stepquant':
        raise ValueError(f'{expected} plan requires its matching --state-format')
    if spec.kind != "stepquant" and expected != spec.name:
        raise ValueError(f"plan format {expected} does not match {spec.name}")
    cls=QuantizationPlan if spec.kind=="stepquant" else FormatPlan
    mapping={}
    for name,entry in artifact()['plans'].items():
        match=re.search(r'\.layers\.(\d+)\.',name)
        if not match:
            raise ValueError(f'cannot map layer name {name}')
        mapping[int(match[1])]=entry
    plans=[]
    for layer in layer_ids:
        p=cls(**{k:v for k,v in mapping[layer].items() if k in ("architecture","bits","impact","value_group_size")})
        if p.bits.shape[0]!=heads*world:
            raise ValueError('calibration head count does not match tensor-parallel shard')
        sl=slice(rank*heads,(rank+1)*heads)
        plans.append(cls(p.architecture,p.bits[sl].contiguous(),p.impact[sl].contiguous(),
                                      p.value_group_size).to(device))
    return plans


class TemporalPages:
    def __init__(self,pages):
        self.pages=pages
        _temporal_pools.add(self)
    def __getitem__(self,index):
        if not isinstance(index,int):
            raise NotImplementedError('use packed pool slot operations')
        return self.pages[index]
    def numel(self):
        return sum(p.nbytes for p in self.pages)
    def element_size(self):
        return 1
    @property
    def shape(self):
        return (self.numel(),)
    dtype=torch.uint8


def pool_init(original,self,**kwargs):
    if kwargs.get('speculative_num_draft_tokens') is not None:
        raise ValueError('STEPQuant currently does not support speculative decoding')
    params=kwargs['cache_params']
    heads,values,keys=params.shape.temporal  # SGLang uses [H,V,K]
    plans=local_plans(kwargs['mamba_layer_ids'],heads,kwargs['device'],keys)
    if any(p.bits.shape[1]!=keys for p in plans):
        raise ValueError('calibration key dimension mismatch')
    # Let SGLang initialize its convolution cache, allocators and lifecycle without
    # ever allocating the original dense recurrent pool (not even temporarily).
    empty_shape=dataclasses.replace(params.shape,temporal=(heads,0,0))
    tiny_params=dataclasses.replace(params,shape=empty_shape)
    original(self,**dict(kwargs,cache_params=tiny_params))
    stream=None
    architecture=plans[0].architecture if plans else os.environ.get('STEPQUANT_ARCHITECTURE','gdn')
    sms,chunk=writeback_settings(architecture, min(512, max(1, kwargs['size'])))
    if os.environ.get('STEPQUANT_ASYNC_WRITEBACK')=='1':
        from stepquant.kernels.resources import writeback_resources
        resources=writeback_resources(kwargs['device'],sms)
        stream=resources.stream
        log.warning('STEPQuant background SMs: requested=%d actual=%d total=%d',
                    resources.requested_sms,resources.actual_sms,resources.total_sms)
        os.environ['STEPQUANT_WRITEBACK_ACTUAL_SMS']=str(resources.actual_sms)
    spec=state_format()
    state_pool=PackedStatePool
    workspace={}
    pages=TemporalPages([(state_pool(p,kwargs['size']+1,values,writeback_stream=stream,writeback_workspace=workspace,
                         writeback_chunk=chunk,storage=os.environ.get('STEPQUANT_STATE_STORAGE','packed')) if spec.kind=='stepquant'
                          else FormatStatePool(p,kwargs['size']+1,values,spec,writeback_stream=stream)) for p in plans])
    self.mamba_cache=dataclasses.replace(self.mamba_cache,temporal=pages)
    self.mem_usage=self.mamba_cache.mem_usage_bytes()/1024**3
    dense=len(plans)*(kwargs['size']+1)*heads*values*keys*4
    log.warning('STEPQuant state pool (%s): %d bytes (FP32 %d bytes), %d layers, slots=%d',
                os.environ.get('STEPQUANT_STATE_STORAGE','packed'),
                pages.numel(),dense,len(plans),kwargs['size']+1)


def pool_alloc(original,self,need_size):
    if need_size>len(self.free_slots):
        return None
    slots=self.free_slots[:need_size]
    self.free_slots=self.free_slots[need_size:]
    for conv in self.mamba_cache.conv:
        conv[:,slots]=0
    for p in self.mamba_cache.temporal.pages:
        p.clear(slots)
    return slots


def pool_copy_from(original,self,src_index,dst_index):
    for conv in self.mamba_cache.conv:
        conv[:,dst_index]=conv[:,src_index]
    for p in self.mamba_cache.temporal.pages:
        p.copy(src_index,dst_index)


def pool_get_cpu_copy(original,self,indices):
    torch.cuda.synchronize()
    return ([c[:,indices].cpu() for c in self.mamba_cache.conv],
            [p.data[indices].cpu() for p in self.mamba_cache.temporal.pages])


def pool_load_cpu_copy(original,self,mamba_cache_cpu,indices):
    convs,pages=mamba_cache_cpu
    for dst,src in zip(self.mamba_cache.conv,convs):
        dst[:,indices]=src.to(dst.device)
    for dst,src in zip(self.mamba_cache.temporal.pages,pages):
        dst.wait()
        dst.data[indices]=src.to(dst.device)


def pool_get_contiguous_buf_infos(original,self):
    raise NotImplementedError('STEPQuant v1 does not support disaggregated serving')


def kernel_decode(original,self,q,k,v,a,b,*,A_log,dt_bias,ssm_states,cache_indices,query_start_loc,**kwargs):
    if not isinstance(ssm_states,PackedStatePool):
        return original(self,q,k,v,a,b,A_log=A_log,dt_bias=dt_bias,ssm_states=ssm_states,
                        cache_indices=cache_indices,query_start_loc=query_start_loc,**kwargs)
    out=ssm_states.step(q,k,v,a,b,cache_indices,normalize=True,scale=ssm_states.keys**-.5,
                        a_log=A_log,dt_bias=dt_bias)
    return out.unsqueeze(0)


def kernel_packed_decode(original,self,mixed_qkv,a,b,*,A_log,dt_bias,scale,ssm_states,cache_indices,
                         num_v_heads,head_v_dim,**kwargs):
    if not isinstance(ssm_states,PackedStatePool):
        return original(self,mixed_qkv,a,b,A_log=A_log,dt_bias=dt_bias,scale=scale,ssm_states=ssm_states,
                        cache_indices=cache_indices,num_v_heads=num_v_heads,head_v_dim=head_v_dim,**kwargs)
    vdim=num_v_heads*head_v_dim
    kdim=(mixed_qkv.shape[-1]-vdim)//2
    q,k,v=mixed_qkv.split([kdim,kdim,vdim],-1)
    return ssm_states.step(q,k,v,a,b,cache_indices,normalize=True,scale=scale,a_log=A_log,dt_bias=dt_bias).unsqueeze(0)


def kernel_extend(original,self,q,k,v,g,beta,*,ssm_states,cache_indices,query_start_loc,**kwargs):
    if not isinstance(ssm_states,PackedStatePool):
        return original(self,q,k,v,g,beta,ssm_states=ssm_states,cache_indices=cache_indices,
                        query_start_loc=query_start_loc,**kwargs)
    # Native chunk kernels read and update V-first FP32 active states in-place.
    active=ssm_states.decode(cache_indices).transpose(-1,-2).contiguous()
    indices=torch.arange(cache_indices.numel(),device=cache_indices.device,dtype=cache_indices.dtype)
    output=original(self,q,k,v,g,beta,ssm_states=active,cache_indices=indices,query_start_loc=query_start_loc,**kwargs)
    ssm_states.encode(active.transpose(-1,-2),cache_indices)
    return output


def attention_forward(original,self,*args,**kwargs):
    from stepquant.kernels.writeback import attention_window
    attention_window()
    return original(self,*args,**kwargs)


def capture_graph(original,self,bs,forward,stream_idx=None):
    from stepquant.kernels.writeback import defer_writeback, WritebackGraph
    state_format_name=state_format().name
    six_priority=False
    if state_format_name=='stepquant6':
        from stepquant.kernels.profiles import stepquant6_profile
        from stepquant.kernels.resources import writeback_resources
        pools=[pool for temporal in tuple(_temporal_pools) for pool in temporal.pages
               if pool.writeback_stream is not None]
        for pool in pools:
            sms,chunk=writeback_settings(pool.plan.architecture,bs)
            resources=writeback_resources(pool.device,sms)
            pool.configure_writeback(resources.stream,chunk)
        if pools:
            six_priority=stepquant6_profile(pools[0].plan.architecture,bs).foreground_priority
            _writeback_profiles[str(bs)]=dict(requested_sms=resources.requested_sms,
                                             actual_sms=resources.actual_sms,chunk=chunk)
            log.warning('STEPQuant6 capture: batch=%d background_sms=%d requested_sms=%d chunk=%d',
                        bs,resources.actual_sms,resources.requested_sms,chunk)
    tasks=[]
    @functools.wraps(forward)
    def capture_forward(*args,**kwargs):
        if not torch.cuda.is_current_stream_capturing():
            return forward(*args,**kwargs)
        pools=[pool for temporal in tuple(_temporal_pools) for pool in temporal.pages
               if pool.writeback_stream is not None]
        with defer_writeback(pools,attention_windows=True,joined=True) as captured:
            result=forward(*args,**kwargs)
        tasks.extend(captured)
        return result
    # Balance the background budget with foreground priority for each batch.
    # A large batch needs more capacity to avoid a writeback tail.
    selected=(os.environ.get('STEPQUANT_ARCHITECTURE'),bs,
              os.environ.get('STEPQUANT_WRITEBACK_ACTUAL_SMS'))
    format_name=os.environ.get('STEPQUANT_STATE_FORMAT')
    storage=os.environ.get('STEPQUANT_STATE_STORAGE')
    byte_priority=(storage=='byte'
        and selected in (('kda',64,'24'),('kda',512,'64'),
                         ('gdn',128,'56'),('gdn',256,'56'),('gdn',512,'64'))
        and format_name=='stepquant4')
    packed_priority=(storage=='packed'
        and selected in (('kda',512,'56'),('kda',512,'64'))
        and format_name=='stepquant4')

    priority=(six_priority or byte_priority or packed_priority) and torch.cuda.get_device_capability()==(8,0)
    if priority:
        from stepquant.kernels.priority_graph import PriorityDecodeGraph
        create_graph=self._create_device_graph
        self._create_device_graph=PriorityDecodeGraph
        try:
            graph,output=original(self,bs,capture_forward,stream_idx)
            log.warning('STEPQuant priority decode: batch=%d foreground_nodes=%d writeback_nodes=%d',
                        bs,*graph.priority_counts)
        finally:
            self._create_device_graph=create_graph
    else:
        graph,output=original(self,bs,capture_forward,stream_idx)
    return (WritebackGraph(graph,tasks,joined=True) if tasks else graph),output


def validate_runner(original,self,*args,**kwargs):
    from .plugin import verify_hooks
    verify_hooks()
    bound=inspect.signature(original).bind(self,*args,**kwargs).arguments
    server=bound['server_args']
    if not server.disable_radix_cache:
        raise ValueError('STEPQuant requires --disable-radix-cache until prefix-state tracking is enabled')
    if server.enable_mamba_extra_buffer():
        raise ValueError('STEPQuant requires --mamba-scheduler-strategy no_buffer')
    if getattr(server,'enable_memory_saver',False):
        raise ValueError('STEPQuant does not yet support SGLang memory-saver pause/resume')
    if getattr(server,'pp_size',1)!=1 or getattr(server,'enable_dp_attention',False):
        raise ValueError('STEPQuant currently supports ordinary tensor parallelism only')
    if getattr(server,'speculative_algorithm',None):
        raise ValueError('STEPQuant does not support speculative decoding')
    if getattr(server,'disaggregation_mode','null') not in ('null',None):
        raise ValueError('STEPQuant does not support disaggregated serving')
    if any(getattr(server,key,None) not in (None,'triton') for key in
           ('linear_attn_backend','linear_attn_decode_backend','linear_attn_prefill_backend')):
        raise ValueError('STEPQuant requires --linear-attn-backend triton')
    from stepquant.cli import fingerprint
    if state_format().calibrated and artifact().get('checkpoint_fingerprint')!=fingerprint(server.model_path):
        raise ValueError('STEPQuant plan checkpoint fingerprint mismatch')
    return original(self,*args,**kwargs)


def size_pool(original,self,total_rest_memory):
    from stepquant.kernels.pool import packed_layout
    params=self.mambaish_config.mamba2_cache_params
    heads,values,keys=params.shape.temporal
    plans=local_plans(params.layers,heads,'cpu',keys)
    conv_bytes=sum(__import__('math').prod(shape) for shape in params.shape.conv)*params.dtype.conv.itemsize*len(params.layers)
    spec=state_format()
    compact=conv_bytes+sum((packed_layout(p,values,os.environ.get("STEPQUANT_STATE_STORAGE","packed")) if spec.kind=="stepquant" else format_layout(p,values,spec))[-1] for p in plans)
    dense=params.mamba_cache_per_req
    server=self.server_args
    if server.max_mamba_cache_size is None and not (server.disable_radix_cache and server.max_running_requests is not None):
        state_budget=total_rest_memory*server.mamba_full_memory_ratio/(1+server.mamba_full_memory_ratio)
        count=max(0,int(state_budget*(1<<30)//compact)-1)
        # Mixed precision can give TP ranks different page sizes. Every rank
        # must expose the same number of request slots to the scheduler.
        from sglang.srt.distributed import get_tp_group
        group=get_tp_group()
        if group.world_size>1:
            shared=torch.tensor(count,dtype=torch.int64)
            torch.distributed.all_reduce(shared,op=torch.distributed.ReduceOp.MIN,group=group.cpu_group)
            count=shared.item()
        if count<1:
            raise ValueError('insufficient memory for a STEPQuant request slot')
        server.max_mamba_cache_size=count*(server.dp_size if server.enable_dp_attention else 1)
    result=original(self,total_rest_memory)
    # Upstream deducted the dense estimate; use the actual packed bytes instead.
    # Slot zero is allocated as a sentinel and is accounted for as well.
    return result+(server.max_mamba_cache_size*(dense-compact)-compact)/(1<<30)
