"""Actual-allocation admission for SGLang 0.5.12 (fresh requests only)."""
import copy
import logging
import os
from functools import lru_cache


@lru_cache(maxsize=1)
def runtime_identity(model_path):
    from stepquant.reproducibility import source_identity, checkpoint_identity
    return dict(model_path=model_path, checkpoint=checkpoint_identity(model_path),
                source_sha256=source_identity(), settings={k: os.environ.get(k) for k in (
                    'STEPQUANT_STATE_STORAGE','STEPQUANT_KERNEL_BACKEND','STEPQUANT_FIT_MODE',
                    'STEPQUANT_ASYNC_WRITEBACK','STEPQUANT_WRITEBACK_CHUNK','STEPQUANT_WRITEBACK_ACTUAL_SMS')})

log = logging.getLogger(__name__)
METRICS = dict(admitted=0, paused=0, occupancy=0., peak_occupancy=0.)


def occupancy(adder):
    allocator = adder.token_to_kv_pool_allocator
    # Include prefills accepted in this scheduler pass but not allocated yet.
    return min(1., (allocator.size - allocator.available_size() + adder.cur_rem_token_offset) / allocator.size)


def running_offset(original, self, req):
    return self.page_size  # one decode page, never the requested completion budget


def update_budget(original, self, prefix_len, extend_input_len, max_new_tokens):
    return original(self, prefix_len, extend_input_len, min(max_new_tokens, 1))


def add_one(original, self, req, *args, **kwargs):
    from sglang.srt.managers.schedule_policy import AddReqResult
    used = occupancy(self)
    METRICS['occupancy'] = used
    METRICS['peak_occupancy'] = max(used, METRICS['peak_occupancy'])
    if used >= float(os.environ.get('STEPQUANT_KV_WATERMARK', '.6')):
        METRICS['paused'] += 1
        # NO_TOKEN sets the scheduler's sticky batch_is_full flag. OTHER lets
        # admission resume immediately when completed requests release pages.
        return AddReqResult.OTHER
    params = req.sampling_params
    temporary = copy.copy(params)
    temporary.max_new_tokens = len(req.output_ids) + 1
    temporary.ignore_eos = False  # avoid upstream full-completion reservation
    req.sampling_params = temporary
    before = len(self.can_run_list)
    try:
        return original(self, req, *args, **kwargs)
    finally:
        req.sampling_params = params
        METRICS['admitted'] += len(self.can_run_list) - before


def internal_state(original, self, request):
    result = original(self, request)
    allocator = self.token_to_kv_pool_allocator
    import hashlib
    from pathlib import Path
    from importlib.metadata import version
    result.internal_state['stepquant_versions']={k:version(k) for k in ('torch','sglang','triton','transformers')}
    result.internal_state['stepquant_state_format']=os.environ.get('STEPQUANT_STATE_FORMAT','stepquant')
    result.internal_state['stepquant_runtime_identity']=dict(runtime_identity(self.server_args.model_path),
        server_args={k:getattr(self.server_args,k,None) for k in (
            'dtype','quantization','tp_size','context_length','attention_backend','linear_attn_backend',
            'reasoning_parser','disable_cuda_graph','cuda_graph_bs','cuda_graph_max_bs','max_running_requests',
            'mem_fraction_static','chunked_prefill_size','max_total_tokens','kv_cache_dtype',
            'disable_radix_cache','mamba_scheduler_strategy','mamba_ssm_dtype','random_seed')})
    plan=os.environ.get('STEPQUANT_PLAN')
    result.internal_state['stepquant_plan_sha256']=hashlib.sha256(Path(plan).read_bytes()).hexdigest() if plan else None
    result.internal_state['stepquant_admission'] = dict(
        **METRICS, kv_used_tokens=allocator.size-allocator.available_size(),
        kv_capacity_tokens=allocator.size, running_requests=len(self.running_batch.reqs),
        waiting_requests=len(self.waiting_queue), watermark=float(os.environ.get('STEPQUANT_KV_WATERMARK','.6')))
    return result


def run_batch(original, self, batch, *args, **kwargs):
    count=len(batch.reqs)
    METRICS['peak_running_requests']=max(METRICS.get('peak_running_requests',0),count)
    allocator=self.token_to_kv_pool_allocator
    used=(allocator.size-allocator.available_size())/allocator.size
    METRICS['peak_actual_occupancy']=max(METRICS.get('peak_actual_occupancy',0.),used)
    return original(self,batch,*args,**kwargs)


def prefill_pass(original, self, *args, **kwargs):
    chunk=self.chunked_req
    allocator=self.token_to_kv_pool_allocator
    used=(allocator.size-allocator.available_size())/allocator.size
    if chunk is None and self.waiting_queue and used>=float(os.environ.get('STEPQUANT_KV_WATERMARK','.6')):
        # Stop before req.init_next_round_input allocates/zeros a recurrent
        # slot. Repeated failed admissions must not launch cache-clear kernels
        # on every decode step while the water gate is closed.
        METRICS['paused']+=1
        METRICS['occupancy']=used
        METRICS['peak_occupancy']=max(METRICS['peak_occupancy'],used)
        return None
    self._stepquant_chunk_slot_credit=int(chunk is not None and chunk.req_pool_idx is not None)
    try:
        return original(self,*args,**kwargs)
    finally:
        self._stepquant_chunk_slot_credit=0


def allocatable(original, self, running_bs):
    # A chunk continuation already owns a request slot but is also included in
    # adder.can_run_list. Upstream's comparison otherwise counts it twice and
    # latches batch_is_full one request below the configured capacity.
    from sglang.srt.server_args import get_global_server_args
    return min(get_global_server_args().pp_max_micro_batch_size-running_bs,
               self.req_to_token_pool.available_size()+getattr(self,'_stepquant_chunk_slot_credit',0))


def retract(original, self, *args, **kwargs):
    result=original(self,*args,**kwargs)
    METRICS['retracted_requests']=METRICS.get('retracted_requests',0)+len(result[0])
    METRICS['aborted_requests']=METRICS.get('aborted_requests',0)+len(result[2])
    return result
