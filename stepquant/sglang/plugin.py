"""SGLang general-plugin entrypoint. Uses official per-process plugin hooks."""
import os


def verify_hooks():
    from sglang.srt.plugins.hook_registry import HookRegistry
    required = [target for target, hooks in HookRegistry._hooks.items()
                if any(getattr(fn, '__module__', '') in ('stepquant.sglang.runtime','stepquant.sglang.admission','stepquant.sglang.logits','stepquant.sglang.prefill_reduce')
                       for _, fn, _ in hooks)]
    missing = set(required) - HookRegistry._patched
    if not required or missing:
        raise RuntimeError(f'STEPQuant plugin hooks incomplete: {sorted(missing)}; refusing FP32 fallback')


def register():
    if not (os.environ.get('STEPQUANT_PLAN') or os.environ.get('STEPQUANT_STATE_FORMAT')):
        return
    from importlib.metadata import version
    if version('sglang') != '0.5.12':
        raise RuntimeError('STEPQuant integration is pinned to sglang==0.5.12')
    from sglang.srt.plugins.hook_registry import HookRegistry,HookType
    from . import runtime
    base='sglang.srt.'
    def hook(path,fn):
        HookRegistry.register(base+path,fn,HookType.AROUND)
    # A common B512 memory-copy optimization, including the native baseline.
    if os.environ.get('STEPQUANT_ARCHITECTURE')=='gdn':
        from .logits import get_logits
        hook('layers.logits_processor.LogitsProcessor._get_logits',get_logits)
    if os.environ.get('STEPQUANT_ARCHITECTURE')=='kda':
        from . import prefill_reduce
        hook('model_executor.model_runner.ModelRunner.forward',prefill_reduce.forward)
        hook('distributed.parallel_state.GroupCoordinator.all_reduce',prefill_reduce.all_reduce)
    native=os.environ.get('STEPQUANT_KERNEL_BACKEND')=='native'
    if native and os.environ.get('STEPQUANT_STATE_FORMAT')!='fp32':
        raise ValueError('native benchmark backend requires FP32')
    if not native:
        pool='mem_cache.memory_pool.MambaPool.'
        for method in ('__init__','alloc','copy_from','get_cpu_copy','load_cpu_copy','get_contiguous_buf_infos'):
            hook(pool+method,getattr(runtime,'pool_'+method.strip('_')))
        for cls,architecture in [('gdn_triton.TritonGDNKernel','gdn'),('kda_triton.TritonKDAKernel','kda')]:
            prefix='layers.attention.linear.kernels.'+cls+'.'
            hook(prefix+'decode',runtime.kernel_decode)
            hook(prefix+'extend',runtime.kernel_extend)
            if architecture=='gdn':
                hook(prefix+'packed_decode',runtime.kernel_packed_decode)
    hook('model_executor.model_runner.ModelRunner.__init__',runtime.validate_runner)
    if not native:
        hook('model_executor.model_runner_kv_cache_mixin.ModelRunnerKVCacheMixin.handle_max_mamba_cache',runtime.size_pool)

    if os.environ.get('STEPQUANT_ASYNC_WRITEBACK') == '1':
        hook('model_executor.cuda_graph_runner.CudaGraphRunner.capture_one_batch_size',runtime.capture_graph)
        if os.environ.get('STEPQUANT_ARCHITECTURE') == 'kda':
            hook('models.kimi_linear.KimiDecoderLayer.forward',runtime.attention_forward)
        else:
            for cls in ('Qwen3_5LinearDecoderLayer','Qwen3_5AttentionDecoderLayer'):
                hook('models.qwen3_5.'+cls+'.forward',runtime.attention_forward)

    if os.environ.get('STEPQUANT_TIMING_DIR'):
        from . import timing
        hook('model_executor.model_runner.ModelRunner.forward',timing.forward)
        hook('managers.scheduler.Scheduler.get_internal_state',timing.internal_state)

    if os.environ.get('STEPQUANT_DYNAMIC_ADMISSION') == '1':
        from . import admission
        prefix='managers.schedule_policy.PrefillAdder.'
        hook(prefix+'_get_running_request_total_token_offset',admission.running_offset)
        hook(prefix+'_update_prefill_budget',admission.update_budget)
        hook(prefix+'add_one_req',admission.add_one)
        hook('managers.scheduler.Scheduler.get_internal_state',admission.internal_state)
        hook('managers.scheduler.Scheduler.run_batch',admission.run_batch)
        hook('managers.scheduler.Scheduler._get_new_batch_prefill_raw',admission.prefill_pass)
        hook('managers.scheduler.Scheduler.get_num_allocatable_reqs',admission.allocatable)
        hook('managers.schedule_batch.ScheduleBatch.retract_decode',admission.retract)
