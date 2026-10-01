"""Optional native SGLang 0.5.12 integration tests (run in its own environment)."""
import os
import pytest
import torch
pytest.importorskip('sglang')
pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
from stepquant.quantization import QuantizationPlan
from stepquant.kernels.pool import PackedStatePool


@pytest.mark.parametrize('architecture',['gdn','kda'])
@pytest.mark.parametrize('state_format',['stepquant','sym_int4_g128','sym_int6_g128','sym_int8_g128','dsq_int4','dsq_int6'])
def test_native_prefill_and_fused_decode(architecture,state_format):
    # Exercise real upstream chunk kernels and dispatcher signatures, not stubs.
    if architecture=='gdn':
        from sglang.srt.layers.attention.linear.kernels.gdn_triton import TritonGDNKernel as Kernel
    else:
        from sglang.srt.layers.attention.linear.kernels.kda_triton import TritonKDAKernel as Kernel
    from stepquant.sglang.runtime import kernel_extend,kernel_decode
    torch.manual_seed(54)
    h,k,v=4,128,128
    bits=torch.full((h,k),6,device='cuda')
    plan=QuantizationPlan(architecture,bits,torch.ones_like(bits).float())
    if state_format=='stepquant':
        pool=PackedStatePool(plan,5,v)
    else:
        from stepquant.formats import FormatPlan,get_format,group_fit,dual_fit
        from stepquant.kernels.format_pool import FormatStatePool
        spec=get_format(state_format)
        bits.fill_(spec.bits)
        plan=FormatPlan(architecture,bits,torch.ones_like(bits).float())
        pool=FormatStatePool(plan,5,v,spec)
    slots=torch.tensor([3,1],device='cuda')
    cu=torch.tensor([0,8,24],device='cuda',dtype=torch.int32)
    q,key,val=[torch.randn(1,24,h,k,device='cuda',dtype=torch.bfloat16) for _ in range(3)]
    alog=torch.zeros(h,device='cuda');bias=torch.zeros(h*k if architecture=='kda' else h,device='cuda')
    g=-torch.rand(1,24,h,device='cuda') if architecture=='gdn' else torch.randn(1,24,h,k,device='cuda')
    beta=torch.rand(1,24,h,device='cuda')
    extra={} if architecture=='gdn' else dict(A_log=alog,dt_bias=bias)
    kernel=Kernel()
    dense=torch.zeros(2,h,v,k,device='cuda')
    native_indices=torch.arange(2,device='cuda')
    # Native KDA uses v as its output buffer, so each run needs its own input.
    expected=Kernel.extend(kernel,q,key,val.clone(),g,beta,ssm_states=dense,cache_indices=native_indices,query_start_loc=cu,**extra)
    actual=kernel_extend(Kernel.extend,kernel,q,key,val.clone(),g,beta,ssm_states=pool,cache_indices=slots,query_start_loc=cu,**extra)
    torch.testing.assert_close(actual[0] if isinstance(actual,tuple) else actual,
                               expected[0] if isinstance(expected,tuple) else expected)
    # Prefill must pack the final native state in the correct V/K orientation.
    from stepquant.quantization import fit_state
    decoded=pool.decode(slots)
    x=dense.transpose(-1,-2)
    if state_format=='stepquant':
        expected_state=fit_state(x,plan)[-1]
    elif spec.kind == 'dsq':
        expected_state=dual_fit(x,bits,plan.impact)[-1]
    else:
        expected_state=group_fit(x,spec)[-1]
    torch.testing.assert_close(decoded,expected_state,atol=.005,rtol=.015)
    q,key,val=[torch.randn(1,2,h,k,device='cuda',dtype=torch.bfloat16) for _ in range(3)]
    a=torch.randn(2,h*k,device='cuda') if architecture=='kda' else torch.randn(2,h,device='cuda')
    b=torch.randn(2,h,device='cuda')
    cu=torch.arange(3,device='cuda',dtype=torch.int32)
    dense=decoded.transpose(-1,-2).contiguous()
    expected=Kernel.decode(kernel,q,key,val,a,b,A_log=alog,dt_bias=bias,ssm_states=dense,cache_indices=native_indices,query_start_loc=cu)
    actual=kernel_decode(Kernel.decode,kernel,q,key,val,a,b,A_log=alog,dt_bias=bias,ssm_states=pool,cache_indices=slots,query_start_loc=cu)
    torch.testing.assert_close(actual,expected,atol=.004,rtol=.02)


def test_native_pool_lifecycle(monkeypatch):
    from sglang.srt.configs.mamba_utils import KimiLinearCacheParams,KimiLinearStateShape
    from sglang.srt.mem_cache.memory_pool import MambaPool
    from stepquant.sglang import runtime
    plan=QuantizationPlan('kda',torch.full((4,128),6,device='cuda'),torch.ones(4,128,device='cuda'))
    monkeypatch.setattr(runtime,'local_plans',lambda *args:[plan,plan])
    monkeypatch.setenv('STEPQUANT_ASYNC_WRITEBACK','0')
    params=KimiLinearCacheParams(layers=[0,2],shape=KimiLinearStateShape.create(
        tp_world_size=1,num_heads=4,head_dim=128))
    pool=object.__new__(MambaPool)
    runtime.pool_init(MambaPool.__init__,pool,size=3,spec_state_size=0,
                      cache_params=params,mamba_layer_ids=[0,2],device='cuda')
    slots=runtime.pool_alloc(MambaPool.alloc,pool,3)
    assert runtime.pool_alloc(MambaPool.alloc,pool,1) is None
    page=pool.mamba2_layer_cache(0).temporal
    assert isinstance(page,PackedStatePool)
    page.encode(torch.randn(1,4,128,128,device='cuda'),slots[:1])
    for conv in pool.mamba_cache.conv:
        conv[:,slots[:1]]=1
    runtime.pool_copy_from(MambaPool.copy_from,pool,slots[:1],slots[1:2])
    torch.testing.assert_close(page.decode(slots[:1]),page.decode(slots[1:2]),atol=0,rtol=0)
    saved=runtime.pool_get_cpu_copy(MambaPool.get_cpu_copy,pool,slots[1:2])
    page.clear(slots[1:2])
    runtime.pool_load_cpu_copy(MambaPool.load_cpu_copy,pool,saved,slots[1:2])
    torch.testing.assert_close(page.decode(slots[:1]),page.decode(slots[1:2]),atol=0,rtol=0)
    pool.free(slots[1:2])
    reused=runtime.pool_alloc(MambaPool.alloc,pool,1)
    assert reused.item()==slots[1].item()
    assert page.decode(reused).count_nonzero()==0
    assert all(conv[:,reused].count_nonzero()==0 for conv in pool.mamba_cache.conv)
    assert pool.mamba_cache.mem_usage_bytes()<params.mamba_cache_per_req*4
