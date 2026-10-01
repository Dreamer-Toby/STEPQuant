"""Packing alignment boundaries for optimized kernels."""
import pytest
import torch
from stepquant.kernels.pool import PackedStatePool
from stepquant.quantization import QuantizationPlan,fit_state

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


@pytest.mark.parametrize('values',[4,32])
@pytest.mark.parametrize('asynchronous',[False,True])
def test_mixed_pages_with_only_half_alignment(values,asynchronous):
    bits=torch.tensor([[2,4,8,16]],device='cuda')
    plan=QuantizationPlan('kda',bits,torch.ones_like(bits).float())
    pool=PackedStatePool(plan,4,values,writeback_stream=torch.cuda.Stream() if asynchronous else None)
    assert pool.page_bytes%4==2
    slots=torch.tensor([1,2,3],device='cuda')
    torch.manual_seed(5)
    x=torch.randn(3,1,4,values,device='cuda')
    pool.encode(x,slots)
    torch.testing.assert_close(pool.decode(slots),fit_state(x,plan)[-1],atol=2e-4,rtol=2e-3)
    from stepquant.core import delta_step
    before=pool.decode(slots)
    q=torch.randn(3,1,4,device='cuda')
    k=torch.nn.functional.normalize(torch.randn_like(q),dim=-1)
    v=torch.randn(3,1,values,device='cuda')
    gates=-torch.rand_like(q);beta=torch.rand(3,1,device='cuda')
    output,updated=delta_step(before,q,k,v,gates,beta)
    torch.testing.assert_close(pool.step(q,k,v,gates,beta,slots),output,atol=5e-5,rtol=1e-4)
    torch.testing.assert_close(pool.decode(slots),fit_state(updated,plan)[-1],atol=2e-4,rtol=2e-3)
    pool.clear(slots)
    assert not pool.decode(slots).count_nonzero()


@pytest.mark.parametrize('values',[8,16,32])
def test_gdn_word_loads_small_dimensions(values):
    # V=8 takes the byte-addressed fallback; V>=16 takes aligned word loads.
    torch.manual_seed(224)
    bits=torch.tensor([2,4,6,8,16],device='cuda')[:,None].expand(5,4).clone()
    plan=QuantizationPlan('gdn',bits,torch.rand(5,4,device='cuda')+.5,min(values,32))
    pool=PackedStatePool(plan,4,values,writeback_stream=torch.cuda.Stream())
    assert pool.page_bytes%4==0
    slots=torch.tensor([3,1],device='cuda')
    pool.encode(torch.randn(2,5,4,values,device='cuda'),slots)
    q=torch.randn(2,5,4,device='cuda');k=torch.nn.functional.normalize(torch.randn_like(q),dim=-1)
    v=torch.randn(2,5,values,device='cuda');g=-torch.rand(2,5,device='cuda');beta=torch.rand_like(g)
    from stepquant.core import delta_step
    expected,state=delta_step(pool.decode(slots),q,k,v,g,beta)
    torch.testing.assert_close(pool.step(q,k,v,g,beta,slots),expected,atol=5e-5,rtol=1e-4)
    torch.testing.assert_close(pool.decode(slots),fit_state(state,plan)[-1],atol=3e-4,rtol=2e-3)
