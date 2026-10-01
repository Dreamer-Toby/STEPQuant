import pytest
import torch
from stepquant.formats import FORMAT_NAMES, get_format, group_fit, dual_fit
from stepquant.formats import FormatPlan as QuantizationPlan
from stepquant.core import delta_step

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


@pytest.mark.parametrize('name', [n for n in FORMAT_NAMES if n not in ('stepquant4','stepquant6')])
@pytest.mark.parametrize('architecture', ['gdn', 'kda'])
def test_fused_format(name, architecture):
    from stepquant.kernels.format_pool import FormatStatePool
    torch.manual_seed(42)
    spec=get_format(name)
    h,k,v=2,32,128
    bits=torch.full((h,k),spec.bits,device='cuda',dtype=torch.int32)
    impact=torch.ones((h,k),device='cuda')
    plan=QuantizationPlan(architecture,bits,impact)
    pool=FormatStatePool(plan,4,v,spec)
    slots=torch.tensor([1,2],device='cuda')
    x=torch.randn(2,h,k,v,device='cuda')*.3
    def oracle(x):
        if spec.kind == 'dsq':
            return dual_fit(x,bits,impact)[-1]
        recon=group_fit(x,spec)[-1]
        return recon
    pool.encode(x,slots)
    torch.testing.assert_close(pool.decode(slots),oracle(x),atol=4e-3,rtol=3e-3)
    for _ in range(3):
        initial=pool.decode(slots)
        q=torch.nn.functional.normalize(torch.randn(2,h,k,device='cuda'),dim=-1)
        key=torch.nn.functional.normalize(torch.randn_like(q),dim=-1)
        val=torch.randn(2,h,v,device='cuda')
        g=-torch.rand((2,h,k) if architecture=='kda' else (2,h),device='cuda')
        beta=torch.rand(2,h,device='cuda')
        expected,updated=delta_step(initial,q,key,val,g,beta)
        output=pool.step(q,key,val,g,beta,slots)
        torch.testing.assert_close(output,expected,atol=1e-5,rtol=1e-4)
        actual=pool.decode(slots)
        torch.testing.assert_close(actual,oracle(updated),atol=5e-3,rtol=5e-3)
    pool.copy(slots[:1],slots[1:])
    torch.testing.assert_close(pool.decode(slots)[0],pool.decode(slots)[1],atol=0,rtol=0)
    pool.clear(slots)
    assert pool.decode(slots).count_nonzero()==0
    padded=torch.tensor([0,-1],device='cuda')
    saved=pool.data.clone()
    assert pool.step(q,key,val,g,beta,padded).count_nonzero()==0
    assert torch.equal(pool.data,saved)


@pytest.mark.parametrize('name',[n for n in FORMAT_NAMES if n not in ('stepquant4','stepquant6')])
@pytest.mark.parametrize('architecture',['gdn','kda'])
@pytest.mark.parametrize('batch',[2,32,64,128,256,512])
def test_format_cuda_graph(name,architecture,batch):
    from stepquant.kernels.format_pool import FormatStatePool
    torch.manual_seed(11)
    spec=get_format(name)
    bits=torch.full((2,128),spec.bits,device='cuda')
    plan=QuantizationPlan(architecture,bits,torch.ones_like(bits).float())
    pool=FormatStatePool(plan,batch+1,128,spec)
    slots=torch.arange(1,batch+1,device='cuda')
    q,key=[torch.nn.functional.normalize(torch.randn(batch,2,128,device='cuda'),dim=-1) for _ in range(2)]
    val=torch.randn(batch,2,128,device='cuda');g=-torch.rand_like(q) if architecture=='kda' else -torch.rand(batch,2,device='cuda');beta=torch.rand(batch,2,device='cuda')
    for _ in range(3):pool.step(q,key,val,g,beta,slots)
    torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out=pool.step(q,key,val,g,beta,slots)
    pool.clear(slots)
    graph.replay()
    actual=out.clone();packed=pool.data.clone()
    pool.clear(slots)
    expected=pool.step(q,key,val,g,beta,slots)
    torch.testing.assert_close(actual,expected,atol=0,rtol=0)
    assert torch.equal(pool.data,packed)
