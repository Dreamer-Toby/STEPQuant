import pytest
import torch
triton=pytest.importorskip('triton')
from packaging.version import Version
if Version(triton.__version__) < Version('3.3'):
    pytest.skip('fused tests require Triton >= 3.3',allow_module_level=True)
pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
from stepquant.quantization import QuantizationPlan,fit_state
from stepquant.core import delta_step
from stepquant.kernels.pool import PackedStatePool


@pytest.mark.parametrize('architecture',['gdn','kda'])
@pytest.mark.parametrize('size,group',[(32,32),(128,32),(128,128)])
@pytest.mark.parametrize('asynchronous,storage',[(False,'packed'),(True,'packed'),(True,'byte')])
def test_codec_and_fused_decode(architecture,size,group,asynchronous,storage):
    torch.manual_seed(31)
    bits=torch.tensor([2,4,6,8,16],device='cuda')[:,None].expand(5,size).clone()
    if architecture=='kda':
        bits=bits.roll(1,0)
        bits[:,::3]=16
        bits[:,1::4]=2
    w=torch.rand(5,size,device='cuda')+.5
    plan=QuantizationPlan(architecture,bits,w,value_group_size=group)
    pool=PackedStatePool(plan,6,size,writeback_stream=torch.cuda.Stream() if asynchronous else None,storage=storage)
    slots=torch.tensor([3,1],device='cuda')
    x=torch.randn(2,5,size,size,device='cuda')
    pool.encode(x,slots)
    actual=pool.decode(slots)
    expected=fit_state(x,plan)[-1]
    # Different reduction trees can move near-midpoint entries across a code boundary.
    torch.testing.assert_close(actual,expected,atol=3e-4,rtol=2e-3)
    for _ in range(3):
        q=torch.randn(2,5,size,device='cuda')
        k=torch.nn.functional.normalize(torch.randn_like(q),dim=-1)
        v=torch.randn_like(q)
        g=-torch.rand(2,5,size,device='cuda') if architecture=='kda' else -torch.rand(2,5,device='cuda')
        beta=torch.rand(2,5,device='cuda')
        output,updated=delta_step(actual,q,k,v,g,beta)
        result=pool.step(q,k,v,g,beta,slots)
        torch.testing.assert_close(result,output,atol=5e-5,rtol=1e-4)
        actual=pool.decode(slots)
        reference=fit_state(updated,plan)[-1]
        for bit in (2,4,6,8):
            mask=bits==bit
            if mask.any():
                ref_error=(reference[:,mask]-updated[:,mask]).square().mean()
                discrepancy=(actual[:,mask]-reference[:,mask]).square().mean()
                assert discrepancy < .01*ref_error+1e-10
                assert (actual[:,mask]-updated[:,mask]).square().mean() < 1.02*ref_error+1e-10
    pool.copy(slots[:1],slots[1:])
    torch.testing.assert_close(pool.decode(slots[:1]),pool.decode(slots[1:]),atol=0,rtol=0)
    pool.clear(slots)
    assert pool.decode(slots).count_nonzero()==0
    assert pool.nbytes < 6*5*size*size*4


@pytest.mark.parametrize('architecture',['gdn','kda'])
@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
@pytest.mark.parametrize('batch',[3,16])
@pytest.mark.parametrize('asynchronous,storage',[(False,'packed'),(True,'packed'),(True,'byte')])
def test_raw_gates_gqa_strides_padding_and_cuda_graph(architecture,dtype,batch,asynchronous,storage):
    torch.manual_seed(8)
    h,hq,k,v=4,2,32,32
    bits=torch.full((h,k),6,device='cuda')
    cls=PackedStatePool
    pool=cls(QuantizationPlan(architecture,bits,torch.ones_like(bits).float()),batch+1,v,
             writeback_stream=torch.cuda.Stream() if asynchronous else None,storage=storage)
    slots=torch.arange(batch-1,-1,-1,device='cuda')
    # Views into post-convolution QKV: non-contiguous token stride, no QKV copies.
    mixed=torch.randn(batch,2*hq*k+h*v,device='cuda',dtype=dtype)
    q,key,val=mixed.split([hq*k,hq*k,h*v],-1)
    a=torch.randn(batch,h,k,device='cuda') if architecture=='kda' else torch.randn(batch,h,device='cuda')
    bias=torch.randn(h*k if architecture=='kda' else h,device='cuda')
    a_log=torch.randn(h,device='cuda')
    beta=torch.randn(batch,h,device='cuda')
    def call():
        return pool.step(q,key,val,a,beta,slots,normalize=True,scale=k**-.5,a_log=a_log,dt_bias=bias)
    call()
    pool.wait()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out=call()
        pool.wait()
    for _ in range(3):
        state=pool.decode(slots)
        q2=q.reshape(batch,hq,k).float();q2=q2*torch.rsqrt(q2.square().sum(-1,keepdim=True)+1e-6)
        k2=key.reshape(batch,hq,k).float();k2=k2*torch.rsqrt(k2.square().sum(-1,keepdim=True)+1e-6)
        q2=q2.repeat_interleave(h//hq,1)*k**-.5
        k2=k2.repeat_interleave(h//hq,1)
        g=-a_log.exp()[None,:,None]*torch.nn.functional.softplus(a+bias.reshape(h,k)) if architecture=='kda' else -a_log.exp()*torch.nn.functional.softplus(a+bias)
        expected,_=delta_step(state,q2,k2,val.reshape(batch,h,v),g,beta.sigmoid())
        graph.replay()
        torch.testing.assert_close(out[:-1],expected[:-1].to(dtype),atol=1e-5 if dtype==torch.float32 else .002,rtol=1e-4 if dtype==torch.float32 else .008)
        assert out[-1].count_nonzero()==0
        assert pool.data[0].count_nonzero()==0


def test_async_writeback_metadata_reuse_and_slot_copy():
    torch.manual_seed(72)
    plan=QuantizationPlan('kda',torch.full((4,32),6,device='cuda'),torch.ones(4,32,device='cuda'))
    sync=PackedStatePool(plan,4,32)
    async_pool=PackedStatePool(plan,4,32,writeback_stream=torch.cuda.Stream())
    slots=torch.tensor([1,2],device='cuda')
    q=torch.randn(2,4,32,device='cuda');k=torch.nn.functional.normalize(torch.randn_like(q),dim=-1)
    v=torch.randn_like(q);g=-torch.rand_like(q);beta=torch.rand(2,4,device='cuda')
    for _ in range(4):
        expected=sync.step(q,k,v,g,beta,slots)
        out=async_pool.step(q,k,v,g,beta,slots)
        # Simulate in-place reuse of scheduler metadata while the side stream runs.
        slots.fill_(3)
        slots.copy_(torch.tensor([1,2],device='cuda'))
        torch.testing.assert_close(out,expected)
        async_pool.copy(slots[:1],slots[1:]);sync.copy(slots[:1],slots[1:])
        torch.testing.assert_close(async_pool.decode(slots),sync.decode(slots),atol=0,rtol=0)
    async_pool.clear(slots)
    assert async_pool.decode(slots).count_nonzero()==0


@pytest.mark.parametrize('architecture',['gdn','kda'])
def test_async_multilayer_cuda_graph_replay_and_slot_reuse(architecture):
    torch.manual_seed(51)
    bits=torch.tensor([2,4,6,16],device='cuda')[:,None].expand(4,32).clone()
    if architecture=='kda':bits[:,::3]=16
    plan=QuantizationPlan(architecture,bits,torch.rand(4,32,device='cuda')+.5)
    stream=torch.cuda.Stream()
    cls=PackedStatePool
    pools=[cls(plan,6,32,writeback_stream=stream) for _ in range(3)]
    refs=[cls(plan,6,32) for _ in pools]
    slots=torch.tensor([3,1,0],device='cuda')
    q=torch.randn(3,4,32,device='cuda');k=torch.nn.functional.normalize(torch.randn_like(q),dim=-1)
    v=torch.randn_like(q);g=-torch.rand_like(q) if architecture=='kda' else -torch.rand(3,4,device='cuda')
    beta=torch.rand(3,4,device='cuda')
    def forward():
        outs=[pool.step(q,k,v,g,beta,slots) for pool in pools]
        for pool in pools:pool.wait()
        return outs
    for _ in range(3):forward()
    for pool in pools:pool.clear(slots)
    torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):outputs=forward()
    for step in range(8):
        if step==4:
            src=torch.tensor([3],device='cuda');dst=torch.tensor([4],device='cuda')
            for pool in pools+refs:pool.copy(src,dst)
            slots.copy_(torch.tensor([4,1,0],device='cuda'))
        graph.replay()
        for pool,ref,output in zip(pools,refs,outputs):
            expected=ref.step(q,k,v,g,beta,slots)
            torch.testing.assert_close(output,expected,atol=5e-5,rtol=1e-4)
            torch.testing.assert_close(pool.decode(slots),ref.decode(slots),atol=3e-4,rtol=2e-3)
            assert pool.data[0].count_nonzero()==0


@pytest.mark.parametrize('group',[32,128])
def test_full_head_gdn_fit(group):
    torch.manual_seed(87)
    bits=torch.tensor([2,4,6,8,16],device='cuda')[:,None].expand(5,128).clone()
    plan=QuantizationPlan('gdn',bits,torch.rand(5,128,device='cuda')+.5,value_group_size=group)
    pool=PackedStatePool(plan,17)
    slots=torch.arange(16,0,-1,device='cuda');slots[-1]=0
    state=torch.randn(16,5,128,128,device='cuda')*.1;state[-1]=0
    pool.encode(state,slots)
    for _ in range(4):
        state=pool.decode(slots)
        q=torch.randn(16,5,128,device='cuda');k=torch.nn.functional.normalize(torch.randn_like(q),dim=-1)
        v=torch.randn_like(q);g=-torch.rand(16,5,device='cuda');beta=torch.rand_like(g)
        output,updated=delta_step(state,q,k,v,g,beta)
        actual=pool.step(q,k,v,g,beta,slots)
        torch.testing.assert_close(actual[:-1],output[:-1],atol=5e-5,rtol=1e-4)
        expected=fit_state(updated,plan)[-1]
        decoded=pool.decode(slots)
        for head in range(4):
            error=(expected[:-1,head]-updated[:-1,head]).square().mean()
            assert (decoded[:-1,head]-expected[:-1,head]).square().mean()<.01*error+1e-10
            assert (decoded[:-1,head]-updated[:-1,head]).square().mean()<1.02*error+1e-10
        torch.testing.assert_close(decoded[:-1,4],expected[:-1,4],atol=1e-5,rtol=1e-3)
        assert decoded[-1].count_nonzero()==0
        assert actual[-1].count_nonzero()==0
        assert pool.data[0].count_nonzero()==0


@pytest.mark.parametrize('group',[32,128])
@pytest.mark.parametrize('storage,batch',[('byte',128),('packed',64),('packed',128)])
def test_segmented_gdn_reader_against_independent_reference(group,storage,batch):
    torch.manual_seed(31)
    size=128
    bits=torch.tensor([2,4,6,8,16],device='cuda')[:,None].expand(5,size).clone()
    plan=QuantizationPlan('gdn',bits,torch.rand(5,size,device='cuda')+.5,value_group_size=group)
    pool=PackedStatePool(plan,batch+1,size,writeback_stream=torch.cuda.Stream(),storage=storage)
    slots=torch.arange(batch,0,-1,device='cuda')
    pool.encode(torch.randn(batch,5,size,size,device='cuda'),slots)
    # Start from the actual encoded page. Prefill rounding is covered separately;
    # this checks readout and the statistics consumed by the subsequent fit.
    for _ in range(3):
        state=pool.decode(slots)
        q=torch.randn(batch,5,size,device='cuda')
        k=torch.nn.functional.normalize(torch.randn_like(q),dim=-1)
        v=torch.randn_like(q);g=-torch.rand(batch,5,device='cuda');beta=torch.rand_like(g)
        expected,updated=delta_step(state,q,k,v,g,beta)
        torch.testing.assert_close(pool.step(q,k,v,g,beta,slots),expected,atol=5e-5,rtol=1e-4)
        actual=pool.decode(slots)
        reference=fit_state(updated,plan)[-1]
        for bit in (2,4,6,8):
            mask=bits==bit
            ref_error=(reference[:,mask]-updated[:,mask]).square().mean()
            assert (actual[:,mask]-reference[:,mask]).square().mean() < .01*ref_error+1e-10
            assert (actual[:,mask]-updated[:,mask]).square().mean() < 1.02*ref_error+1e-10
