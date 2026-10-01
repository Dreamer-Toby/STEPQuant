"""Cross-replay state dependencies, including graph-size switches and slot reuse."""
import pytest
import torch
pytest.importorskip('triton')
pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
from stepquant.quantization import QuantizationPlan
from stepquant.kernels.pool import PackedStatePool
from stepquant.kernels.writeback import defer_writeback, WritebackGraph, attention_window


@pytest.mark.parametrize('architecture',['gdn','kda'])
@pytest.mark.parametrize('size',[32,128])
@pytest.mark.parametrize('attention_windows',[False,True])
@pytest.mark.parametrize('storage',['packed','byte'])
@pytest.mark.parametrize('large_batch',[16,32,64,128,256,512])
def test_split_graph_queued_replays_and_batch_switch(architecture,size,attention_windows,storage,large_batch,joined=False):
    options={}
    torch.manual_seed(92)
    bits=torch.tensor([2,4,6,16],device='cuda')[:,None].expand(4,size).clone()
    if architecture=='kda':bits[:,::5]=16
    plan=QuantizationPlan(architecture,bits,torch.rand(4,size,device='cuda')+.5)
    from stepquant.kernels.resources import writeback_resources
    resources=writeback_resources('cuda:0',32)
    assert 0 < resources.actual_sms <= resources.total_sms
    stream=resources.stream
    capacity=max(20,large_batch+1)
    workspace={}
    pool_class=PackedStatePool
    pools=[pool_class(plan,capacity,size,writeback_stream=stream,writeback_workspace=workspace,writeback_chunk=5,storage=storage) for _ in range(3)]
    assert len({pool.coefficients.data_ptr() for pool in pools})==1
    # Compare graph replay against eager execution of the same asynchronous
    # arithmetic; independent numerical-oracle checks live in test_fused_cuda.
    ref_workspace={}
    ref_stream=torch.cuda.Stream()
    refs=[pool_class(plan,capacity,size,writeback_stream=ref_stream,
                          writeback_workspace=ref_workspace,storage=storage) for _ in pools]
    graphs={};inputs={};outputs={}
    shared=torch.cuda.graph_pool_handle()
    for batch in (large_batch,3):
        slots=torch.arange(batch,0,-1,device='cuda');slots[-1]=0
        q=torch.randn(batch,4,size,device='cuda')
        k=torch.nn.functional.normalize(torch.randn_like(q),dim=-1)
        v=torch.randn_like(q)
        g=-torch.rand_like(q) if architecture=='kda' else -torch.rand(batch,4,device='cuda')
        beta=torch.rand(batch,4,device='cuda')
        inputs[batch]=(q,k,v,g,beta,slots)
        for _ in range(2):
            for pool in pools:pool.step(*inputs[batch],**options)
        torch.cuda.synchronize()
        fg=torch.cuda.CUDAGraph()
        with torch.cuda.graph(fg,pool=shared):
            with defer_writeback(pools,attention_windows=attention_windows,joined=joined and batch==large_batch) as tasks:
                outputs[batch]=[]
                for pool in pools:
                    attention_window()
                    outputs[batch].append(pool.step(*inputs[batch],**options))
                    attention_window()  # An intervening full-attention layer.
        graphs[batch]=WritebackGraph(fg,tasks,joined=joined and batch==large_batch)
    all_slots=torch.arange(capacity,device='cuda')
    for pool in pools:pool.clear(all_slots)
    # Queue different graphs without reading back or synchronizing each token.
    sequence=[large_batch,3,large_batch,large_batch,3,3]*3
    actual=[]
    for index,batch in enumerate(sequence):
        if index==7:
            for pool in pools:
                pool.copy(all_slots[1:2],all_slots[2:3])
                pool.clear(all_slots[3:4])
        graphs[batch].replay()
        actual.append([x.clone() for x in outputs[batch]])
    # Exercise graph -> eager -> graph handoff of the completion event.
    actual_eager=[pool.step(*inputs[3],**options).clone() for pool in pools]
    graphs[large_batch].replay()
    actual_last=[x.clone() for x in outputs[large_batch]]
    for index,batch in enumerate(sequence):
        if index==7:
            for pool in refs:
                pool.copy(all_slots[1:2],all_slots[2:3])
                pool.clear(all_slots[3:4])
        for ref,result in zip(refs,actual[index]):
            expected=ref.step(*inputs[batch],**options)
            torch.testing.assert_close(result,expected,atol=5e-5,rtol=1e-4)
    for ref,result in zip(refs,actual_eager):
        torch.testing.assert_close(result,ref.step(*inputs[3],**options),atol=5e-5,rtol=1e-4)
    for pool,ref,result in zip(pools,refs,actual_last):
        torch.testing.assert_close(result,ref.step(*inputs[large_batch],**options),atol=5e-5,rtol=1e-4)
        torch.testing.assert_close(pool.decode(all_slots),ref.decode(all_slots),atol=3e-4,rtol=2e-3)
        assert pool.data[0].count_nonzero()==0


def test_forward_completes_before_deferred_writeback(monkeypatch):
    plan=QuantizationPlan('gdn',torch.full((1,32),4,device='cuda'),torch.ones(1,32,device='cuda'))
    pool=PackedStatePool(plan,2,32,writeback_stream=torch.cuda.Stream())
    slots=torch.ones(1,device='cuda',dtype=torch.long)
    q=torch.randn(1,1,32,device='cuda');k=torch.nn.functional.normalize(q,dim=-1)
    args=(q,k,q,-torch.ones(1,1,device='cuda'),torch.ones(1,1,device='cuda'),slots)
    pool.step(*args);torch.cuda.synchronize()
    fg=torch.cuda.CUDAGraph()
    with torch.cuda.graph(fg):
        with defer_writeback([pool]) as tasks:
            pool.step(*args)
    original=pool._finish_update
    def delayed(*args,**kwargs):
        torch.cuda._sleep(50_000_000)
        return original(*args,**kwargs)
    monkeypatch.setattr(pool,'_finish_update',delayed)
    graph=WritebackGraph(fg,tasks)
    graph.replay()
    forward_done=torch.cuda.Event()
    forward_done.record();forward_done.synchronize()
    assert not pool.graph_ready.query(), 'forward waited for unrelated writeback tail'
    pool.wait();torch.cuda.synchronize()


def test_writeback_dependency_is_kept_for_multiple_consumer_streams(monkeypatch):
    plan=QuantizationPlan('kda',torch.full((1,32),4,device='cuda'),torch.ones(1,32,device='cuda'))
    pool=PackedStatePool(plan,2,32,writeback_stream=torch.cuda.Stream())
    slots=torch.ones(1,device='cuda',dtype=torch.long)
    q=torch.randn(1,1,32,device='cuda');k=torch.nn.functional.normalize(torch.randn_like(q),dim=-1)
    args=(q,k,torch.randn_like(q),-torch.ones_like(q),torch.ones(1,1,device='cuda'),slots)
    pool.step(*args);torch.cuda.synchronize()
    original=pool._finish_update
    def delayed(*args):
        torch.cuda._sleep(100_000_000)
        return original(*args)
    monkeypatch.setattr(pool,'_finish_update',delayed)
    pool.step(*args)
    first,second=torch.cuda.Stream(),torch.cuda.Stream()
    with torch.cuda.stream(first):
        pool.wait()
    with torch.cuda.stream(second):
        pool.wait()
        consumed=torch.cuda.Event()
        consumed.record()
    consumed.synchronize()
    assert pool.completion.query(), 'a wait on one stream must not drop another stream dependency'



def test_graph_prefix_allocations_cannot_alias_late_background_inputs(monkeypatch):
    torch.manual_seed(37)
    plan=QuantizationPlan('kda',torch.full((1,32),4,device='cuda'),torch.ones(1,32,device='cuda'))
    pool=PackedStatePool(plan,5,32,writeback_stream=torch.cuda.Stream())
    ref=PackedStatePool(plan,5,32,writeback_stream=torch.cuda.Stream())
    slots=torch.arange(1,5,device='cuda')
    q=torch.randn(4,1,32,device='cuda');k=torch.nn.functional.normalize(torch.randn_like(q),dim=-1)
    args=(q,k,torch.randn_like(q),-torch.ones_like(q)*.1,torch.ones(4,1,device='cuda')*.5,slots)
    state=torch.randn(4,1,32,32,device='cuda')*.1
    pool.step(*args);torch.cuda.synchronize()
    fg=torch.cuda.CUDAGraph()
    with torch.cuda.graph(fg):
        # This allocation is freed before the state layer. A later graph-private
        # writer input could reuse it, then be clobbered by the next graph prefix.
        prefix=torch.empty_like(q)
        prefix.fill_(1000.)
        del prefix
        with defer_writeback([pool]) as tasks:
            output=pool.step(*args)
    original=pool._finish_update
    def delayed(*args):
        torch.cuda._sleep(50_000_000)
        return original(*args)
    monkeypatch.setattr(pool,'_finish_update',delayed)
    graph=WritebackGraph(fg,tasks)
    pool.encode(state,slots);ref.encode(state,slots)
    graph.replay();graph.replay()
    ref.step(*args);expected=ref.step(*args)
    torch.testing.assert_close(output,expected,atol=5e-5,rtol=1e-4)
    torch.testing.assert_close(pool.decode(slots),ref.decode(slots),atol=3e-4,rtol=2e-3)


@pytest.mark.parametrize('architecture',['gdn','kda'])
@pytest.mark.parametrize('storage',['packed','byte'])
@pytest.mark.parametrize('batch',[32,512])
def test_joined_and_paired_graph_handoff(architecture,storage,batch):
    test_split_graph_queued_replays_and_batch_switch(
        architecture,128,True,storage,batch,joined=True)


@pytest.mark.parametrize('next_producer',['same','joined','paired','eager'])
def test_joined_writeback_protects_an_independently_captured_consumer(monkeypatch,next_producer):
    plan=QuantizationPlan('kda',torch.full((1,32),4,device='cuda'),torch.ones(1,32,device='cuda'))
    pool=PackedStatePool(plan,2,32,writeback_stream=torch.cuda.Stream(),storage='byte')
    slots=torch.ones(1,device='cuda',dtype=torch.long)
    q=torch.randn(1,1,32,device='cuda')
    k=torch.nn.functional.normalize(torch.randn_like(q),dim=-1)
    v=torch.ones_like(q)
    args=(q,k,v,-torch.ones_like(q)*.1,torch.ones(1,1,device='cuda')*.5,slots)
    for _ in range(2):pool.step(*args)
    torch.cuda.synchronize()
    original=pool._finish_update
    def delayed(*args):
        torch.cuda._sleep(50_000_000)
        return original(*args)
    monkeypatch.setattr(pool,'_finish_update',delayed)
    forward=torch.cuda.CUDAGraph()
    with torch.cuda.graph(forward):
        with defer_writeback([pool],joined=True) as tasks:
            pool.step(*args)
    graph=WritebackGraph(forward,tasks,joined=True)
    graph.replay()
    torch.cuda.synchronize()
    consumer=torch.cuda.CUDAGraph()
    with torch.cuda.graph(consumer):
        observed=pool.decode(slots)
    if next_producer in ('joined','paired'):
        other=torch.cuda.CUDAGraph()
        joined=next_producer=='joined'
        with torch.cuda.graph(other):
            with defer_writeback([pool],joined=joined) as tasks:
                pool.step(*args)
        graph=WritebackGraph(other,tasks,joined=joined)
    v.fill_(7.)
    producer=torch.cuda.Stream()
    producer.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(producer):
        if next_producer=='eager':pool.step(*args)
        else:graph.replay()
    consumer.replay()
    done=torch.cuda.Event()
    done.record();done.synchronize()
    assert pool.completion.query(), 'captured consumer waited on a stale producer event'
    torch.testing.assert_close(observed,pool.decode(slots),atol=0,rtol=0)


def _priority_first_graph(monkeypatch):
    from stepquant.kernels.priority_graph import PriorityDecodeGraph
    original_graph=torch.cuda.CUDAGraph
    first=True
    def create():
        nonlocal first
        graph=PriorityDecodeGraph() if first else original_graph()
        first=False
        return graph
    monkeypatch.setattr(torch.cuda,'CUDAGraph',create)


@pytest.mark.parametrize('joined',[False,True])
@pytest.mark.parametrize('architecture,batch,sms',[('kda',64,24),('kda',512,64),('gdn',256,56),('gdn',512,64)])
def test_priority_decode_graph_queued_replays(monkeypatch,joined,architecture,batch,sms):
    from stepquant.kernels import resources
    original_resources=resources.writeback_resources
    _priority_first_graph(monkeypatch)
    monkeypatch.setattr(resources,'writeback_resources',lambda device,sm_budget:original_resources(device,sms))
    test_split_graph_queued_replays_and_batch_switch(architecture,128,True,'byte',batch,joined)


@pytest.mark.parametrize('handoff',['same','joined','paired','eager'])
def test_priority_decode_graph_independent_consumer(monkeypatch,handoff):
    _priority_first_graph(monkeypatch)
    test_joined_writeback_protects_an_independently_captured_consumer(monkeypatch,handoff)


def test_priority_graph_keeps_native_rng_replay():
    from stepquant.kernels.priority_graph import PriorityDecodeGraph
    stream=torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for _ in range(3):output=torch.rand(256,device='cuda')
    stream.synchronize()
    graph=PriorityDecodeGraph()
    with torch.cuda.graph(graph,stream=stream):
        output=torch.rand(256,device='cuda')
    graph.replay();first=output.clone()
    graph.replay();second=output.clone()
    torch.cuda.synchronize()
    assert not torch.equal(first,second)
    graph.reset()
