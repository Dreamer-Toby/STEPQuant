"""Lossless TP shard arrangement, including graph buffers and fallback paths."""
from types import SimpleNamespace
import pytest
import torch
triton=pytest.importorskip('triton')
pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
from stepquant.sglang.logits import arrange_logits,get_logits

@pytest.mark.parametrize('dtype',[torch.bfloat16,torch.float32])
@pytest.mark.parametrize('buffered',[False,True])
def test_logits_gather_cast_matches_upstream(monkeypatch,dtype,buffered):
    import sglang.srt.distributed as distributed
    torch.manual_seed(9)
    n,shard,vocab=512,37,143
    source=torch.randn(4,n,shard,device='cuda',dtype=dtype)
    expected_source=source.clone().mul_(.7)
    def gather(dst,local):
        torch.testing.assert_close(local,expected_source[0],atol=0,rtol=0)
        dst.copy_(expected_source.reshape(4*n,shard))
    group=SimpleNamespace(world_size=4,all_gather_into_tensor=gather)
    monkeypatch.setattr(distributed,'get_tp_group',lambda:group)
    processor=SimpleNamespace(do_tensor_parallel_all_gather=True,use_attn_tp_group=False,
        do_tensor_parallel_all_gather_dp_attn=False,final_logit_softcapping=None,
        logit_scale=.7,vocab_size=vocab,_compute_lm_head=lambda *args:source[0].clone())
    buffer=torch.empty(n,vocab+8,device='cuda')[:,:vocab] if buffered else None
    meta=SimpleNamespace(next_token_logits_buffer=buffer)
    def unexpected(*args):raise AssertionError('unexpected fallback')
    result=get_logits(unexpected,processor,torch.empty(n,1,device='cuda'),None,meta)
    expected=expected_source.permute(1,0,2).reshape(n,4*shard)[:,:vocab].float()
    torch.testing.assert_close(result,expected,atol=0,rtol=0)
    if buffered:assert result.data_ptr()==buffer.data_ptr()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        arrange_logits[(triton.cdiv(n*vocab,2048),)](expected_source,result,n,shard,vocab,result.stride(0),2048)
    expected_source.add_(1)
    graph.replay()
    torch.testing.assert_close(result,expected_source.permute(1,0,2).reshape(n,4*shard)[:,:vocab].float(),atol=0,rtol=0)

@pytest.mark.parametrize('condition',['batch','no_tp','attn_tp','dp','softcap'])
def test_logits_keeps_upstream_other_paths(condition):
    p=SimpleNamespace(do_tensor_parallel_all_gather=condition!='no_tp',use_attn_tp_group=condition=='attn_tp',
        do_tensor_parallel_all_gather_dp_attn=condition=='dp',final_logit_softcapping=1 if condition=='softcap' else None)
    hidden=torch.empty(64 if condition=='batch' else 512,1,device='cuda');sentinel=object()
    def original(*args):
        assert args[0] is p and args[1] is hidden
        return sentinel
    assert get_logits(original,p,hidden,None,None) is sentinel
