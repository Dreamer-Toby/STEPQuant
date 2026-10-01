from types import SimpleNamespace as NS
import sys
import pytest
from stepquant.sglang import admission


@pytest.fixture
def fake(monkeypatch):
    result=NS(OTHER='pause',CONTINUE='continue',NO_TOKEN='full')
    monkeypatch.setitem(sys.modules,'sglang.srt.managers.schedule_policy',NS(AddReqResult=result))
    allocator=NS(size=100, available_size=lambda:41)
    adder=NS(token_to_kv_pool_allocator=allocator,cur_rem_token_offset=0,page_size=1,can_run_list=[])
    req=NS(sampling_params=NS(max_new_tokens=65536,ignore_eos=True),output_ids=[])
    return adder,req


def test_watermark_resume_and_restore(fake):
    adder,req=fake
    params=req.sampling_params
    def original(self,r):
        assert r.sampling_params.max_new_tokens==1
        assert r.sampling_params.ignore_eos is False
        self.can_run_list.append(r)
        return 'continue'
    assert admission.add_one(original,adder,req)=='continue'
    assert req.sampling_params is params and params.max_new_tokens==65536
    adder.cur_rem_token_offset=1
    assert admission.add_one(original,adder,req)=='pause'  # exactly 60%
    adder.token_to_kv_pool_allocator.available_size=lambda:90
    assert admission.add_one(original,adder,req)=='continue'


def test_exception_restores_sampling(fake):
    adder,req=fake
    params=req.sampling_params
    def fail(*args):
        raise RuntimeError('allocation failed')
    with pytest.raises(RuntimeError):
        admission.add_one(fail,adder,req)
    assert req.sampling_params is params


def test_no_future_budget(fake):
    adder,req=fake
    req.output_ids=[0]*40000
    assert admission.running_offset(None,adder,req)==1
    assert admission.update_budget(lambda _,p,e,m:m,adder,0,10,65536)==1


@pytest.mark.parametrize('cap',[32,64,128,256,512])
def test_chunk_continuation_slot_credit(monkeypatch,cap):
    monkeypatch.setitem(sys.modules,'sglang.srt.server_args',NS(get_global_server_args=lambda:NS(pp_max_micro_batch_size=cap)))
    scheduler=NS(chunked_req=NS(req_pool_idx=47),req_to_token_pool=NS(available_size=lambda:17),token_to_kv_pool_allocator=NS(size=100,available_size=lambda:100))
    assert admission.allocatable(None,scheduler,cap-18)==17
    def original(self):
        self.chunked_req=None
        assert admission.allocatable(None,self,cap-18)==18
    admission.prefill_pass(original,scheduler)
    assert scheduler._stepquant_chunk_slot_credit==0


def test_watermark_before_slot_allocation():
    scheduler=NS(chunked_req=None,waiting_queue=[object()],token_to_kv_pool_allocator=NS(size=100,available_size=lambda:40))
    def must_not_allocate(*args):raise AssertionError('gate must precede state allocation')
    assert admission.prefill_pass(must_not_allocate,scheduler) is None
    scheduler.chunked_req=NS(req_pool_idx=1)
    assert admission.prefill_pass(lambda self:'continued',scheduler)=='continued'
