from types import SimpleNamespace
import pytest
import torch
from stepquant.sglang import prefill_reduce


def batch(extend):
    return SimpleNamespace(forward_mode=SimpleNamespace(is_extend=lambda:extend))


@pytest.mark.parametrize('dtype',[torch.bfloat16,torch.float16])
def test_prefill_rounds_only_after_rank_sum(dtype):
    group=SimpleNamespace(world_size=4)
    # Intermediate low-precision rounding loses the small contribution.
    values=[torch.tensor([4096.,4096.],dtype=dtype),
            torch.tensor([1.,1.],dtype=dtype),
            torch.tensor([-4096.,-4096.],dtype=dtype),
            torch.tensor([1.,1.],dtype=dtype)]
    calls=[]
    def reduce(group,x):
        calls.append(x.dtype)
        for value in values[1:]:x=x+value.to(x.dtype)
        return x
    def run(self,fb):return prefill_reduce.all_reduce(reduce,group,values[0])
    result=prefill_reduce.forward(run,None,batch(True))
    assert calls==[torch.float32]
    torch.testing.assert_close(result,sum(x.double() for x in values).to(dtype),rtol=0,atol=0)


def test_decode_and_single_rank_preserve_collective_input():
    x=torch.ones(2,dtype=torch.bfloat16)
    def reduce(group,value):
        assert value is x
        return value
    for extend,size in [(False,4),(True,1)]:
        def run(self,fb):return prefill_reduce.all_reduce(reduce,SimpleNamespace(world_size=size),x)
        assert prefill_reduce.forward(run,None,batch(extend)) is x


def test_prefill_context_resets_after_failure():
    def fail(self,fb):raise RuntimeError('forward failed')
    with pytest.raises(RuntimeError,match='forward failed'):
        prefill_reduce.forward(fail,None,batch(True))
    x=torch.ones(2,dtype=torch.bfloat16)
    assert prefill_reduce.all_reduce(lambda group,value:value,SimpleNamespace(world_size=4),x) is x
