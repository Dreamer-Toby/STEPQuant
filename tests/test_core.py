import itertools
import torch
import pytest
from stepquant.core import delta_step, row_impact, lifetime_weight
from stepquant.packing import pack, unpack
from stepquant.quantization import QuantizationPlan, StateCodec, fit_state
from stepquant.allocation import allocate_dp, allocate_lagrangian
from stepquant.calibration import LayerStatistics, calibrate


@pytest.mark.parametrize('architecture', ['gdn', 'kda'])
def test_error_propagation_and_readout_order(architecture):
    torch.manual_seed(2)
    b,h,k,v = 2,3,7,5
    state = torch.randn(b,h,k,v)
    error = torch.randn_like(state) * .01
    q, key = torch.randn(b,h,k), torch.nn.functional.normalize(torch.randn(b,h,k), dim=-1)
    val, beta = torch.randn(b,h,v), torch.rand(b,h)
    g = -torch.rand(b,h) if architecture == 'gdn' else -torch.rand(b,h,k)
    y, s = delta_step(state, q, key, val, g, beta)
    yhat, shat = delta_step(state + error, q, key, val, g, beta)
    d = g.exp().unsqueeze(-1) if architecture == 'gdn' else g.exp()
    a = (torch.eye(k) - beta[...,None,None] * key[...,None] * key[...,None,:]) * d[...,None,:]
    torch.testing.assert_close(shat - s, a @ error, atol=1e-6, rtol=1e-4)
    transported = (a.transpose(-1,-2) @ q[...,None]).squeeze(-1)
    torch.testing.assert_close(row_impact(q,key,g,beta), transported.square())
    torch.testing.assert_close(yhat-y, (error.transpose(-1,-2) @ transported[...,None]).squeeze(-1), atol=1e-6, rtol=1e-4)


def test_lifetime_extremes():
    g = torch.tensor([0., -1e-10, -.1, -100, -float('inf')], dtype=torch.float64)
    expected = torch.stack([torch.exp(2*x*torch.arange(1,128)).sum()+1 for x in g])
    torch.testing.assert_close(lifetime_weight(g,128), expected)


@pytest.mark.parametrize('bits', [2,4,6,8])
@pytest.mark.parametrize('n', [0,1,7,128,513])
def test_bitpacking(bits,n):
    x=torch.randint(2**bits,(n,))
    p=pack(x,bits)
    assert p.dtype == torch.uint8
    torch.testing.assert_close(unpack(p,bits,n).long(),x)
    assert p.numel() <= (n*bits+7)//8 + 2


@pytest.mark.parametrize('architecture', ['gdn','kda'])
def test_packed_equals_fitted_and_pivots(architecture):
    torch.manual_seed(1)
    bits=torch.tensor([2,4,6,8,16])[:,None].expand(5,8).clone()
    if architecture=='kda':
        bits=bits.T.contiguous()
    w=torch.rand(bits.shape)+.2
    plan=QuantizationPlan(architecture,bits,w,value_group_size=4)
    codec=StateCodec(plan)
    x=torch.randn(2,*bits.shape,16)
    fitted=fit_state(x,plan)[-1]
    packed=codec.encode(x)
    out=codec.decode(packed)
    torch.testing.assert_close(out,fitted,atol=1e-6,rtol=1e-6)
    torch.testing.assert_close(out[:,bits==16],x[:,bits==16].half().float(),atol=0,rtol=0)
    assert packed.nbytes < x.numel()*4
    zero=codec.decode(codec.encode(torch.zeros_like(x)))
    assert torch.isfinite(zero).all()
    assert zero.abs().max()<1e-10


def test_dp_matches_brute_force_and_lagrangian_feasible():
    torch.manual_seed(3)
    costs=torch.rand(5,3).double()
    candidates=[4,6,8]; budget=30
    assignment=allocate_dp(costs,candidates,budget)
    result=sum(costs[i,candidates.index(int(b))] for i,b in enumerate(assignment))
    expected=min(sum(costs[i,candidates.index(b)] for i,b in enumerate(bs)) for bs in itertools.product(candidates,repeat=5) if sum(bs)<=budget)
    assert abs(result-expected)<1e-10
    assert allocate_lagrangian(costs,candidates,budget).sum()<=budget


@pytest.mark.parametrize('architecture', ['gdn','kda'])
@pytest.mark.parametrize('budget',[4,6])
def test_calibration_pipeline(architecture,budget):
    torch.manual_seed(5)
    observer=LayerStatistics(architecture, sample_every=2,max_snapshots=4)
    state=torch.zeros(1,4,8,8)
    for _ in range(12):
        q,k=torch.randn(1,4,8),torch.nn.functional.normalize(torch.randn(1,4,8),dim=-1)
        v,beta=torch.randn(1,4,8),torch.rand(1,4)
        g=-torch.rand(1,4) if architecture=='gdn' else -torch.rand(1,4,8)
        _,state=delta_step(state,q,k,v,g,beta)
        observer.observe(q,k,v,g,beta,state)
    artifact=calibrate({'layer':observer.export()},nominal_bits=budget,pivots=1,value_group_size=4)
    p=QuantizationPlan(**artifact['plans']['layer'])
    assert artifact['settings']['integer_bits_used']<=artifact['settings']['integer_budget']
    assert (p.bits==16).sum()==(8 if architecture=='gdn' else 1)
    assert torch.isfinite(StateCodec(p).decode(StateCodec(p).encode(state))).all()


def test_kda_pivots_excluded_from_shared_column_fit():
    bits=torch.tensor([[2,4,6,8,16]])
    codec=StateCodec(QuantizationPlan('kda',bits,torch.ones_like(bits).float()))
    torch.manual_seed(7)
    x=torch.randn(1,1,5,32)
    y=x.clone(); y[:,:,4]*=1000
    a,b=codec.encode(x),codec.encode(y)
    torch.testing.assert_close(a.columns,b.columns,atol=0,rtol=0)
    for bit in (2,4,6,8):
        torch.testing.assert_close(a.codes[bit],b.codes[bit],atol=0,rtol=0)


def test_longer_lifetime_receives_higher_precision():
    distortions=torch.tensor([[1.,.1,.01],[1.,.1,.01]])
    lifetime=lifetime_weight(torch.tensor([-.5,-.001]),2048)
    allocated=allocate_dp(distortions*lifetime[:,None],[4,6,8],12)
    assert allocated.tolist()==[4,8]
