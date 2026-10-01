import pytest
from stepquant.sglang.config import BATCH_SIZES, graph_batches, request_limit


@pytest.mark.parametrize('workload',['long','short'])
@pytest.mark.parametrize('cap',BATCH_SIZES)
def test_explicit_capacity_has_exact_graphs(workload,cap):
    assert request_limit(workload,cap)==cap
    sizes=graph_batches(cap)
    assert sizes[-1]==cap
    assert all(n in sizes for n in BATCH_SIZES if n<=cap)
    assert sizes==sorted(set(sizes))


def test_defaults_and_invalid_capacity():
    assert request_limit('long')==64
    assert request_limit('short')==256
    for cap in (0,-1,513,True,32.5):
        with pytest.raises(ValueError):request_limit('long',cap)
