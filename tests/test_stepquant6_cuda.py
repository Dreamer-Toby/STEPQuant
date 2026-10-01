"""STEPQuant@6 packed forward/fit and graph replay across writer SM budgets."""
from types import SimpleNamespace
import pytest
import torch
pytest.importorskip('triton')
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
from stepquant.core import delta_step
from stepquant.quantization import QuantizationPlan, fit_state
from stepquant.kernels.pool import PackedStatePool
from stepquant.kernels.profiles import stepquant6_profile
from stepquant.kernels.resources import writeback_resources


def make_plan(architecture, size=128):
    bits = torch.tensor([4, 6, 8, 16], device='cuda')[:, None].expand(4, size).clone()
    if architecture == 'kda':
        bits = torch.tensor([4, 6, 8], device='cuda').repeat((2, (size + 2) // 3))[:, :size].clone()
        bits[:, ::17] = 16
    weights = torch.linspace(.25, 2., bits.numel(), device='cuda').reshape_as(bits)
    return QuantizationPlan(architecture, bits, weights)


@pytest.mark.parametrize('architecture', ['gdn', 'kda'])
@pytest.mark.parametrize('batch', [32, 64, 128, 256, 512])
def test_six_bit_profile_forward_and_single_weighted_fit(architecture, batch):
    torch.manual_seed(6)
    plan = make_plan(architecture)
    profile = stepquant6_profile(architecture, batch)
    resources = writeback_resources('cuda:0', profile.sms)
    pool = PackedStatePool(plan, batch + 1, 128, writeback_stream=resources.stream,
                           writeback_chunk=profile.chunk, storage='packed')
    slots = torch.arange(batch, 0, -1, device='cuda')
    slots[-1] = 0
    state = torch.randn(batch, *plan.bits.shape, 128, device='cuda') * .1
    pool.encode(state, slots)
    for _ in range(2):
        before = pool.decode(slots)
        q = torch.randn(batch, *plan.bits.shape, device='cuda')
        k = torch.nn.functional.normalize(torch.randn_like(q), dim=-1)
        v = torch.randn(batch, plan.bits.shape[0], 128, device='cuda')
        g = -torch.rand_like(q) if architecture == 'kda' else -torch.rand(batch, plan.bits.shape[0], device='cuda')
        beta = torch.rand(batch, plan.bits.shape[0], device='cuda')
        output, updated = delta_step(before, q, k, v, g, beta)
        actual = pool.step(q, k, v, g, beta, slots)
        torch.testing.assert_close(actual[:-1], output[:-1], atol=5e-5, rtol=1e-4)
        reconstructed = pool.decode(slots)
        reference = fit_state(updated, plan)[-1]
        for bit in (4, 6, 8):
            mask = plan.bits == bit
            oracle_error = (reference[:-1, mask] - updated[:-1, mask]).square().mean()
            disagreement = (reconstructed[:-1, mask] - reference[:-1, mask]).square().mean()
            assert disagreement < .01 * oracle_error + 1e-10
            assert (reconstructed[:-1, mask] - updated[:-1, mask]).square().mean() < 1.02 * oracle_error + 1e-10
        torch.testing.assert_close(reconstructed[:-1, plan.bits == 16], reference[:-1, plan.bits == 16], atol=1e-5, rtol=1e-3)
        assert actual[-1].count_nonzero() == 0
        assert pool.data[0].count_nonzero() == 0


@pytest.mark.parametrize('architecture', ['gdn', 'kda'])
def test_six_bit_graphs_keep_streams_and_order_shared_scratch(monkeypatch, architecture):
    from stepquant.sglang import runtime
    from stepquant.kernels.writeback import attention_window
    import weakref
    monkeypatch.setattr(runtime, '_temporal_pools', weakref.WeakSet())
    monkeypatch.setattr(runtime, '_writeback_profiles', {})
    monkeypatch.setenv('STEPQUANT_STATE_FORMAT', 'stepquant6')
    monkeypatch.setenv('STEPQUANT_ARCHITECTURE', architecture)
    monkeypatch.setenv('STEPQUANT_STATE_STORAGE', 'packed')
    monkeypatch.setenv('STEPQUANT_WRITEBACK_SMS', 'auto')
    monkeypatch.setenv('STEPQUANT_WRITEBACK_CHUNK', 'auto')
    torch.manual_seed(16)
    plan = make_plan(architecture, 32)
    stream = writeback_resources('cuda:0', 32).stream
    workspace = {}
    pools = [PackedStatePool(plan, 513, 32, writeback_stream=stream, writeback_workspace=workspace) for _ in range(2)]
    temporal = runtime.TemporalPages(pools)  # Retain the weakly registered pages.
    refs = [PackedStatePool(plan, 513, 32, writeback_stream=torch.cuda.Stream()) for _ in pools]
    assert pools[0].coefficients.data_ptr() == pools[1].coefficients.data_ptr()
    runner = SimpleNamespace(_create_device_graph=torch.cuda.CUDAGraph)

    def original(runner, batch, forward, stream_idx):
        for _ in range(2):
            forward()
        torch.cuda.synchronize()
        graph = runner._create_device_graph()
        with torch.cuda.graph(graph):
            outputs = forward()
        return graph, outputs

    graphs, outputs, inputs = {}, {}, {}
    for batch in (512, 256, 128, 64, 32):
        q = torch.randn(batch, *plan.bits.shape, device='cuda')
        k = torch.nn.functional.normalize(torch.randn_like(q), dim=-1)
        v = torch.randn(batch, plan.bits.shape[0], 32, device='cuda')
        g = -torch.rand_like(q) if architecture == 'kda' else -torch.rand(batch, plan.bits.shape[0], device='cuda')
        beta = torch.rand(batch, plan.bits.shape[0], device='cuda')
        slots = torch.arange(batch, 0, -1, device='cuda')
        inputs[batch] = (q, k, v, g, beta, slots)

        def forward():
            result = []
            for pool in temporal.pages:
                attention_window()
                result.append(pool.step(*inputs[batch]))
                attention_window()
            return result

        graphs[batch], outputs[batch] = runtime.capture_graph(original, runner, batch, forward)
        expected_stream = writeback_resources('cuda:0', stepquant6_profile(architecture, batch).sms).stream
        assert graphs[batch].stream == expected_stream
        if torch.cuda.get_device_capability() == (8, 0):
            assert all(n > 0 for n in graphs[batch].forward.priority_counts)
    slots = torch.arange(513, device='cuda')
    for pool in pools + refs:
        pool.clear(slots)
    sequence = (512, 32, 256, 64, 128, 512, 32)
    results = []
    for batch in sequence:
        graphs[batch].replay()
        results.append([output.clone() for output in outputs[batch]])
    # Queue the graph-size changes without synchronizing between requests.
    for batch, actual in zip(sequence, results):
        for ref, output in zip(refs, actual):
            torch.testing.assert_close(output, ref.step(*inputs[batch]), atol=5e-5, rtol=1e-4)
    for pool, ref in zip(pools, refs):
        torch.testing.assert_close(pool.decode(slots), ref.decode(slots), atol=3e-4, rtol=2e-3)

    # A graph captured earlier retains its original writer after rebinding.
    different_batch = 512 if architecture == 'kda' else 64
    assert graphs[different_batch].stream != graphs[32].stream
    assert set(runtime._writeback_profiles) == {'32', '64', '128', '256', '512'}
