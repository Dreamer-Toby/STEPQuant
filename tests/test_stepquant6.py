"""Nominal @6 dispatch, tight storage, and explicit scheduling overrides."""
import json
from pathlib import Path
import pytest
import torch
from stepquant.kernels.profiles import stepquant6_profile


def test_profile_tiers_and_invalid_batches():
    for architecture in ('gdn', 'kda'):
        for batch in (1, 17, 32, 33, 64, 65, 128, 129, 256, 257, 512):
            profile = stepquant6_profile(architecture, batch)
            assert 0 < profile.sms <= 108
            assert profile.chunk % 32 == 0
        assert stepquant6_profile(architecture, 33) == stepquant6_profile(architecture, 64)
    for batch in (0, 513, True, 32.5):
        with pytest.raises(ValueError):
            stepquant6_profile('gdn', batch)
    with pytest.raises(ValueError):
        stepquant6_profile('unsupported', 32)


def test_launch_overrides_do_not_disable_other_automatic_settings(monkeypatch):
    pytest.importorskip('triton')
    from stepquant.sglang.runtime import writeback_settings
    monkeypatch.setenv('STEPQUANT_STATE_FORMAT', 'stepquant6')
    monkeypatch.setenv('STEPQUANT_WRITEBACK_SMS', 'auto')
    monkeypatch.setenv('STEPQUANT_WRITEBACK_CHUNK', 'auto')
    assert writeback_settings('gdn', 32) == (32, 64)
    assert writeback_settings('gdn', 512) == (80, 64)
    assert writeback_settings('kda', 512) == (48, 128)
    monkeypatch.setenv('STEPQUANT_WRITEBACK_SMS', '0')
    assert writeback_settings('kda', 512) == (0, 128)
    monkeypatch.setenv('STEPQUANT_WRITEBACK_CHUNK', '0')
    assert writeback_settings('kda', 512) == (0, 0)
    monkeypatch.setenv('STEPQUANT_STATE_FORMAT', 'stepquant4')
    monkeypatch.setenv('STEPQUANT_WRITEBACK_SMS', '56')
    monkeypatch.setenv('STEPQUANT_WRITEBACK_CHUNK', '256')
    assert writeback_settings('gdn', 32) == (56, 256)


@pytest.mark.parametrize('architecture', ['gdn', 'kda'])
def test_six_bit_payload_has_no_byte_padding(architecture):
    pytest.importorskip('triton')
    from stepquant.kernels.pool import packed_layout
    from stepquant.quantization import QuantizationPlan
    heads, keys, values = 2, 128, 128
    bits = torch.full((heads, keys), 6)
    plan = QuantizationPlan(architecture, bits, torch.ones_like(bits).float())
    metadata = 2 * heads * (keys + values)
    packed = packed_layout(plan, values, 'packed')[-1]
    assert packed - metadata == heads * keys * values * 6 // 8
    assert packed_layout(plan, values, 'byte')[-1] - metadata == heads * keys * values


def test_six_bit_benchmark_pairs_full_length_native_fp32():
    preset = json.loads((Path(__file__).parents[1] / 'configs/benchmarks/stepquant6.json').read_text())
    args = preset['common_arguments']
    assert args[args.index('--state-format') + 1] == 'stepquant6'
    assert args[args.index('--output-tokens') + 1] == '1024'
    assert args[args.index('--prompt-tokens') + 1] == '128'
    for model in preset['models'].values():
        assert set(model['batches']) == {'32', '64', '128', '256', '512'}
        assert model['model_arguments'][-1].endswith('-stepquant6.pt')
        for batch_args in model['batches'].values():
            # Let every captured graph select its profile, including smaller
            # batches inside a server whose request capacity is larger.
            assert '--writeback-sm-budget' not in args + batch_args
            assert '--writeback-chunk' not in args + batch_args


def test_automatic_writeback_metadata_is_serializable(tmp_path, monkeypatch):
    pytest.importorskip('triton')
    pytest.importorskip('sglang')
    from types import SimpleNamespace
    from stepquant.sglang import runtime, timing
    profiles = {'64': dict(requested_sms=24, actual_sms=24, chunk=64)}
    monkeypatch.setattr(runtime, '_writeback_profiles', profiles)
    monkeypatch.setattr(timing, '_events', [])
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    monkeypatch.setenv('STEPQUANT_WRITEBACK_SMS', 'auto')
    monkeypatch.setenv('STEPQUANT_WRITEBACK_CHUNK', 'auto')
    monkeypatch.setenv('STEPQUANT_TIMING_DIR', str(tmp_path))
    result = timing.internal_state(lambda *args: SimpleNamespace(internal_state={}), None, None)
    report = json.loads(Path(result.internal_state['stepquant_decode_timing']['path']).read_text())
    assert report['writeback_sms'] is None and report['writeback_chunk'] is None
    assert report['writeback_profiles'] == profiles
