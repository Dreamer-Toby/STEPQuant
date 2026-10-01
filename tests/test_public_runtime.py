"""Public tuning identities, paired presets and immutable protocol migration."""
import importlib.util
import json
from pathlib import Path
import pytest
from stepquant.device import device_label, public_metadata, tuning_identity
from stepquant.moe import materialize_configs
from stepquant.evaluation.data import FrozenTasks, prepare_protocol, tasks_digest


def test_capacity_neutral_device_and_metadata():
    name = 'NVIDIA A100-SXM4-80GB'
    label = 'NVIDIA A100-SXM4'
    assert device_label(name) == label
    info = public_metadata({'device': name, 'total_memory': 80, 'allocated_bytes': 123,
                            'configs': {'device_name=NVIDIA_A100-SXM4-80GB_down.json': 'hash'}})
    assert info == {'device': label, 'allocated_bytes': 123,
                    'configs': {'device_name=NVIDIA_A100-SXM4_down.json': 'hash'}}


def test_public_metadata_preserves_workload_memory_measurements():
    info = {'measured_memory': '131.18 GiB', 'allocated_gib': 12.5,
            'config_path': '/runtime/device_name=NVIDIA_A100-SXM4-80GB.json'}
    cleaned = public_metadata(info)
    assert cleaned['measured_memory'] == info['measured_memory']
    assert cleaned['allocated_gib'] == info['allocated_gib']
    assert cleaned['config_path'] == '/runtime/device_name=NVIDIA_A100-SXM4.json'


@pytest.mark.parametrize('variant', ['sglang', 'sglang_shared_prompt'])
def test_generated_matching_configs_preserve_every_tuning_value(tmp_path, variant):
    source = Path('configs/kernels') / variant
    family = next((source / 'configs').rglob('*.json')).name.split('device_name=')[1].removesuffix('.json').removesuffix('_down')
    actual = family + '-80GB'
    generated = materialize_configs(source, actual, tmp_path)
    assert generated is not None
    assert tuning_identity(source) == tuning_identity(generated)
    assert any(actual in p.name for p in generated.rglob('*.json'))
    before = {p.name: p.read_bytes() for p in generated.rglob('*.json')}
    assert materialize_configs(source, actual, tmp_path) == generated
    assert before == {p.name: p.read_bytes() for p in generated.rglob('*.json')}
    assert materialize_configs(source, 'unrelated_device', tmp_path) is None


def test_short_protocol_migration_preserves_original_tasks_and_manifest(tmp_path):
    old = dict(name='fixture', expected_tasks=1, few_shot=0,
               sampling={'max_tokens': 256}, chat_template_kwargs={'enable_thinking': False})
    tasks = [dict(id='one', messages=[dict(role='user', content='Q')])]
    path = tmp_path / 'fixture.jsonl'
    path.write_text(json.dumps(tasks[0])+'\n')
    metadata = path.with_suffix('.provenance.json')
    metadata.write_text(json.dumps(dict(config=old, tasks=1, tasks_sha256=tasks_digest(tasks))))
    saved = metadata.read_bytes()
    new = dict(old, sampling={'max_tokens': 2048})
    migrated = prepare_protocol(new, path, 'unused')
    assert migrated != path and migrated.read_bytes() == path.read_bytes()
    assert metadata.read_bytes() == saved
    assert list(FrozenTasks(migrated, new)) == tasks
    assert prepare_protocol(new, path, 'unused') == migrated
    # Corruption must fail before any reuse of a source snapshot.
    path.write_text(json.dumps(dict(tasks[0],id='changed'))+'\n')
    with pytest.raises(ValueError):
        prepare_protocol(dict(old, sampling={'max_tokens': 1024}), path, 'unused')


def test_all_short_configs_use_the_confirmed_generation_protocol():
    configs = [json.loads(p.read_text()) for p in Path('configs/evaluation').glob('*.json')]
    short = [c for c in configs if c.get('workload') == 'short']
    assert len(short) == 6
    for c in short:
        assert c['sampling'] == {'temperature': 0., 'max_tokens': 2048}
        assert c['few_shot'] == 0 and c['chat_template_kwargs']['enable_thinking'] is False


@pytest.mark.parametrize('fmt', ['stepquant4', 'stepquant6'])
def test_preset_commands_use_configured_checkpoints_and_full_decode(fmt):
    spec = importlib.util.spec_from_file_location('run_preset', 'benchmarks/run_preset.py')
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    preset = json.loads(Path(f'configs/benchmarks/{fmt}.json').read_text())
    models = json.loads(Path('configs/models.json').read_text())
    jobs = runner.commands(preset, ['qwen', 'kimi'], [32, 64, 128, 256, 512], models, '/output', '5,6')
    assert len(jobs) == 10
    for cmd in jobs.values():
        assert cmd[cmd.index('--prompt-tokens')+1] == '128'
        assert cmd[cmd.index('--output-tokens')+1] == '1024'
        assert cmd[-1].startswith(f'/output/{fmt}/')
        assert cmd[cmd.index('--state-format')+1] == fmt
    models['qwen']['weights'] = 'awq_int4'
    with pytest.raises(ValueError, match='BF16'):
        runner.commands(preset, ['qwen'], [512], models, '/output')
