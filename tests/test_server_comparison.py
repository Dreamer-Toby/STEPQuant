"""Reject misleading real-model performance comparisons."""
import importlib.util
import json
from pathlib import Path
import pytest

_spec = importlib.util.spec_from_file_location('compare_server', Path(__file__).parents[1]/'benchmarks/compare_server.py')
_compare = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_compare)


def report(tmp_path, name, native, interval):
    directory = tmp_path/name
    directory.mkdir()
    for rank in range(2):
        value = dict(rank=rank, state_format='fp32' if native else 'stepquant4',
                     kernel_backend='native' if native else 'auto', async_writeback=not native,
                     kernel_sha256={'fused.py':'test'}, dropped_events=0,
                     intervals=[dict(segment=s,batch=4,start_to_start_ms=interval+rank*.1)
                                for s in (1,2,3) for _ in range(2)])
        (directory/f'rank-{rank}.json').write_text(json.dumps(value))
    return dict(batch=4,prompt_tokens=8,output_tokens=4,rounds=[{}, {}, {}],
                median_wall_seconds=1.,configuration={'model_path':'same-model','tp_size':2},
                before=[{'stepquant_decode_timing':{'path':str(directory/'rank-0.json'),'last_segment':0}}])


def test_compare_uses_slowest_rank_and_native_fp32(tmp_path):
    baseline=report(tmp_path,'base',True,2.)
    candidate=report(tmp_path,'candidate',False,1.)
    result=_compare.compare(baseline,candidate,2)
    assert result['speedup_vs_fp32']==pytest.approx(2.1/1.1)
    for rank in range(2):
        path=tmp_path/'base'/f'rank-{rank}.json'
        data=json.loads(path.read_text());data['kernel_backend']='auto'
        path.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='native FP32'):
        _compare.compare(baseline,candidate,2)


def test_comparison_uses_the_measured_graph_writeback_profile(tmp_path):
    candidate=report(tmp_path,'candidate',False,1.)
    directory=Path(candidate['before'][0]['stepquant_decode_timing']['path']).parent
    for path in directory.glob('rank-*.json'):
        data=json.loads(path.read_text())
        data.update(state_format='stepquant6',writeback_sms=80,writeback_chunk=None,
                    writeback_profiles={'4':dict(actual_sms=32,chunk=64),'512':dict(actual_sms=80,chunk=64)})
        path.write_text(json.dumps(data))
    measured=_compare.summarize(candidate,2)
    assert measured['writeback_sms']==32
    assert measured['writeback_chunk']==64


@pytest.mark.parametrize('damage',['configuration','batch','rounds'])
def test_reject_unmatched_workloads(tmp_path,damage):
    baseline=report(tmp_path,'base',True,2.)
    candidate=report(tmp_path,'candidate',False,1.)
    if damage=='configuration':candidate['configuration']['model_path']='other-model'
    elif damage=='rounds':candidate['rounds'].pop()
    else:
        path=tmp_path/'candidate'/'rank-0.json'
        data=json.loads(path.read_text());data['intervals'][0]['batch']=3
        path.write_text(json.dumps(data))
    with pytest.raises(ValueError):_compare.compare(baseline,candidate,2)


def test_local_checkpoint_aliases_are_verified_without_rewriting_reports(tmp_path):
    model=tmp_path/'model';model.mkdir()
    alias=tmp_path/'alias';alias.symlink_to(model,target_is_directory=True)
    baseline=report(tmp_path,'base',True,2.)
    candidate=report(tmp_path,'candidate',False,1.)
    baseline['configuration']['model_path']=str(model)
    candidate['configuration']['model_path']=str(alias)
    assert _compare.compare(baseline,candidate,2)['configuration_equality_verified']
    assert candidate['configuration']['model_path']==str(alias)


def test_excludes_profile_requests_after_measurement(tmp_path):
    baseline=report(tmp_path,'base',True,2.)
    candidate=report(tmp_path,'candidate',False,1.)
    for value in (baseline,candidate):
        value['after']=[{'stepquant_decode_timing':{'last_segment':3}}]
        directory=Path(value['before'][0]['stepquant_decode_timing']['path']).parent
        for path in directory.glob('rank-*.json'):
            data=json.loads(path.read_text())
            data['intervals'].append(dict(segment=4,batch=1,start_to_start_ms=999.))
            path.write_text(json.dumps(data))
    assert _compare.compare(baseline,candidate,2)['speedup_vs_fp32']==pytest.approx(2.1/1.1)


@pytest.mark.parametrize('field,value',[('prompt_mode','matrix'),('prompt_sha256','different-token-sequence')])
def test_reject_different_prompt_content(tmp_path,field,value):
    baseline=report(tmp_path,'base',True,2.)
    candidate=report(tmp_path,'candidate',False,1.)
    candidate[field]=value
    with pytest.raises(ValueError,match='prompt'):
        _compare.compare(baseline,candidate,2)


@pytest.mark.parametrize('field,value',[('versions',{'triton':'other-version'}),('moe_config_sha256',{'tiles.json':'different'})])
def test_reject_different_runtime_or_common_moe_tuning(tmp_path,field,value):
    baseline=report(tmp_path,'base',True,2.)
    candidate=report(tmp_path,'candidate',False,1.)
    for rank in range(2):
        path=tmp_path/'candidate'/f'rank-{rank}.json'
        data=json.loads(path.read_text());data[field]=value
        path.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='incompatible'):
        _compare.compare(baseline,candidate,2)
