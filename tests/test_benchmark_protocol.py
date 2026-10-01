"""STEPQuant timing must use a calibrated precision map, including its pivots."""
import importlib.util
from pathlib import Path
import pytest
import torch
from stepquant.formats import BASELINE_NAMES
from stepquant.reproducibility import file_sha256

pytest.importorskip('triton')
_spec=importlib.util.spec_from_file_location('bench_formats',Path(__file__).parents[1]/'benchmarks/bench_formats.py')
bench=importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bench)


def saved_plan(tmp_path, architecture, bits=6):
    precisions=torch.tensor([4,6,8,16])[:,None].expand(4,128).clone()
    if architecture=='kda':
        precisions=torch.tensor([4,6,8,16]).repeat(4,32)
    entry=dict(architecture=architecture,bits=precisions,
               impact=torch.linspace(.2,2.,512).reshape(4,128),value_group_size=32)
    path=tmp_path/'plan.pt'
    torch.save(dict(format_version=2,state_format='stepquant',settings={'nominal_bits':bits},
                    plans={'model.layers.0.linear_attn':entry}),path)
    return path,entry


@pytest.mark.parametrize('architecture',['gdn','kda'])
@pytest.mark.parametrize('bits',[4,6])
def test_plan_selects_matching_format_and_preserves_sharded_map(tmp_path,architecture,bits):
    path,entry=saved_plan(tmp_path,architecture,bits)
    args,calibration,metadata=bench.parse_args(['--plan',str(path),'--tp-size','2','--tp-rank','1','--output',str(tmp_path/'out.json')])
    assert args.formats==['fp32',f'stepquant{bits}'] and args.architectures==[architecture]
    assert args.heads==2
    assert torch.equal(calibration['bits'],entry['bits'][2:])
    assert torch.equal(calibration['impact'],entry['impact'][2:])
    assert metadata['plan_sha256']==file_sha256(path) and metadata['tp_rank']==1


@pytest.mark.parametrize('format',['stepquant4','stepquant6'])
def test_stepquant_without_plan_fails_before_gpu_allocation(tmp_path,format,monkeypatch):
    monkeypatch.setattr(torch.cuda,'Stream',lambda *a,**kw:pytest.fail('must reject before GPU work'))
    with pytest.raises(SystemExit):
        bench.parse_args(['--formats',format,'--output',str(tmp_path/'out.json')])
    with pytest.raises(ValueError,match='requires --plan'):
        bench.make_pool(format,'gdn',4,33)
    assert not (tmp_path/'out.json').exists()


def test_no_plan_defaults_to_paper_baselines(tmp_path):
    args,calibration,metadata=bench.parse_args(['--output',str(tmp_path/'out.json')])
    assert args.formats==BASELINE_NAMES and args.architectures==['gdn','kda']
    assert calibration is None and not metadata


@pytest.mark.parametrize('damage',['int2','format','architecture'])
def test_reject_plan_incompatible_with_six_bit_protocol(tmp_path,damage):
    path,entry=saved_plan(tmp_path,'gdn')
    extra=[]
    if damage=='int2':
        entry['bits'][0]=2
        artifact=torch.load(path,weights_only=True)
        artifact['plans']['model.layers.0.linear_attn']=entry
        torch.save(artifact,path)
    elif damage=='format':extra=['--formats','stepquant4']
    else:extra=['--architectures','kda']
    with pytest.raises(SystemExit):
        bench.parse_args(['--plan',str(path),'--tp-size','1','--output',str(tmp_path/'out.json'),*extra])
