import json
from argparse import Namespace
from pathlib import Path
import pytest
import torch
from stepquant.evaluation.data import FrozenTasks, tasks_digest
from stepquant.evaluation.pipeline import digest
from stepquant.evaluation.grade import final_answer
from stepquant.evaluation.suite import server_command, validate_plan
from stepquant.reproduction import commands


def frozen(tmp_path):
    tasks=[dict(id='a',messages=[dict(role='user',content='中文')],source={'tests':'x'*1000})]
    config=dict(name='fixture',expected_tasks=1,protocol_notes=['old description'])
    path=tmp_path/'tasks.jsonl'
    path.write_text('\n'.join(json.dumps(t) for t in tasks)+'\n')
    path.with_suffix('.provenance.json').write_text(json.dumps(dict(config=config,tasks=1,tasks_sha256=digest(tasks))))
    return path,config,tasks


def test_streamed_tasks_match_existing_hash_and_reject_tampering(tmp_path):
    path,config,tasks=frozen(tmp_path)
    assert tasks_digest(iter(tasks))==digest(tasks)
    loaded=FrozenTasks(path,dict(config,protocol_notes=['new description']))
    assert len(loaded)==1 and list(loaded)==tasks and tasks_digest(loaded)==digest(tasks)
    with pytest.raises(ValueError,match='config mismatch'):
        FrozenTasks(path,dict(config,samples=2))
    path.write_text(path.read_text().replace('xxxxx','yyyyy'))
    with pytest.raises(ValueError,match='checksum'):
        FrozenTasks(path,config)


def test_native_fp32_command_and_packed_storage(tmp_path,monkeypatch):
    from stepquant.evaluation import suite
    monkeypatch.setattr(suite,'validate_plan',lambda *args:None)
    a=Namespace(max_running_requests=None,server_python='python',tp=4,attention_backend='triton',context_length=None,
                mem_fraction_static=.85,port=31080,disable_cuda_graph=False,cuda_graph_max_bs=None,
                stepquant_storage='packed',allow_smoke_plan=False)
    model=dict(model_path='/checkpoint',weights='bf16',plans={'stepquant6':'plan.pt'})
    native,cap=server_command(a,model,'fp32','long')
    assert cap==64 and native[native.index('--kernel-backend')+1]=='native'
    packed,cap=server_command(a,model,'stepquant6','short')
    assert cap==256 and packed[packed.index('--state-storage')+1]=='packed'


def test_smoke_plans_are_not_formal(tmp_path):
    from stepquant.cli import fingerprint
    (tmp_path/'config.json').write_text('{}')
    path=tmp_path/'plan.pt';model=dict(model_path=str(tmp_path))
    value=dict(format_version=2,checkpoint_fingerprint=fingerprint(tmp_path),state_format='stepquant',
               settings={'nominal_bits':6},data=dict(segments=1,sequence_length=64))
    torch.save(value,path)
    with pytest.raises(ValueError,match='smoke calibration'):validate_plan(path,model,'stepquant6')
    validate_plan(path,model,'stepquant6',True)
    value['settings']['nominal_bits']=4;torch.save(value,path)
    with pytest.raises(ValueError,match='bit budget'):validate_plan(path,model,'stepquant6',True)


def test_calibration_collects_once_then_refits(tmp_path):
    m=dict(calibration_python='existing-python',model_path='/checkpoint')
    jobs=commands('kimi',m,tmp_path,['stepquant4','stepquant6'])
    assert [j[3] for j in jobs]==['calibrate','fit']
    assert jobs[0][jobs[0].index('--sequence-length')+1]=='2048'
    assert jobs[1][jobs[1].index('--bits')+1]=='6'
    with pytest.raises(ValueError,match='only STEPQuant'):commands('kimi',m,tmp_path,['damp'])


def test_unfinished_reasoning_cannot_be_scored_as_final_answer():
    assert final_answer({'answer':'<think>Maybe \\boxed{42}'})==''
    assert final_answer({'answer':'<think>working</think>\\boxed{42}'})=='\\boxed{42}'


def test_calibration_resume_checks_output_hashes(tmp_path,monkeypatch):
    from stepquant import reproduction
    models=tmp_path/'models.json';models.write_text(json.dumps({'qwen':{'calibration_python':'python','model_path':'/model'}}))
    calls=[]
    def execute(cmd,**kwargs):
        calls.append(cmd)
        for flag in ('--output','--statistics-output'):
            if flag in cmd:
                path=Path(cmd[cmd.index(flag)+1]);path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'fixture')
    monkeypatch.setattr(reproduction.subprocess,'run',execute)
    monkeypatch.setattr(reproduction,'source_identity',lambda:{'source':'fixed'})
    monkeypatch.setattr('sys.argv',['reproduction','--models-config',str(models),'--models','qwen',
                                  '--formats','stepquant4','stepquant6','--devices','0','--output',str(tmp_path/'run')])
    reproduction.main();assert len(calls)==2
    reproduction.main();assert len(calls)==2
    (tmp_path/'run/plans/qwen-stepquant4.pt').write_bytes(b'changed')
    with pytest.raises(ValueError,match='outputs changed'):reproduction.main()
