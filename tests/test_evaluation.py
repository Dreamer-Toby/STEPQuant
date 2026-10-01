import json
from pathlib import Path
import pytest
from stepquant.evaluation.prepare import render
from stepquant.evaluation.pipeline import read_results, seed_for, digest, run
from stepquant.evaluation.grade import letter_answer, final_answer


def test_protocols():
    root=Path(__file__).resolve().parents[1]/'configs/evaluation'
    expected={'lcb_v6':5,'humaneval_plus':5,'mbpp_plus':5,'aime2026':64,'math500':4,'hmmt_feb2026':64,'gpqa_diamond':8,'ifbench':4}
    for name,n in expected.items():
        c=json.loads((root/f'{name}.json').read_text())
        assert c['samples']==n
        assert c['sampling']==dict(temperature=1.,top_p=.95,top_k=20,max_tokens=65536)
    assert len({seed_for(0,'task',i) for i in range(64)})==64


def test_prepare_and_parsing():
    task=render('arc_challenge',dict(id='arc',question='Q?',choices=dict(label=['1','2'],text=['x','y']),answerKey='2'),0)
    assert task['answer']=='B' and 'B. y' in task['messages'][0]['content']
    row={'Question':'Q?','Correct Answer':'truth',**{f'Incorrect Answer {i}':f'wrong{i}' for i in (1,2,3)}}
    a=render('gpqa_diamond',row,0)
    assert a==render('gpqa_diamond',row,0)
    assert f"{a['answer']}. truth" in a['messages'][0]['content']
    assert letter_answer('The correct answer is (C)')=='C'
    assert letter_answer('A long discussion with B in it') is None
    assert final_answer(dict(answer='',reasoning='42'))==''


def test_partial_jsonl(tmp_path):
    p=tmp_path/'r.jsonl'
    p.write_text('{"a":1}\n{"broken"')
    assert read_results(p)==[{'a':1}]
    assert p.read_text()=='{"a":1}\n'
    p.write_text('broken\n{"a":1}\n')
    with pytest.raises(json.JSONDecodeError):
        read_results(p)


@pytest.mark.parametrize('cap',[64,512])
def test_pipeline_resume_and_identity(tmp_path,monkeypatch,cap):
    from stepquant.evaluation import pipeline
    state=dict(stepquant_admission=dict(watermark=.6),max_running_requests=cap,stepquant_state_format='fp32',model_path='/model',
               stepquant_runtime_identity={'revision':'test'})
    calls=[]
    def request(url,body=None,timeout=None):
        if body is None:
            return dict(internal_states=[state])
        calls.append(body)
        return dict(choices=[dict(message=dict(content='B'),finish_reason='stop')])
    monkeypatch.setattr(pipeline,'request',request)
    config=dict(workload='long',max_running_requests=cap,samples=2,seed=0,sampling=dict(temperature=1,max_tokens=65536))
    tasks=[dict(id='q',messages=[dict(role='user',content='Q')])]
    run(config,tasks,tmp_path,'http://local',workers=2)
    assert len(calls)==2 and calls[0]['seed']!=calls[1]['seed']
    run(config,tasks,tmp_path,'http://local',workers=2)
    assert len(calls)==2
    state['stepquant_state_format']='sym_int8_g128'
    with pytest.raises(ValueError,match='mismatch'):
        run(config,tasks,tmp_path,'http://local',workers=2)
