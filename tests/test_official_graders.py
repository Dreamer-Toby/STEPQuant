"""Optional evaluator integration; run after scripts/reuse_eval_env.sh."""
import json
import os
from pathlib import Path
import pytest
from stepquant.evaluation.upstream import use
from stepquant.evaluation.pipeline import digest
from stepquant.evaluation.grade import grade
from stepquant.evaluation.prepare import render

pytestmark=pytest.mark.skipif(os.environ.get('STEPQUANT_EVAL_TESTS')!='1',reason='set STEPQUANT_EVAL_TESTS=1 in .venv-eval to run official graders')


def fixture_run(tmp_path,config,tasks,answers):
    manifest=dict(config=config,tasks_sha256=digest(tasks),run_hash='fixture')
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    (tmp_path/'responses.jsonl').write_text(''.join(json.dumps(dict(sample_id=f"{t['id']}:0",task_id=t['id'],sample=0,answer=a,finish_reason='stop',run_hash='fixture'))+'\n' for t,a in zip(tasks,answers)))


@pytest.mark.parametrize('name,kind,source,answer',[
 ('aime2026','matharena',dict(problem='6 * 7?',answer='42'),r'\boxed{42}'),
 ('math500','math500',dict(problem='6 * 7?',answer='42'),r'\boxed{42}'),
 ('gpqa_diamond','gpqa',{'Question':'Q?','Correct Answer':'yes','Incorrect Answer 1':'no','Incorrect Answer 2':'wrong','Incorrect Answer 3':'false'},None),
 ('ifbench','ifbench',dict(key=1,prompt='Say hello',instruction_id_list=['keywords:existence'],kwargs=[dict(keywords=['hello'])]),'hello'),
])
def test_official_noncode(tmp_path,name,kind,source,answer):
    c=dict(name=name,grader=kind,samples=1)
    tasks=[render(name,source,0)]
    if answer is None:answer=f"The correct answer is ({tasks[0]['answer']})"
    fixture_run(tmp_path,c,tasks,[answer])
    assert grade(c,tasks,tmp_path)['accuracy']==1


def test_official_lcb(tmp_path):
    row=dict(question_title='Add',question_content='Read two integers and print their sum.',question_id='fixture',
             platform='atcoder',contest_id='fixture',contest_date='2025-01-01T00:00:00',starter_code='',difficulty='easy',
             public_test_cases=json.dumps([dict(input='2 3\n',output='5\n',testtype='stdin')]),
             private_test_cases=json.dumps([dict(input='4 7\n',output='11\n',testtype='stdin')]),metadata='{}')
    c=dict(name='lcb_v6',grader='lcb',samples=1)
    tasks=[render('lcb_v6',row,0)]
    fixture_run(tmp_path,c,tasks,['print(sum(map(int,input().split())))'])
    assert grade(c,tasks,tmp_path,processes=1)['metrics']['pass@1']==1


@pytest.mark.parametrize('name',['humaneval_plus','mbpp_plus'])
def test_official_evalplus(tmp_path,name,monkeypatch):
    use('evalplus','artifacts/upstream')
    from evalplus.data import get_human_eval_plus,get_mbpp_plus
    rows=get_human_eval_plus() if name=='humaneval_plus' else get_mbpp_plus()
    row=next(iter(rows.values()))
    # Official override API narrows this integration test to one task. Formal
    # generation/grade commands never set an override and require all tasks.
    subset=tmp_path/'official-fixture.jsonl'
    subset.write_text(json.dumps(row)+'\n')
    monkeypatch.setenv('HUMANEVAL_OVERRIDE_PATH' if name=='humaneval_plus' else 'MBPP_OVERRIDE_PATH',str(subset))
    c=dict(name=name,grader='evalplus',samples=1)
    tasks=[render(name,row,0)]
    solutions=[row['prompt']+row['canonical_solution'] if name=='humaneval_plus' else row['canonical_solution']]
    fixture_run(tmp_path,c,tasks,solutions)
    result=grade(c,tasks,tmp_path,processes=1)
    assert Path(result['official_results']).exists()
    data=json.loads(Path(result['official_results']).read_text())
    assert all(v[0]['plus_status']=='pass' for v in data['eval'].values())
