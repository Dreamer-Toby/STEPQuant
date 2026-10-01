"""Freeze datasets and rendered prompts before generation (no implicit test split)."""
import argparse
import os
os.environ.setdefault("HF_HUB_DISABLE_XET","1")
import hashlib
import json
from pathlib import Path
import random
from .pipeline import digest
from .upstream import use
from .data import tasks_digest


def choices_prompt(question, choices):
    return question+'\n\n'+'\n'.join(f'{chr(65+i)}. {value}' for i,value in enumerate(choices))


def render(name, row, index, seed=0):
    identifier=str(row.get('task_id',row.get('question_id',row.get('id',index))))
    answer=None
    if name in ('aime2026','hmmt_feb2026','math500'):
        prompt=row['problem']+'\n\nSolve the problem and put your final answer in \\boxed{}.'
        answer=str(row['answer'])
    elif name=='gpqa_diamond':
        choices=[row[f'Incorrect Answer {i}'] for i in (1,2,3)]+[row['Correct Answer']]
        rng=random.Random(f'{seed}:{identifier}')
        order=list(range(4)); rng.shuffle(order)
        prompt=choices_prompt(row['Question'],[choices[i] for i in order])+'\n\nGive your reasoning, then write "The correct answer is (X)" with the answer letter.'
        answer=chr(65+order.index(3))
    elif name=='ifbench':
        prompt=row['prompt']
        identifier=str(row.get('key',identifier))
    elif name in ('humaneval_plus','mbpp_plus'):
        prompt='Write a complete Python solution to this task. Return the solution in a Python code block.\n\n'+row['prompt']
    elif name=='lcb_v6':
        prompt='Solve this programming problem in Python.\n\n'+row['question_content']
        if row.get('starter_code'):
            prompt+='\nUse this interface:\n```python\n'+row['starter_code']+'\n```'
        else:
            prompt+='\nRead from stdin and write to stdout.'
        prompt+='\nReturn the complete solution in a Python code block.'
    elif name=='mmlu':
        prompt=choices_prompt(row['question'],row['choices'])
        answer=chr(65+int(row['answer']))
    elif name in ('arc_challenge','openbookqa'):
        prompt=choices_prompt(row.get('question',row.get('question_stem')),row['choices']['text'])
        answer=chr(65+row['choices']['label'].index(row['answerKey']))
    elif name=='hellaswag':
        prompt=choices_prompt('Choose the most plausible continuation:\n'+row['ctx'],row['endings'])
        answer=chr(65+int(row['label']))
    elif name=='winogrande':
        prompt=choices_prompt('Fill the blank:\n'+row['sentence'],[row['option1'],row['option2']])
        answer=chr(64+int(row['answer']))
    elif name=='lambada':
        context,answer=row['text'].rsplit(' ',1)
        prompt='Complete this passage with its next word. Return only that word.\n\n'+context
    else:
        raise ValueError(name)
    if name in ('mmlu','arc_challenge','openbookqa','hellaswag','winogrande'):
        prompt+='\n\nReturn only the letter of the correct answer.'
    return dict(id=identifier,messages=[dict(role='user',content=prompt)],answer=answer,source=row)


def prepare(config, output, upstream='artifacts/upstream', local=None):
    name=config['name']
    output=Path(output)
    if output.exists():
        raise FileExistsError(f'{output} exists; preserve frozen task files or choose a new path')
    provenance={}
    if local:
        path=Path(local)
        rows=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        provenance=dict(local_source=str(path.resolve()),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    elif config['grader']=='evalplus':
        use('evalplus',upstream)
        from evalplus.data import get_human_eval_plus,get_mbpp_plus
        rows=list((get_human_eval_plus() if name=='humaneval_plus' else get_mbpp_plus()).values())
        if name=='mbpp_plus':
            from evalplus.data.mbpp import mbpp_serialize_inputs
            rows=[dict(row,**{key:mbpp_serialize_inputs(row['task_id'],row[key]) for key in ('base_input','plus_input')}) for row in rows]
        provenance=dict(source='pinned EvalPlus release loader',data_sha256=tasks_digest(rows))
    elif name=='gpqa_diamond':
        import csv,io,zipfile
        root=use('gpqa',upstream)
        path=root/'dataset.zip'
        with zipfile.ZipFile(path) as archive:
            # Public password explicitly supplied by the dataset authors' README.
            text=archive.read('dataset/gpqa_diamond.csv',pwd=b'deserted-untie-orchid').decode('utf-8-sig')
        rows=list(csv.DictReader(io.StringIO(text)))
        if len(rows)!=198:raise ValueError('unexpected GPQA Diamond size')
        provenance=dict(source='pinned official GPQA dataset.zip:gpqa_diamond.csv',sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    elif name=='ifbench':
        root=use('IFBench',upstream)
        path=root/'data/IFBench_test.jsonl'
        rows=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        provenance=dict(source='pinned IFBench data/IFBench_test.jsonl',sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    else:
        from huggingface_hub import HfApi
        from datasets import load_dataset
        spec=config['dataset']
        revision=spec.get('revision') or HfApi().dataset_info(spec['repo']).sha
        if name=='mmlu':
            # Download only the requested split, not the 47 MB auxiliary train.
            from huggingface_hub import hf_hub_download
            import pyarrow.parquet as pq
            path=hf_hub_download(spec['repo'],f"all/{spec['split']}-00000-of-00001.parquet",repo_type='dataset',revision=revision)
            rows=pq.read_table(path).to_pylist()
            dataset=None
        elif name=='lcb_v6':
            # Match the pinned official loader's release_v6 file list without
            # executing its remote dataset script (modern datasets compatibility).
            from huggingface_hub import hf_hub_download
            files=['test.jsonl']+[f'test{i}.jsonl' for i in range(2,7)]
            rows=[]
            for file in files:
                path=hf_hub_download(spec['repo'],file,repo_type='dataset',revision=revision)
                with open(path) as stream:
                    rows.extend(json.loads(s) for s in stream if s.strip())
            dataset=None
        else:
            dataset=load_dataset(spec['repo'],name=spec['subset'],split=spec['split'],revision=revision)
        if dataset is not None:
            rows=list(dataset)
        provenance=dict(**spec,resolved_revision=revision,data_sha256=tasks_digest(rows))
    if config.get('expected_tasks',len(rows))!=len(rows):
        raise ValueError(f'expected {config["expected_tasks"]} tasks, received {len(rows)}')
    tasks=[render(name,row,i,config['seed']) for i,row in enumerate(rows)]
    if len({t['id'] for t in tasks})!=len(tasks):
        raise ValueError('source has duplicate IDs')
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('w') as stream:
        for task in tasks:stream.write(json.dumps(task,ensure_ascii=False)+'\n')
    provenance.update(config=config,tasks=len(tasks),tasks_sha256=tasks_digest(tasks))
    output.with_suffix('.provenance.json').write_text(json.dumps(provenance,indent=2,ensure_ascii=False)+'\n')
    print(f'Prepared {len(tasks)} tasks: {output}')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--upstream',default='artifacts/upstream')
    p.add_argument('--input-jsonl',help='offline source records, before prompt rendering')
    a=p.parse_args()
    prepare(json.loads(Path(a.config).read_text()),a.output,a.upstream,a.input_jsonl)


if __name__=='__main__':
    main()
