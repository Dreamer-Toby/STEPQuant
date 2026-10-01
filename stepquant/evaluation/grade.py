"""Grade frozen responses; code/IF/math competition checks call pinned upstreams."""
import argparse
import ast
import os
from types import SimpleNamespace
from collections import defaultdict
import json
from pathlib import Path
import re
import subprocess
import sys
from .pipeline import read_results, digest
from .upstream import use, LOCK
from .data import tasks_digest
from importlib.metadata import version, PackageNotFoundError
from stepquant.reproducibility import source_identity


def final_answer(record):
    # SGLang's reasoning parser returns reasoning_content separately. Never use
    # truncated reasoning as an answer if no final content was produced.
    text=record.get('answer','')
    if '<think>' in text and '</think>' not in text:
        return ''
    return text.rsplit('</think>',1)[-1].strip()


def code_answer(text):
    blocks=re.findall(r'```(?:python|py)?\s*\n(.*?)```',text,re.S)
    return blocks[-1].strip() if blocks else text.strip()


def letter_answer(text):
    matches=re.findall(r'(?:correct answer is|answer\s*:)\s*\(?([A-Z])\)?',text,re.I)
    if matches:
        return matches[-1].upper()
    match=re.fullmatch(r'\s*\(?([A-Z])\)?[.\s]*',text)
    return match[1] if match else None


def grade(config,tasks,run_dir,upstream='artifacts/upstream',processes=4):
    root=Path(run_dir)
    manifest=json.loads((root/'manifest.json').read_text())
    if manifest['tasks_sha256']!=tasks_digest(tasks) or manifest['config']!=config:
        raise ValueError('grading inputs do not match the generation manifest')
    successful=[r for r in read_results(root/'responses.jsonl') if 'error' not in r]
    records={r['sample_id']:r for r in successful}
    if len(successful)!=len(records):
        raise ValueError('duplicate successful samples')
    expected={f"{t['id']}:{s}" for t in tasks for s in range(config['samples'])}
    if set(records)!=expected or any(r['run_hash']!=manifest['run_hash'] for r in records.values()):
        raise ValueError('missing/unexpected samples or run identity mismatch; finish generation first')
    kind=config['grader']
    ordered=[[records[f"{t['id']}:{s}"] for s in range(config['samples'])] for t in tasks]
    details=[]
    if kind=='evalplus':
        source=use('evalplus',upstream)
        env=dict(os.environ,PYTHONPATH=str(source.resolve())+os.pathsep+os.environ.get('PYTHONPATH',''))
        dataset='humaneval' if config['name']=='humaneval_plus' else 'mbpp'
        exported=root/'evalplus.samples.jsonl'
        exported.write_text(''.join(json.dumps(dict(task_id=r['task_id'],solution=code_answer(final_answer(r))))+'\n' for rows in ordered for r in rows))
        # Upstream sanitizer and evaluator, not a string-matching substitute.
        subprocess.run([sys.executable,'-m','evalplus.sanitize','--samples',str(exported)],check=True,env=env)
        sanitized=exported.with_name(exported.stem+'-sanitized.jsonl')
        subprocess.run([sys.executable,'-m','evalplus.evaluate','--dataset',dataset,'--samples',str(sanitized),'--parallel',str(processes)],check=True,env=env)
        result_path=sanitized.with_name(sanitized.stem+'.eval_results.json')
        official=json.loads(result_path.read_text())
        scored=[r for rows in official['eval'].values() for r in rows]
        result=dict(official_results=str(result_path),
                    base_pass_at_1=sum(r['base_status']=='pass' for r in scored)/len(scored),
                    plus_pass_at_1=sum(r['base_status']=='pass' and r['plus_status']=='pass' for r in scored)/len(scored))
    elif kind=='lcb':
        use('LiveCodeBench',upstream)
        from lcb_runner.benchmarks.code_generation import CodeGenerationProblem
        from lcb_runner.evaluation import codegen_metrics
        samples=[CodeGenerationProblem(**t['source']).get_evaluation_sample() for t in tasks]
        outputs=[[code_answer(final_answer(r)) for r in rows] for rows in ordered]
        metrics,raw,metadata=codegen_metrics(samples,outputs,k_list=[1,5],num_process_evaluate=processes,timeout=6)
        result=dict(metrics=metrics,results=raw,metadata=metadata)
    else:
        if kind=='ifbench':
            use('IFBench',upstream)
            import evaluation_lib
            from langdetect import DetectorFactory
            DetectorFactory.seed=config.get('seed',0)
        if kind=='matharena':
            use('matharena',upstream)
            from matharena.parser import extract_answer,parse_answer,check_answers
        if kind=='gpqa':
            source=use('gpqa',upstream)
            # Load the original pure parsing method without importing its API
            # clients/retrieval stack. Its function body is compiled unchanged.
            tree=ast.parse((source/'baselines/closed_book.py').read_text())
            cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='AnswerPredictor')
            function=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='parse_sampled_answer')
            function.decorator_list=[]
            namespace=dict(re=re,AnswerPredictor=SimpleNamespace(LETTER_TO_INDEX={c:i for i,c in enumerate('ABCD')}))
            exec(compile(ast.Module(body=[function],type_ignores=[]),str(source/'baselines/closed_book.py'),'exec'),namespace)
            gpqa_parse=namespace['parse_sampled_answer']
        if kind=='math500':
            from math_verify import parse,verify
        for task,rows in zip(tasks,ordered):
            for record in rows:
                text=final_answer(record)
                if kind=='ifbench':
                    inp=evaluation_lib.InputExample(**{k:task['source'][k] for k in ('key','instruction_id_list','prompt','kwargs')})
                    strict=evaluation_lib.test_instruction_following_strict(inp,{inp.prompt:text})
                    loose=evaluation_lib.test_instruction_following_loose(inp,{inp.prompt:text})
                    details.append(dict(sample_id=record['sample_id'],correct=strict.follow_all_instructions,
                                        loose=loose.follow_all_instructions,instructions=strict.follow_instruction_list))
                    continue
                elif kind=='matharena':
                    extracted,_=extract_answer(text,strict_parsing=True)
                    gold,_=parse_answer(task['answer'])
                    correct=bool(check_answers(extracted,gold)) if extracted is not None else False
                elif kind=='math500':
                    correct=bool(verify(parse('\\boxed{'+task['answer']+'}'),parse(text)))
                elif kind=='gpqa':
                    correct=gpqa_parse(text)==task['answer']
                elif kind=='generated_exact_match':
                    correct=(text.strip()==task['answer']) if config['name']=='lambada' else letter_answer(text)==task['answer']
                else:
                    raise ValueError(kind)
                details.append(dict(sample_id=record['sample_id'],correct=correct))
        result=dict(accuracy=sum(d['correct'] for d in details)/len(details),samples=len(details),details=details)
        if kind=='ifbench':
            result['loose_accuracy']=sum(d['loose'] for d in details)/len(details)
            result['strict_instruction_accuracy']=sum(sum(d['instructions']) for d in details)/sum(len(d['instructions']) for d in details)
    versions={}
    for name in ('math-verify','sympy','antlr4-python3-runtime','numpy'):
        try:versions[name]=version(name)
        except PackageNotFoundError:pass
    result.update(run_hash=manifest['run_hash'],grader=kind,grader_versions=versions,grading_source_sha256=source_identity(),upstream_lock=json.loads(LOCK.read_text()),
                  truncated_samples=sum(r.get('finish_reason')=='length' for r in records.values()))
    (root/'scores.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,default=str)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('details','results','metadata','upstream_lock','grading_source_sha256')}))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True)
    p.add_argument('--tasks',required=True)
    p.add_argument('--run',required=True)
    p.add_argument('--upstream',default='artifacts/upstream')
    p.add_argument('--processes',type=int,default=4)
    a=p.parse_args()
    grade(json.loads(Path(a.config).read_text()),[json.loads(s) for s in Path(a.tasks).read_text().splitlines()],a.run,a.upstream,a.processes)


if __name__=='__main__':
    main()
