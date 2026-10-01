"""Compare fixed-batch real-model CUDA decode intervals with an FP32 baseline."""
import argparse
import json
import math
import statistics
from pathlib import Path


def summarize(report,expected_ranks=4):
    marker=report['before'][0]['stepquant_decode_timing']
    end=report.get('after',[{}])[0].get('stepquant_decode_timing',{}).get('last_segment',math.inf)
    directory=Path(marker['path']).parent
    ranks={}
    for path in directory.glob('rank-*.json'):
        value=json.loads(path.read_text())
        if value['dropped_events']:raise ValueError('timing event limit was reached')
        if value['rank'] in ranks:raise ValueError('duplicate TP rank')
        ranks[value['rank']]=value
    if set(ranks)!=set(range(expected_ranks)):raise ValueError('missing or unexpected TP rank files')
    group_sets=[]
    reference=ranks[0]
    for value in ranks.values():
        for key in ('state_format','state_storage','fit_mode','kernel_backend','async_writeback','kernel_sha256','writeback_sms','writeback_chunk','moe_config_sha256','versions'):
            if value.get(key)!=reference.get(key):raise ValueError(f'TP metadata mismatch: {key}')
    for rank in sorted(ranks):
        value=ranks[rank]
        groups={}
        for row in value['intervals']:
            if marker['last_segment'] < row['segment'] <= end:
                if row['batch']!=report['batch']:raise ValueError('batch changed during timed decode')
                dt=row['start_to_start_ms']
                if not math.isfinite(dt) or dt<=0:raise ValueError('invalid interval')
                groups.setdefault(row['segment'],[]).append(dt)
        if len(groups)!=len(report['rounds']):raise ValueError('unexpected number of timed request rounds')
        if any(len(v)<report['output_tokens']-2 for v in groups.values()):
            raise ValueError('incomplete fixed-batch decode trace')
        group_sets.append(groups)
    segments=sorted(group_sets[0])
    if any(sorted(g)!=segments for g in group_sets):raise ValueError('TP segment mismatch')
    rounds=[]
    for segment in segments:
        sizes=[len(g[segment]) for g in group_sets]
        if len(set(sizes))!=1:raise ValueError('TP interval count mismatch')
        means=[statistics.mean(g[segment]) for g in group_sets]
        rounds.append(dict(intervals=sizes[0],rank_mean_ms=means,slowest_rank_mean_ms=max(means)))
    first=ranks[0]
    profile=first.get('writeback_profiles',{}).get(str(report['batch']),{})
    return dict(fit_mode=first.get('fit_mode','requantize'),versions=first.get('versions',{}),moe_config_sha256=first.get('moe_config_sha256',{}),state_format=first['state_format'],state_storage=first.get('state_storage','packed'),kernel_backend=first['kernel_backend'],
                async_writeback=first['async_writeback'],rounds=rounds,
                writeback_sms=profile.get('actual_sms',first.get('writeback_sms')),
                writeback_chunk=profile.get('chunk',first.get('writeback_chunk')),
                median_decode_step_ms=statistics.median(r['slowest_rank_mean_ms'] for r in rounds),
                kernel_sha256=first.get('kernel_sha256'),
                median_http_wall_seconds=report['median_wall_seconds'])


def compare(baseline,candidate,expected_ranks=4):
    if baseline.get('prompt_mode','varied_tokens')!=candidate.get('prompt_mode','varied_tokens'):
        raise ValueError('incompatible prompt mode')
    if baseline.get('prompt_sha256')!=candidate.get('prompt_sha256'):
        raise ValueError('incompatible or missing prompt token hash')
    for key in ('batch','prompt_tokens','output_tokens'):
        if baseline[key]!=candidate[key]:raise ValueError(f'incompatible {key}')
    if len(baseline['rounds']) < 3 or len(baseline['rounds']) != len(candidate['rounds']):
        raise ValueError('comparison requires matching counts of at least three rounds')
    checked='configuration' in baseline and 'configuration' in candidate
    if not checked or any(v is None for report in (baseline,candidate) for v in report['configuration'].values()):
        raise ValueError('missing serving configuration')
    configurations=[]
    for report in (baseline,candidate):
        config=dict(report['configuration'])
        # Local model aliases may name the same checkpoint directory.
        model=Path(config.get('model_path',''))
        if model.is_dir():config['model_path']=str(model.resolve(strict=True))
        configurations.append(config)
    if configurations[0]!=configurations[1]:
        raise ValueError('incompatible serving configurations')
    a,b=summarize(baseline,expected_ranks),summarize(candidate,expected_ranks)
    if a['versions']!=b['versions']:
        raise ValueError('incompatible runtime versions')
    if a['moe_config_sha256']!=b['moe_config_sha256']:
        raise ValueError('incompatible MoE configurations')
    if a['state_format']!='fp32' or a['kernel_backend']!='native' or a['async_writeback']:
        raise ValueError('baseline must be native FP32 without background writeback')
    return dict(scope='Real-model fixed-batch CUDA decode start-to-start timing; synthetic prompts, not accuracy',
                protocol='Per-round mean on the slowest TP rank; median across rounds. Prefill/warmup excluded; step gaps included.',
                configuration_equality_verified=checked,batch=baseline['batch'],
                prompt_tokens=baseline['prompt_tokens'],output_tokens=baseline['output_tokens'],
                prompt_mode=baseline.get('prompt_mode','varied_tokens'),prompt_sha256=baseline.get('prompt_sha256'),
                baseline=a,candidate=b,speedup_vs_fp32=a['median_decode_step_ms']/b['median_decode_step_ms'],
                http_speedup_vs_fp32=a['median_http_wall_seconds']/b['median_http_wall_seconds'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',required=True);p.add_argument('--candidate',required=True)
    p.add_argument('--expected-ranks',type=int,default=4);p.add_argument('--output',required=True)
    args=p.parse_args()
    report=compare(json.loads(Path(args.baseline).read_text()),json.loads(Path(args.candidate).read_text()),args.expected_ranks)
    Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('batch','speedup_vs_fp32','http_speedup_vs_fp32')}))


if __name__=='__main__':main()
