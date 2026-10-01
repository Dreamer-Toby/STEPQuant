"""Compare complete state updates against FP32."""
import argparse
import json
import math
from pathlib import Path


def compare_fp32(report):
    rows=report['results']
    def key(row):return row['architecture'],row['batch'],row['heads']
    index={(key(r),r['format']):r for r in rows}
    if len(index)!=len(rows):raise ValueError('duplicate benchmark cases')
    cases=[]
    for row in rows:
        if row['format']=='fp32':continue
        base=index.get((key(row),'fp32'))
        if base is None:raise ValueError(f'missing matched FP32 baseline: {key(row)}')
        if min(base['step_us'],row['step_us'],base['bytes_per_slot'],row['bytes_per_slot'])<=0:
            raise ValueError('timings and state sizes must be positive')
        cases.append(dict(architecture=row['architecture'],format=row['format'],batch=row['batch'],heads=row['heads'],
                          fp32_step_us=base['step_us'],quantized_step_us=row['step_us'],
                          speedup_vs_fp32=base['step_us']/row['step_us'],
                          latency_ratio_vs_fp32=row['step_us']/base['step_us'],
                          extra_step_us=row['step_us']-base['step_us'],
                          state_compression=base['bytes_per_slot']/row['bytes_per_slot'],
                          fit_or_encode_us=row['fit_us']))
    if not cases:raise ValueError('no non-FP32 cases to compare')
    summary={}
    for name in sorted({r['format'] for r in cases}):
        group=[r for r in cases if r['format']==name]
        summary[name]=dict(cases=len(group),
            speedup_vs_fp32_geomean=math.exp(sum(math.log(r['speedup_vs_fp32']) for r in group)/len(group)),
            speedup_vs_fp32_min=min(r['speedup_vs_fp32'] for r in group),
            speedup_vs_fp32_max=max(r['speedup_vs_fp32'] for r in group),
            faster_than_fp32_cases=sum(r['speedup_vs_fp32']>1 for r in group),
            state_compression_geomean=math.exp(sum(math.log(r['state_compression']) for r in group)/len(group)))
    return dict(primary_baseline='fp32',baseline_implementation='STEPQuant floating pool FP32 fused kernel; not upstream native SGLang',
                scope=report['scope'],device=report['device'],case_count=len(cases),
                metric='FP32 step time / quantized step time; >1 faster, <1 slower. Complete quantized step includes fitting and packing.',
                fit_note='fit_or_encode_us is a separate diagnostic; FP32 has no quantization fit, so no fitting speedup against FP32 is defined.',
                summary=summary,cases=cases)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--after',required=True);p.add_argument('--output',required=True)
    a=p.parse_args()
    after=json.loads(Path(a.after).read_text())
    report=compare_fp32(after)
    Path(a.output).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='cases'},indent=2))


if __name__=='__main__':main()
