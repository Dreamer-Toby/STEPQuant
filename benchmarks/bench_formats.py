"""CUDA-graph microbenchmarks: recurrent update and codec fitting, all formats.

FP32 is always included as the primary baseline on the same otherwise idle GPU.
These measurements exclude Python/HTTP overhead and do not imply model throughput.
"""
import argparse
import json
from pathlib import Path
import statistics
import hashlib
import torch
import triton
from stepquant.device import device_label, public_metadata
from stepquant.formats import BASELINE_NAMES, FORMAT_NAMES, FormatPlan, get_format
from stepquant.quantization import QuantizationPlan
from stepquant.kernels.pool import PackedStatePool
from stepquant.kernels.format_pool import FormatStatePool


def make_pool(name, architecture, heads, slots, writeback_stream=None, calibration=None,storage='packed'):
    spec=get_format(name)
    if spec.kind=='stepquant':
        validate_calibration(name,calibration)
        bits=calibration['bits'].to('cuda');impact=calibration['impact'].to('cuda')
        plan=QuantizationPlan(architecture,bits,impact,calibration.get('value_group_size',32))
        return PackedStatePool(plan,slots,128,writeback_stream=writeback_stream,storage=storage)
    bits=torch.full((heads,128),spec.bits,device='cuda',dtype=torch.int32)
    impact=torch.ones((heads,128),device='cuda')
    plan=FormatPlan(architecture,bits,impact)
    return FormatStatePool(plan,slots,128,spec)


def validate_calibration(name, calibration):
    if calibration is None:
        raise ValueError(f'{name} requires --plan; no synthetic precision map is used')
    allowed=(2,4,6,8,16) if get_format(name).bits==4 else (4,6,8,16)
    if not torch.isin(calibration['bits'],torch.tensor(allowed,device=calibration['bits'].device)).all():
        raise ValueError(f'{name} plan contains precision outside {allowed}')


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--formats',nargs='+',choices=FORMAT_NAMES,
                   help='default: paper baselines without --plan; FP32 and the calibrated format with --plan')
    p.add_argument('--architectures',nargs='+',choices=['gdn','kda'],help='default: both; with --plan use its architecture')
    p.add_argument('--batch-sizes',nargs='+',type=int,default=[1,16,64,256])
    p.add_argument('--heads',type=int,help='default: one TP4 shard, GDN=12, KDA=8')
    p.add_argument('--plan',help='STEPQuant calibration plan; requires one architecture and matching STEPQuant/FP32 formats')
    p.add_argument('--layer',help='calibration layer name; defaults to the first layer')
    p.add_argument('--tp-size',type=int,default=4)
    p.add_argument('--tp-rank',type=int,default=0)
    p.add_argument('--raw-inputs',action='store_true',help='include Q/K normalization and raw model gate transforms')
    p.add_argument('--rep',type=int,default=30)
    p.add_argument('--rounds',type=int,default=3)
    p.add_argument('--state-storage',choices=('packed','byte'),default='packed')
    p.add_argument('--async-writeback',action='store_true',help='compact STEPQuant path; timings include completed writeback')
    p.add_argument('--output',required=True)
    a=p.parse_args(argv)
    calibration=None;plan_metadata={}
    if a.plan:
        artifact=torch.load(a.plan,map_location='cpu',weights_only=True)
        if artifact.get('format_version')!=2:p.error('expected a version 2 STEPQuant calibration plan')
        plan_format=artifact.get('state_format')
        if plan_format=='stepquant':plan_format='stepquant'+str(artifact['settings']['nominal_bits'])
        a.formats=a.formats or ['fp32',plan_format]
        if plan_format not in ('stepquant4','stepquant6') or set(a.formats)-{'fp32',plan_format}:
            p.error('expected a calibration plan matching the requested format')
        layer=a.layer or next(iter(artifact['plans']))
        calibration=dict(artifact['plans'][layer])
        a.architectures=a.architectures or [calibration['architecture']]
        if len(a.architectures)!=1 or a.heads is not None:
            p.error('--plan requires one architecture and no --heads')
        try:
            validate_calibration(plan_format,calibration)
        except ValueError as error:
            p.error(str(error))
        if calibration['architecture']!=a.architectures[0] or a.tp_size<1 or not 0<=a.tp_rank<a.tp_size or calibration['bits'].shape[0]%a.tp_size:
            p.error('plan architecture or TP shard mismatch')
        a.heads=calibration['bits'].shape[0]//a.tp_size
        shard=slice(a.tp_rank*a.heads,(a.tp_rank+1)*a.heads)
        calibration.update(bits=calibration['bits'][shard].contiguous(),impact=calibration['impact'][shard].contiguous())
        if calibration['bits'].shape[1]!=128:p.error('benchmark currently requires key_dim=128')
        plan_metadata=dict(plan_sha256=hashlib.sha256(Path(a.plan).read_bytes()).hexdigest(),layer=layer,tp_size=a.tp_size,tp_rank=a.tp_rank)
    else:
        a.formats=a.formats or list(BASELINE_NAMES)
        a.architectures=a.architectures or ['gdn','kda']
        if any(get_format(name).calibrated for name in a.formats):
            p.error('STEPQuant benchmarks require --plan')
    if a.state_storage=='byte' and (not a.async_writeback or set(a.formats)-{'fp32','stepquant4','stepquant6'}):p.error('byte carriers require asynchronous STEPQuant')
    return a,calibration,plan_metadata


@torch.inference_mode()
def main():
    a,calibration,plan_metadata=parse_args()
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    stream=torch.cuda.Stream() if a.async_writeback else None
    a.formats=["fp32"]+[name for name in dict.fromkeys(a.formats) if name!="fp32"]
    report=dict(raw_inputs=a.raw_inputs,state_storage=a.state_storage,calibration=plan_metadata,async_writeback=a.async_writeback,primary_baseline="fp32",fp32_mode="auto",device=device_label(torch.cuda.get_device_name()),torch=torch.__version__,triton=triton.__version__,
                scope='CUDA-graph per-layer microbenchmark; STEPQuant uses calibrated precision maps and row-impact factors; one weighted fit per STEPQuant state',
                kernel_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path("stepquant/kernels").glob("*.py"))},results=[])
    for arch in a.architectures:
        for name in a.formats:
            for batch in a.batch_sizes:
                torch.manual_seed(42)
                h=a.heads or (12 if arch=='gdn' else 8)
                pool=make_pool(name,arch,h,batch+1,stream,calibration,a.state_storage)
                slots=torch.arange(1,batch+1,device='cuda')
                state=torch.randn(batch,h,128,128,device='cuda')*.1
                pool.encode(state,slots)
                q,k=[torch.nn.functional.normalize(torch.randn(batch,h,128,device='cuda'),dim=-1) for _ in range(2)]
                v=torch.randn_like(q);g=-torch.rand_like(q)*.1 if arch=='kda' else -torch.rand(batch,h,device='cuda')*.1
                beta=torch.rand(batch,h,device='cuda')
                options={}
                if a.raw_inputs:
                    q,k=[torch.randn(batch,h,128,device='cuda',dtype=torch.bfloat16) for _ in range(2)]
                    v=torch.randn_like(q)
                    g=torch.randn_like(q) if arch=='kda' else torch.randn(batch,h,device='cuda')
                    beta=torch.randn(batch,h,device='cuda')
                    options=dict(normalize=True,scale=128**-.5,a_log=torch.randn(h,device='cuda'),
                                 dt_bias=torch.randn(h*128 if arch=='kda' else h,device='cuda'))
                def step():
                    pool.step(q,k,v,g,beta,slots,**options)
                    pool.wait()  # Include background completion in isolated layer timing.
                fit=lambda:pool.encode(state,slots)
                # Warm compilation outside timing; fixed inputs to fitting, recurrent feedback to step.
                step();fit();torch.cuda.synchronize()
                times={}
                for label,fn in [('step_us',step),('fit_us',fit)]:
                    samples=[triton.testing.do_bench_cudagraph(fn,rep=a.rep)*1000 for _ in range(a.rounds)]
                    times[label]=statistics.median(samples)
                    times[label+'_samples']=samples
                row=dict(architecture=arch,format=name,batch=batch,heads=h,async_writeback=get_format(name).kind in ('stepquant',) and stream is not None,bytes_per_slot=pool.page_bytes,**times)
                report['results'].append(row)
                Path(a.output).write_text(json.dumps(public_metadata(report),indent=2)+'\n')
                print(f'{arch} {name:26} B={batch:3} step={times["step_us"]:.2f} us fit={times["fit_us"]:.2f} us',flush=True)


if __name__=='__main__':main()
