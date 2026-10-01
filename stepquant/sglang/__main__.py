"""Launch the pinned SGLang server with the STEPQuant plugin in every TP worker."""
import argparse
import os
from pathlib import Path
from importlib.metadata import version,entry_points
from .config import request_limit, graph_batches


def main():
    parser=argparse.ArgumentParser(description=__doc__,epilog='Other options are forwarded to SGLang 0.5.12.')
    parser.add_argument('--plan')
    parser.add_argument('--state-format',default='stepquant')
    parser.add_argument('--state-storage',choices=('packed','byte'),default='packed',help='byte: one signed byte per STEPQuant code; uses more memory than packed')
    parser.add_argument('--moe-config-dir',help='capacity-neutral SGLang MoE tuning templates')
    parser.add_argument('--fit-mode',choices=('requantize',),default='requantize',help='one weighted fit followed by final requantization')
    parser.add_argument('--workload',choices=('long','short'),default='long',help='continuous admission profile (default: long, cap 64)')
    parser.add_argument('--max-running-requests',type=int,help='request cap, 1..512; defaults: long 64, short 256')
    parser.add_argument('--kv-watermark',type=float,default=.6)
    parser.add_argument('--async-writeback',action=argparse.BooleanOptionalAction,default=None,
                        help='background fitting/writeback; enabled by default for STEPQuant')
    parser.add_argument('--writeback-sm-budget',type=int,default=None,help='background SM budget (STEPQuant6: per graph batch; other formats: GDN 24, KDA 32); 0 uses an unrestricted stream')
    parser.add_argument('--writeback-chunk',type=int,default=None,help='requests per captured writeback chunk (STEPQuant6: per graph batch; other formats: 64); 0 uses the whole batch')
    parser.add_argument('--kernel-backend',choices=('auto','native'),default='auto',
                        help='auto: fused state kernels; native: upstream FP32 baseline')
    args,remaining=parser.parse_known_args()
    try:
        limit=request_limit(args.workload,args.max_running_requests)
    except ValueError as error:
        parser.error(str(error))
    if version('sglang')!='0.5.12':
        raise RuntimeError('install the sglang==0.5.12 optional dependency')
    if not any(ep.name=='stepquant' for ep in entry_points(group='sglang.srt.plugins')):
        raise RuntimeError('install STEPQuant with pip install -e . so worker processes can load its plugin')
    from stepquant.formats import get_format
    spec=get_format(args.state_format)
    if args.async_writeback is None:
        args.async_writeback=spec.kind in ('stepquant',)
    if args.kernel_backend=='native' and (spec.name!='fp32' or args.async_writeback):
        parser.error('native backend requires FP32 without async writeback')
    if args.state_storage=='byte' and (spec.kind not in ('stepquant',) or not args.async_writeback):
        parser.error('byte storage requires STEPQuant with async writeback')
    os.environ['STEPQUANT_FIT_MODE']=args.fit_mode
    os.environ['STEPQUANT_STATE_STORAGE']=args.state_storage
    os.environ['STEPQUANT_KERNEL_BACKEND']=args.kernel_backend
    if spec.calibrated and not args.plan:
        parser.error(f'{spec.name} requires a calibrated --plan')
    if args.plan:
        os.environ['STEPQUANT_PLAN']=str(Path(args.plan).resolve(strict=True))
    else:
        os.environ.pop('STEPQUANT_PLAN',None)
    os.environ['STEPQUANT_STATE_FORMAT']=spec.name
    if not 0 < args.kv_watermark < 1:
        parser.error('--kv-watermark must be between 0 and 1')
    os.environ['STEPQUANT_DYNAMIC_ADMISSION']='1' if args.workload else '0'
    os.environ['STEPQUANT_KV_WATERMARK']=str(args.kv_watermark)
    os.environ['STEPQUANT_ASYNC_WRITEBACK']='1' if args.async_writeback else '0'
    if ((args.writeback_chunk is not None and args.writeback_chunk<0)
            or (args.writeback_sm_budget is not None and args.writeback_sm_budget<0)):
        parser.error('writeback SM budget and chunk must be nonnegative')
    chunk=args.writeback_chunk if args.writeback_chunk is not None else ('auto' if spec.name=='stepquant6' else 64)
    os.environ['STEPQUANT_WRITEBACK_CHUNK']=str(chunk)
    templates=args.moe_config_dir or os.environ.get('SGLANG_MOE_CONFIG_DIR')
    if templates:
        import json
        from stepquant.moe import materialize_configs
        from stepquant.device import tuning_identity
        os.environ['STEPQUANT_MOE_TEMPLATE_SHA256']=json.dumps(tuning_identity(templates),sort_keys=True)
        generated=materialize_configs(templates)
        if generated is None:
            os.environ.pop('SGLANG_MOE_CONFIG_DIR',None)
        else:
            os.environ['SGLANG_MOE_CONFIG_DIR']=str(generated.resolve())
    # Reused environments may contain unrelated serving plugins.
    plugins=os.environ.setdefault('SGLANG_PLUGINS','stepquant')
    if plugins and 'stepquant' not in {p.strip() for p in plugins.split(',')}:
        raise ValueError('SGLANG_PLUGINS whitelist excludes stepquant')
    from sglang.srt.plugins import load_plugins
    load_plugins()
    from .plugin import verify_hooks
    verify_hooks()
    from sglang.launch_server import run_server
    from sglang.srt.server_args import prepare_server_args
    from sglang.srt.utils import kill_process_tree
    # Evaluation disables radix caching. Paper five-slot serving memory uses
    # prefix-state accounting; these evaluation defaults remain unchanged.
    defaults=['--disable-radix-cache','--mamba-scheduler-strategy','no_buffer','--linear-attn-backend','triton',
              '--linear-attn-decode-backend','triton','--linear-attn-prefill-backend','triton']
    if args.kernel_backend=='native':defaults+=['--mamba-ssm-dtype','float32']
    defaults += ['--max-running-requests',str(limit)]
    # Capture every requested tier directly instead of padding B32 to B64.
    options={arg.split('=',1)[0] for arg in remaining}
    if not options.intersection({'--cuda-graph-bs','--cuda-graph-max-bs'}):
        defaults += ['--cuda-graph-bs',*[str(n) for n in graph_batches(limit)],'--cuda-graph-max-bs',str(limit)]
    server_args=prepare_server_args(defaults+remaining)
    if args.kernel_backend=='native' and server_args.mamba_ssm_dtype!='float32':
        parser.error('native FP32 comparison requires --mamba-ssm-dtype float32')
    import json
    config=json.loads((Path(server_args.model_path)/'config.json').read_text())
    model_type=config.get('text_config',config).get('model_type',config.get('model_type'))
    architectures={'qwen3_5':'gdn','qwen3_5_text':'gdn','kimi_linear':'kda'}
    if model_type not in architectures:
        parser.error(f'unsupported architecture {model_type}')
    architecture=architectures[model_type]
    os.environ['STEPQUANT_ARCHITECTURE']=architecture
    default_budget=24 if architecture=="gdn" else 32
    budget=args.writeback_sm_budget if args.writeback_sm_budget is not None else ('auto' if spec.name=='stepquant6' else default_budget)
    os.environ['STEPQUANT_WRITEBACK_SMS']=str(budget)
    try:
        run_server(server_args)
    finally:
        kill_process_tree(os.getpid(),include_parent=False)


if __name__=='__main__':
    main()
