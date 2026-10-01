"""Launch matching native-FP32 and STEPQuant servers and compare real decode.

Run with the reused serving Python environment. Outputs are local artifacts.
"""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from compare_server import compare
from stepquant.device import public_metadata
from stepquant.sglang.config import graph_batches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--state-format',choices=('stepquant4','stepquant6'),default='stepquant4')
    parser.add_argument('--state-storage',choices=('packed','byte'),default='packed')
    parser.add_argument('--fit-mode',choices=('requantize',),default='requantize')
    parser.add_argument('--devices', required=True, help='explicit CUDA device list, e.g. 0,1,2,3')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--max-running-requests',type=int,help='server capacity; defaults to the measured batch')
    parser.add_argument('--workload', choices=('long','short'), default='long')
    parser.add_argument('--mem-fraction-static',type=float,default=.85)
    parser.add_argument('--context-length',type=int,default=2048)
    parser.add_argument('--max-total-tokens',type=int,default=None)
    parser.add_argument('--chunked-prefill-size', type=int, default=8192)
    parser.add_argument('--cuda-graph-bs', type=int, nargs='+', help='explicit capture sizes; defaults to all supported tiers')
    parser.add_argument('--prompt-tokens', type=int, default=128)
    parser.add_argument('--prompt-mode', choices=('varied_tokens','matrix'),default='varied_tokens')
    parser.add_argument('--output-tokens', type=int, default=1024)
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--random-seed', type=int, default=42)
    parser.add_argument('--writeback-sm-budget', type=int, default=None)
    parser.add_argument('--writeback-chunk', type=int, default=None)
    parser.add_argument('--port', type=int, default=31082)
    parser.add_argument('--moe-config-dir', help='same SGLang MoE tile configuration for both modes')
    parser.add_argument('--attention-backend', choices=('triton','flashinfer'), default='triton',
                        help='same full-attention backend for FP32 and STEPQuant')
    parser.add_argument('--profile-steps', type=int, default=0, help='collect separate decode traces after timing; zero disables profiling')
    parser.add_argument('--profile-at-token', type=int, default=0)
    parser.add_argument('--candidate-first', action='store_true', help='reverse order for a repeat pair')
    args = parser.parse_args()
    if not 0<args.mem_fraction_static<1 or (args.max_total_tokens is not None and args.max_total_tokens<1):
        parser.error('invalid memory configuration')
    if args.prompt_tokens+args.output_tokens>args.context_length:
        parser.error('context length must hold the prompt and requested output')
    if args.profile_at_token and not args.profile_steps:
        parser.error('--profile-at-token requires --profile-steps')
    if args.profile_steps < 0 or args.profile_at_token < 0:
        parser.error('--profile-steps must be nonnegative')
    profile_tokens=max(64,args.profile_at_token+args.profile_steps+128) if args.profile_at_token else 64
    if args.profile_steps and args.prompt_tokens+profile_tokens>args.context_length:
        parser.error('context length must also hold the profiling request and its trigger margin')
    limit=args.batch if args.max_running_requests is None else args.max_running_requests
    if not 1 <= args.batch <= limit <= 512 or args.rounds < 3:
        parser.error('require 1 <= batch <= max-running-requests <= 512 and at least three rounds')
    captures=sorted(set(args.cuda_graph_bs)) if args.cuda_graph_bs else graph_batches(limit)
    if args.chunked_prefill_size<1 or any(n<1 or n>limit for n in captures) or args.batch not in captures:
        parser.error('require positive prefill size and capture sizes within capacity, including the measured batch')
    # Fixed-batch timing needs space for all generated tokens; admission itself
    # still reserves only current allocations, not maximum generation lengths.
    token_capacity=args.max_total_tokens or max(
        327680 if args.workload=='long' else 593613,
        args.batch*(args.prompt_tokens+args.output_tokens),
        math.ceil(args.batch*args.prompt_tokens/.6)+1)
    root = Path(__file__).resolve().parents[1]
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    devices = [x.strip() for x in args.devices.split(',')]
    if any(not x for x in devices) or len(set(devices)) != len(devices):
        parser.error('devices must be distinct and nonempty')
    model, plan = str(Path(args.model).resolve(strict=True)), str(Path(args.plan).resolve(strict=True))
    modes = [args.state_format, 'fp32'] if args.candidate_first else ['fp32', args.state_format]
    reports = {}
    for mode in modes:
        with socket.socket() as probe:
            if probe.connect_ex(('127.0.0.1', args.port)) == 0:
                raise RuntimeError('benchmark port is already in use')
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.devices, SGLANG_PLUGINS='stepquant',
                   STEPQUANT_TIMING_DIR=str(output / f'timing-{mode}'))
        if args.moe_config_dir:
            env['SGLANG_MOE_CONFIG_DIR']=str(Path(args.moe_config_dir).resolve(strict=True))
        env['PATH'] = str(Path(sys.executable).parent) + os.pathsep + env['PATH']
        cmd = [sys.executable, '-m', 'stepquant.sglang', '--state-format', mode, '--workload', args.workload,
               '--model-path', model, '--trust-remote-code', '--dtype', 'bfloat16', '--tp-size', str(len(devices)),
               '--attention-backend', args.attention_backend, '--context-length', str(args.context_length),
               '--chunked-prefill-size', str(args.chunked_prefill_size), '--mem-fraction-static', str(args.mem_fraction_static),
               '--max-total-tokens', str(token_capacity), '--disable-piecewise-cuda-graph',
               '--max-running-requests', str(limit),
               '--random-seed', str(args.random_seed),
               '--cuda-graph-bs', *[str(n) for n in captures],
               '--cuda-graph-max-bs', str(limit),
               '--host', '127.0.0.1', '--port', str(args.port)]
        if mode == 'fp32':
            cmd += ['--kernel-backend', 'native']
        else:
            cmd += ['--fit-mode',args.fit_mode,'--state-storage', args.state_storage, '--async-writeback', '--plan', plan]
            if args.writeback_chunk is not None:
                cmd += ['--writeback-chunk', str(args.writeback_chunk)]
            if args.writeback_sm_budget is not None:
                cmd += ['--writeback-sm-budget', str(args.writeback_sm_budget)]
        (output / f'{mode}-launch.json').write_text(json.dumps(public_metadata(dict(command=cmd, devices=devices, environment={k:env[k] for k in ('SGLANG_MOE_CONFIG_DIR','CUDA_DEVICE_MAX_CONNECTIONS') if k in env})), indent=2)+'\n')
        print('launch', mode, flush=True)
        with (output / f'{mode}.log').open('w') as log:
            server = subprocess.Popen(cmd, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                deadline = time.monotonic() + 600
                while True:
                    if server.poll() is not None:
                        raise RuntimeError(f'{mode} server failed; see {output / (mode + ".log")}')
                    try:
                        with urllib.request.urlopen(f'http://127.0.0.1:{args.port}/health', timeout=2) as response:
                            if response.status == 200:
                                break
                    except (OSError, TimeoutError):
                        pass
                    if time.monotonic() >= deadline:
                        raise TimeoutError('server readiness timed out')
                    time.sleep(2)
                report = output / f'{mode}.json'
                subprocess.run([sys.executable, str(root / 'benchmarks/bench_server.py'), '--url',
                                f'http://127.0.0.1:{args.port}', '--batch', str(args.batch),
                                '--prompt-tokens', str(args.prompt_tokens), '--prompt-mode',args.prompt_mode,'--tokenizer',model, '--output-tokens', str(args.output_tokens),
                                '--rounds', str(args.rounds), '--output', str(report)], cwd=root, check=True)
                reports[mode] = json.loads(report.read_text())
                if args.profile_steps:
                    if not args.profile_at_token:
                        payload = dict(output_dir=str(output / f'profile-{mode}'),
                                       num_steps=args.profile_steps, activities=['GPU'],
                                       profile_by_stage=True, profile_stages=['decode'])
                        request = urllib.request.Request(f'http://127.0.0.1:{args.port}/start_profile',
                            data=json.dumps(payload).encode(), headers={'Content-Type':'application/json'})
                        with urllib.request.urlopen(request, timeout=120) as response:
                            response.read()
                    # Profiling runs after all measured rounds and is never used for throughput.
                    subprocess.run([sys.executable, str(root / 'benchmarks/bench_server.py'), '--url',
                                    f'http://127.0.0.1:{args.port}', '--batch', str(args.batch),
                                    '--prompt-tokens', str(args.prompt_tokens), '--prompt-mode',args.prompt_mode,'--tokenizer',model, '--output-tokens',
                                    str(profile_tokens), '--rounds', '1', '--output',
                                    str(output / f'profile-request-{mode}.json'),
                                    *(['--profile-at-token',str(args.profile_at_token),'--profile-dir',str(output / f'profile-{mode}'),
                                       '--profile-steps',str(args.profile_steps)] if args.profile_at_token else [])], cwd=root, check=True)
                    traces=list((output / f'profile-{mode}').glob('*.trace.json*'))
                    if len(traces)<len(devices):
                        raise RuntimeError(f'profiling did not produce all TP traces: {output / f"profile-{mode}"}')
            finally:
                if server.poll() is None:
                    os.killpg(server.pid, signal.SIGTERM)
                    try:
                        server.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(server.pid, signal.SIGKILL)
                        server.wait()
            time.sleep(3)
    result = compare(reports['fp32'], reports[args.state_format], len(devices))
    (output / 'comparison.json').write_text(json.dumps(public_metadata(result), indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('batch','speedup_vs_fp32','http_speedup_vs_fp32')}), flush=True)


if __name__ == '__main__':
    main()
