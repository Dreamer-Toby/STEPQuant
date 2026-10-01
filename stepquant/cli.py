"""Calibrate, save a precision map, and run real greedy autoregressive decoding."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import torch
from .calibration import calibrate
from .reproducibility import checkpoint_identity, source_identity, file_sha256, package_versions


def fingerprint(path):
    root = Path(path)
    digest = hashlib.sha256()
    for name in ('config.json', 'model.safetensors.index.json'):
        if (root / name).exists():
            digest.update((root / name).read_bytes())
    return digest.hexdigest()


def save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(value, path)


def calibration_segments(tokenizer, args):
    if args.text_file:
        text = Path(args.text_file).read_text()
    else:
        from datasets import load_dataset
        dataset = load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1', split='train',revision=args.data_revision)
        text = '\n\n'.join(dataset['text'])
    ids = tokenizer(text, return_tensors='pt', add_special_tokens=False).input_ids[0]
    if len(ids) < args.sequence_length:
        raise ValueError('calibration text is shorter than --sequence-length; provide more text')
    rng = torch.Generator().manual_seed(args.seed)
    for _ in range(args.segments):
        start = int(torch.randint(len(ids) - args.sequence_length + 1, (), generator=rng))
        yield ids[start:start + args.sequence_length][None]


def fit_statistics(statistics, args):
    result=calibrate(statistics, nominal_bits=args.bits, pivots=args.pivots,
                     horizon=args.horizon,
                     value_group_size=args.value_group_size, device=args.fit_device)
    result['state_format']=args.state_format
    return result


@torch.inference_mode()
def run_model(args):
    from .models import load_model, generate, last_logits_kwargs
    from .adapters import ModelAdapter
    model, tokenizer = load_model(args.model, args.max_memory_gib)
    artifact = None
    report = {'model': str(Path(args.model).resolve()), 'command': args.command}
    if args.command in ('calibrate', 'smoke'):
        adapter = ModelAdapter(model, collect=True, sample_every=args.sample_every, max_snapshots=args.max_snapshots)
        device = model.get_input_embeddings().weight.device
        start = time.monotonic()
        token_hash = hashlib.sha256()
        segment_count = 0
        for i, ids in enumerate(calibration_segments(tokenizer, args)):
            token_hash.update(ids.numpy().tobytes())
            segment_count += 1
            # Independent segments: cache/state starts at zero for every segment.
            output = model(input_ids=ids.to(device), use_cache=True, **last_logits_kwargs(model))
            del output
            print(f'Calibration segment {i+1}/{args.segments}', flush=True)
        statistics = adapter.export_statistics()
        adapter.close()
        data = dict(protocol=args.calibration_protocol, revision=args.data_revision if not args.text_file else None,
                    source=args.text_file or 'Salesforce/wikitext:wikitext-2-raw-v1:train',
                    source_sha256=file_sha256(args.text_file) if args.text_file else None,
                    segments=segment_count, sequence_length=args.sequence_length, seed=args.seed,
                    sample_every=args.sample_every, max_snapshots=args.max_snapshots,
                    token_ids_sha256=token_hash.hexdigest())
        provenance = dict(checkpoint_fingerprint=fingerprint(args.model),
                          checkpoint_identity=checkpoint_identity(args.model), data=data,
                          calibration_versions=package_versions(),
                          calibration_source_sha256=source_identity())
        if args.statistics_output:
            save(dict(statistics_version=1, statistics=statistics, **provenance), args.statistics_output)
        artifact = fit_statistics(statistics, args)
        artifact.update(provenance)
        save(artifact, args.output)
        report.update(calibration_seconds=time.monotonic()-start, artifact=args.output, settings=artifact['settings'])
        print(f'Saved calibration: {args.output}', flush=True)
        if args.command == 'calibrate':
            print(json.dumps(report, indent=2))
            return
    elif args.plan:
        artifact = torch.load(args.plan, map_location='cpu', weights_only=True)
        if artifact.get('format_version') != 2 or artifact.get('checkpoint_fingerprint') != fingerprint(args.model):
            raise ValueError('calibration artifact version/checkpoint mismatch; recalibrate this checkpoint')
    if args.command == 'smoke':
        adapter = ModelAdapter(model)
        baseline, baseline_ids = generate(model, tokenizer, args.prompt, args.max_new_tokens)
        adapter.close()
        report.update(fp32_text=baseline, fp32_token_ids=baseline_ids)
    if artifact and artifact.get('state_format','stepquant') != 'stepquant':
        raise ValueError('these formats require the SGLang launcher')
    adapter = ModelAdapter(model, artifact=artifact)
    start = time.monotonic()
    response, tokens = generate(model, tokenizer, args.prompt, args.max_new_tokens)
    report.update(text=response, token_ids=tokens, generation_seconds=time.monotonic()-start,
                  recurrent_state_bytes=sum(adapter.last_bytes.values()), recurrent_layers=len(adapter.layer_names),
                  state_format='stepquant' if artifact else 'fp32')
    adapter.close()
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for command in ('calibrate', 'generate', 'smoke'):
        p = sub.add_parser(command)
        p.add_argument('--model', required=True)
        p.add_argument('--max-memory-gib', type=int,
                       help='optional per-GPU loading budget in GiB; default: automatic placement')
        if command != 'calibrate':
            p.add_argument('--prompt', default='The capital of France is')
            p.add_argument('--max-new-tokens', type=int, default=32)
            p.add_argument('--report')
        if command == 'generate':
            p.add_argument('--plan', help='omit for FP32 recurrent state baseline')
        else:
            p.add_argument('--calibration-protocol',choices=('wikitext',),default='wikitext')
            p.add_argument('--data-revision',default='b08601e04326c79dfdd32d625aee71d232d685c3')
            p.add_argument('--text-file', help='local UTF-8 calibration text; default WikiText-2 train')
            p.add_argument('--segments', type=int, default=32)
            p.add_argument('--sequence-length', type=int, default=2048)
            p.add_argument('--seed', type=int, default=0)
            p.add_argument('--sample-every', type=int, default=8)
            p.add_argument('--max-snapshots', type=int, default=64)
            p.add_argument('--statistics-output')
            p.add_argument('--output', required=True)
            add_fit_arguments(p)
    p = sub.add_parser('fit', help='fit a new precision map without reloading model weights')
    p.add_argument('--statistics', required=True)
    p.add_argument('--model', required=True, help='checkpoint directory for artifact identity')
    p.add_argument('--output', required=True)
    p.add_argument('--allow-legacy-statistics',action='store_true',help='development only: statistics without checkpoint/data provenance')
    add_fit_arguments(p)
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.command == 'fit':
        stats = torch.load(args.statistics, map_location='cpu', weights_only=True)
        provenance={}
        if stats.get('statistics_version')==1:
            if stats['checkpoint_fingerprint']!=fingerprint(args.model) or stats['checkpoint_identity']!=checkpoint_identity(args.model):
                raise ValueError('statistics/checkpoint mismatch')
            provenance={k:v for k,v in stats.items() if k not in ('statistics_version','statistics')}
            stats=stats['statistics']
        elif not args.allow_legacy_statistics:
            raise ValueError('unverified legacy statistics; collect again or use --allow-legacy-statistics for development')
        artifact = fit_statistics(stats, args)
        artifact.update(provenance)
        artifact['fit_versions']=package_versions()
        artifact['fit_source_sha256']=source_identity()
        artifact['statistics_sha256']=file_sha256(args.statistics)
        artifact['checkpoint_fingerprint'] = fingerprint(args.model)
        save(artifact, args.output)
    else:
        if args.command in ('calibrate','smoke') and min(args.segments,args.sequence_length,args.sample_every,args.max_snapshots)<1:
            parser.error('calibration counts must be positive')
        run_model(args)


def add_fit_arguments(p):
    p.add_argument("--state-format",choices=("stepquant",),default="stepquant")
    p.add_argument('--bits', type=int, choices=(4,6), default=6)
    p.add_argument('--pivots', type=int, help='defaults: 32 GDN heads / 512 KDA rows')
    p.add_argument('--horizon', type=int, default=2048)
    p.add_argument('--value-group-size', type=int, default=32)
    p.add_argument('--fit-device', default='cuda:0' if torch.cuda.is_available() else 'cpu')


if __name__ == '__main__':
    main()
