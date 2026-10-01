"""Materialize SGLang device-matched files from capacity-neutral templates."""
import hashlib
import json
import os
from pathlib import Path
from .device import device_label, tuning_identity


def materialize_configs(templates, device_name=None, output_root=None):
    templates = Path(templates).resolve(strict=True)
    if device_name is None:
        from sglang.srt.utils import get_device_name
        device_name = get_device_name()
    actual = device_name.replace(' ', '_')
    identity = tuning_identity(templates)
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    root = Path(output_root) if output_root is not None else Path(__file__).parents[1] / 'artifacts/runtime/moe'
    destination = root / key
    matched = 0
    for source in sorted((templates / 'configs').rglob('*.json')):
        prefix, separator, suffix = source.name.partition('device_name=')
        if not separator:
            raise ValueError(f'tuning template has no device selector: {source.name}')
        family = suffix.split(',', 1)[0].removesuffix('.json').removesuffix('_down')
        if family != device_label(actual):
            continue
        target_name = prefix + separator + suffix.replace(family, actual, 1)
        target = destination / source.relative_to(templates).parent / target_name
        target.parent.mkdir(parents=True, exist_ok=True)
        data = source.read_bytes()
        if not target.exists() or target.read_bytes() != data:
            temporary = target.with_suffix(f'.{os.getpid()}.tmp')
            temporary.write_bytes(data)
            temporary.replace(target)
        matched += 1
    return destination if matched else None
