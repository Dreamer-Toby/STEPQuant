"""Capacity-neutral device labels and stable tuning-template identities."""
import re
from pathlib import Path
from .reproducibility import file_sha256


def device_label(name):
    return re.sub(r'[-_\s]*\d+(?:\.\d+)?\s*(?:GiB|GB|G)(?=$|[-_\s,/.])', '', str(name), flags=re.I).strip()


def public_metadata(value, *, _key=''):
    """Retain workload measurements without physical-device capacity labels."""
    if isinstance(value, dict):
        omitted = {'total_memory', 'total_memory_gib', 'total_memory_gb',
                   'gpu_memory_total', 'gpu_memory_total_gb', 'total_gpu_memory'}
        return {device_label(k): public_metadata(v, _key=str(k))
                for k, v in value.items() if k not in omitted}
    if isinstance(value, (list, tuple)):
        return [public_metadata(v, _key=_key) for v in value]
    if isinstance(value, str):
        is_device = _key.lower() in {'device', 'device_name', 'gpu', 'gpu_name',
                                    'device_model', 'gpu_model'}
        is_device = is_device or bool(re.search(r'NVIDIA|AMD|device_name=', value, re.I))
        return device_label(value) if is_device else value
    return value


def tuning_identity(directory):
    root = Path(directory)
    return {device_label(str(p.relative_to(root))): file_sha256(p)
            for p in sorted((root / 'configs').rglob('*.json'))}
