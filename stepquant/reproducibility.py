"""Small, explicit identities for source and local checkpoints."""
import hashlib
from pathlib import Path
from importlib.metadata import version, PackageNotFoundError


def package_versions():
    result = {}
    for name in ('torch','numpy','transformers','accelerate','fla-core','triton','sglang'):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            pass
    return result


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def source_identity():
    root = Path(__file__).parent
    return {str(p.relative_to(root)): file_sha256(p) for p in sorted(root.rglob('*.py'))}


def checkpoint_identity(path):
    root = Path(path)
    metadata = {p.name: file_sha256(p) for p in sorted(root.iterdir())
                if p.is_file() and (p.suffix == '.json' or p.name.endswith('.jinja'))}
    weights = {p.name: dict(size=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns)
               for p in sorted(root.glob('*.safetensors'))}
    return dict(metadata_sha256=metadata, weights_stat=weights,
                weight_identity_scope='file names/sizes/mtime; not a content checksum')
