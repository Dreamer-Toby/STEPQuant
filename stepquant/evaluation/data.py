"""Validate frozen task files without materializing multi-GB private tests."""
import hashlib
import shutil
import json
from pathlib import Path


def protocol_identity(config):
    protocol = {k: v for k, v in config.items() if k != 'protocol_notes'}
    return hashlib.sha256(json.dumps(protocol, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]


def prepare_protocol(config, path, upstream):
    """Create a new protocol snapshot without overwriting legacy frozen tasks."""
    from .prepare import prepare
    path = Path(path)
    if not path.exists():
        prepare(config, path, upstream)
        return path
    metadata_path = path.with_suffix('.provenance.json')
    provenance = json.loads(metadata_path.read_text())
    previous = provenance['config']
    if protocol_identity(previous) == protocol_identity(config):
        return path
    target = path.parent / 'protocols' / protocol_identity(config) / path.name
    if target.exists():
        FrozenTasks(target, config)
        return target
    generation = {'sampling', 'chat_template_kwargs', 'protocol_notes'}
    rendering = lambda c: {k: v for k, v in c.items() if k not in generation}
    if rendering(previous) != rendering(config):
        prepare(config, target, upstream)
        return target
    # Only generation settings changed: prompts and pinned source rows remain
    # identical. Verify the old checksum before making an independent copy.
    FrozenTasks(path, previous)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, target)
    provenance = dict(provenance, config=config)
    target.with_suffix('.provenance.json').write_text(json.dumps(provenance, indent=2, ensure_ascii=False)+'\n')
    return target


def tasks_digest(tasks):
    if isinstance(tasks, FrozenTasks):
        return tasks.tasks_sha256
    h = hashlib.sha256(b'[')
    for i, task in enumerate(tasks):
        if i:
            h.update(b', ')
        h.update(json.dumps(task, sort_keys=True, ensure_ascii=False).encode())
    h.update(b']')
    return h.hexdigest()


class FrozenTasks:
    def __init__(self, path, config):
        self.path = Path(path)
        provenance = json.loads(self.path.with_suffix('.provenance.json').read_text())
        # Editorial notes may change; all executable protocol fields must match.
        protocol = lambda c: {k: v for k, v in c.items() if k != 'protocol_notes'}
        if protocol(provenance['config']) != protocol(config):
            raise ValueError(f'frozen data/config mismatch: {path}')
        self.ids = set()
        def checked():
            for task in self:
                key = str(task['id'])
                if key in self.ids:
                    raise ValueError(f'duplicate task identifier: {key}')
                self.ids.add(key)
                yield task
        self.tasks_sha256 = tasks_digest(checked())
        if self.tasks_sha256 != provenance['tasks_sha256']:
            raise ValueError(f'frozen task checksum mismatch: {path}')
        if len(self) != provenance['tasks'] or len(self) != config.get('expected_tasks', len(self)):
            raise ValueError(f'frozen task count mismatch: {path}')

    def __len__(self):
        return len(self.ids)

    def __iter__(self):
        with self.path.open() as stream:
            for line in stream:
                if line.strip():
                    yield json.loads(line)
