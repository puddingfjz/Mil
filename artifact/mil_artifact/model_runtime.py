"""Use the same immutable, local model snapshots for preflight and execution."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re

from .config import artifact_root, load_yaml
from .hashing import sha256_file
from .prepare import _model_cache_root, _model_snapshot, model_snapshot_health


def snapshot_records(identifiers=None, *, verify=True):
    entries = load_yaml(artifact_root() / 'manifests/models.yaml')['models']
    selected = set(identifiers) if identifiers is not None else {row['id'] for row in entries}
    if selected - {row['id'] for row in entries}:
        raise ValueError('model is not in the pinned model manifest')
    root = _model_cache_root()
    records = {}
    for entry in entries:
        identifier = entry['id']
        if identifier not in selected:
            continue
        revision = entry.get('revision', '')
        if not re.fullmatch(r'[0-9a-f]{40}', revision):
            raise ValueError('model revision must be an immutable commit: ' + identifier)
        path = _model_snapshot(root, identifier, revision)
        if not path.is_dir():
            raise ValueError('missing pinned model snapshot: ' + str(path))
        if verify:
            health = model_snapshot_health(path, entry.get('integrity'), entry.get('license', {}).get('evidence'))
            if not health['ok']:
                raise ValueError('invalid model snapshot ' + identifier + ': ' + health['detail'])
        records[identifier] = {'revision': revision, 'snapshot_path': str(path),
                               'config_sha256': entry['integrity']['config_sha256']}
    return records


def resolve_model_snapshot(identifier):
    """Never fall back to main, another cache, or an implicit network download."""
    encoded = os.environ.get('MIL_MODEL_SNAPSHOTS')
    if encoded:
        records = json.loads(encoded)
    elif Path(identifier).is_absolute():
        # Inference receives an already-resolved model path. Accept only a path
        # belonging to the same pinned manifest/cache, not an arbitrary directory.
        entries = load_yaml(artifact_root() / 'manifests/models.yaml')['models']
        cache = _model_cache_root()
        selected = [row['id'] for row in entries
                    if str(_model_snapshot(cache, row['id'], row['revision'])) == identifier]
        records = snapshot_records(selected)
    else:
        records = snapshot_records([identifier])
    record = records.get(identifier)
    if record is None:
        record = next((row for row in records.values() if row['snapshot_path'] == identifier), None)
    if record is None:
        raise ValueError('model was not verified by preflight: ' + identifier)
    path = Path(record['snapshot_path'])
    if (not path.is_absolute() or not re.fullmatch(r'[0-9a-f]{40}', record['revision'])
            or path.name != record['revision'] or not path.is_dir()
            or sha256_file(path / 'config.json') != record['config_sha256']):
        raise ValueError('verified model snapshot is missing or changed: ' + identifier)
    return str(path)
