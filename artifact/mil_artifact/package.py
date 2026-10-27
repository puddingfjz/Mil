"""Build a source distribution from an explicit file list, never a directory scan."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import stat
import tarfile

from .config import artifact_root, load_yaml, repo_root
from .hashing import sha256_file
from .plot_entry import new_output_path
from .schema import atomic_write_json
from .source_contract import source_path


SNAPSHOT_RECORD = 'artifact/source_snapshot.json'


def verify_snapshot(root):
    """Verify an unpacked distribution without requiring Git metadata."""
    record_path = source_path(root, SNAPSHOT_RECORD)
    record = json.loads(record_path.read_text())
    manifest = load_yaml(source_path(root, 'artifact/manifests/source_distribution.yaml'))
    if (record.get('schema_version') != 1 or record.get('kind') != 'source_snapshot'
            or set(record.get('files_sha256', {})) != set(manifest['files'])):
        raise ValueError('invalid source snapshot inventory')
    for name, digest in record['files_sha256'].items():
        if sha256_file(source_path(root, name)) != digest:
            raise ValueError('source snapshot checksum mismatch: ' + name)
    return {'files': len(record['files_sha256']), 'sha256': sha256_file(record_path)}


def assemble(root, manifest, output):
    if manifest.get('schema_version') != 1 or manifest.get('kind') != 'source_distribution':
        raise ValueError('unsupported source distribution manifest')
    files = manifest['files']
    if not files or len(set(files)) != len(files):
        raise ValueError('empty or duplicate distribution file list')
    if SNAPSHOT_RECORD in files:
        raise ValueError('source snapshot record is generated, not a distribution input')
    selected = []
    for relative in files:
        path = source_path(root, relative)
        if not stat.S_ISREG(path.stat().st_mode):
            raise ValueError('distribution inputs must be regular files: ' + relative)
        selected.append((relative, path, sha256_file(path)))
    output = new_output_path(output)
    output.mkdir(parents=True, exist_ok=False)
    source = output / 'source'
    for relative, path, digest in selected:
        destination = source / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination, follow_symlinks=False)
        if sha256_file(destination) != digest or sha256_file(path) != digest:
            raise ValueError('source changed while packaging: ' + relative)
    snapshot = {'schema_version': 1, 'kind': 'source_snapshot',
                'files_sha256': {name: digest for name, _, digest in selected}}
    atomic_write_json(source / SNAPSHOT_RECORD, snapshot)
    archive = output / 'mil-source.tar.gz'
    with tarfile.open(archive, 'w:gz', dereference=False) as stream:
        for relative in [name for name, _, _ in selected] + [SNAPSHOT_RECORD]:
            path = source / relative
            info = stream.gettarinfo(str(path), arcname='mil/' + relative)
            info.uid = info.gid = 0
            info.uname = info.gname = ''
            info.mtime = 0
            with path.open('rb') as contents:
                stream.addfile(info, contents)
    report = {'schema_version': 1, 'kind': 'source_distribution',
              'files_sha256': {name: digest for name, _, digest in selected},
              'archive_sha256': sha256_file(archive), 'source_directory': 'source',
              'archive': archive.name, 'gpu_execution': False,
              'scope': 'A file-hashed source snapshot, not a performance certification or a Git push.'}
    atomic_write_json(output / 'manifest.json', report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = load_yaml(artifact_root() / 'manifests/source_distribution.yaml')
    report = assemble(repo_root(), manifest, args.output)
    print(json.dumps({'files': len(report['files_sha256']), 'output': str(args.output),
                      'archive_sha256': report['archive_sha256']}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
