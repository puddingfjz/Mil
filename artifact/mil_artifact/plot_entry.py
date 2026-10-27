"""GPU-free plotting from explicitly supplied experiment results."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import tempfile

from .config import repo_root
from .hashing import sha256_file
from .schema import atomic_write_json
from .source_contract import protected_path_parts



def safe_path(path):
    """Reject backup paths lexically, before inspecting any filesystem object."""
    path = Path(path).absolute()
    if '..' in path.parts or protected_path_parts(path.parts):
        raise ValueError('protected paths and parent traversal are forbidden')
    # Allow the OS /tmp alias, checking its target lexically before any target
    # filesystem operation.
    # Arbitrary user symlinks are still refused (including aliases to backups).
    if path.parts[:2] == ('/', 'tmp') and Path('/tmp').is_symlink():
        target = Path(os.readlink('/tmp'))
        target = target if target.is_absolute() else Path('/') / target
        if target.parts[:2] == ('/', 'tmp'):
            raise ValueError('recursive /tmp alias')
        return safe_path(target.joinpath(*path.parts[2:]))
    # Inspect ancestors first: inspecting a child through a symlinked parent
    # would already traverse its target, even if the child itself is not a link.
    if any(p.is_symlink() for p in (*reversed(path.parents), path)):
        raise ValueError('symlink paths are not accepted')
    return path


def new_output_path(path):
    path = safe_path(path)
    root = repo_root()
    if path == root or root in path.parents:
        relative = path.relative_to(root)
        # In-repository writes belong only to result directories, never source,
        # old figures, expected inputs, evidence, or hidden metadata directories.
        if not (relative.parts[:1] == ('results',) or relative.parts[:2] == ('artifact', 'results')):
            raise ValueError('inside the repository, use a NEW results/ or artifact/results/ directory')
    if path.exists():
        raise FileExistsError(f'output already exists; choose a new directory: {path}')
    return path


@contextmanager
def plotting_environment():
    # Establish this before importing pyplot, and do not write the user's cache.
    previous = os.environ.get('MPLCONFIGDIR')
    with tempfile.TemporaryDirectory(prefix='mil-plot-cache-') as temporary:
        os.environ['MPLCONFIGDIR'] = temporary
        try:
            yield
        finally:
            if previous is None:
                os.environ.pop('MPLCONFIGDIR', None)
            else:
                os.environ['MPLCONFIGDIR'] = previous


def results(run, output, figures=None, config=None):
    """Plot a newly collected metrics.jsonl; no bundled results or fallback."""
    run = safe_path(run)
    digest = sha256_file(safe_path(run / 'metrics.jsonl'))
    output = new_output_path(output)
    from .plot import plot_run
    with plotting_environment():
        paths = plot_run(run, output, figures or (), safe_path(config) if config else None)
    if not paths:
        raise ValueError('no successful results match the selected figures')
    report = {'schema_version': 1, 'mode': 'results', 'passed': True,
              'gpu_execution': False, 'new_searches': 0,
              'input_sha256': digest, 'files': [p.name for p in paths],
              'scope': 'User-supplied measurements; no bundled paper values.'}
    atomic_write_json(output / 'plot_manifest.json', report)
    return report


def chart(data, output, figures=None):
    """Render an explicit chart JSON using the paper-style bar renderer."""
    data = safe_path(data)
    output = new_output_path(output)
    bundle = json.loads(data.read_text())
    if bundle.get('schema_version') != 1 or not isinstance(bundle.get('figures'), dict):
        raise ValueError('expected schema_version=1 and a figures mapping')
    selected = list(dict.fromkeys(figures or bundle['figures']))
    if not selected or set(selected) - set(bundle['figures']):
        raise ValueError('unknown or empty chart selection')
    from .published_plot import render, validate_chart
    names = set()
    for fid in selected:
        spec = bundle['figures'][fid]
        name = spec['png']
        if Path(name).name != name or not name.endswith('.png') or name in names:
            raise ValueError('invalid or duplicate chart image name')
        names.add(name)
        validate_chart(spec)
    output.mkdir(parents=True, exist_ok=False)
    drawn = {}
    with plotting_environment():
        for fid in selected:
            spec = bundle['figures'][fid]
            drawn[fid] = render(spec, output / spec['png'])
    report = {'schema_version': 1, 'mode': 'chart', 'passed': True,
              'gpu_execution': False, 'new_searches': 0,
              'input_sha256': sha256_file(data), 'figures': selected,
              'scope': 'User-supplied chart data; not verification against paper results.'}
    atomic_write_json(output / 'plot_data.json', drawn)
    atomic_write_json(output / 'plot_manifest.json', report)
    return report


def historical(mode, snapshot, output):
    """Plot an explicitly supplied analysis bundle."""
    output = new_output_path(output)
    snapshot = safe_path(snapshot)
    primary = snapshot / ('results.json' if mode == 'candidates' else 'supporting.json')
    digest = sha256_file(safe_path(primary))
    with plotting_environment():
        if mode == 'candidates':
            from .paper_results import plot
        elif mode == 'supporting':
            from .supporting_analyses import plot
        else:
            raise ValueError(f'unknown analysis mode: {mode}')
        plot(snapshot, output)
    report = {'schema_version': 1, 'mode': mode, 'passed': True,
              'scope': 'User-supplied analysis inputs; no bundled historical measurements.',
              'gpu_execution': False, 'new_searches': 0,
              'snapshot': str(snapshot), 'input_sha256': digest,
              'files': sorted(p.name for p in output.iterdir() if p.is_file())}
    atomic_write_json(output / 'entry_manifest.json', report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    result_parser = sub.add_parser('results', help='plot a collected experiment run')
    result_parser.add_argument('--run', required=True, type=Path)
    result_parser.add_argument('--config', type=Path)
    chart_parser = sub.add_parser('chart', help='render explicit numeric chart data')
    chart_parser.add_argument('--input', required=True, type=Path)
    for command in (result_parser, chart_parser):
        command.add_argument('--figure', action='append')
        command.add_argument('--output', required=True, type=Path, help='NEW directory only')
    for mode in ('candidates', 'supporting'):
        command = sub.add_parser(mode, help='plot an explicit analysis bundle')
        command.add_argument('--snapshot', type=Path, required=True)
        command.add_argument('--output', type=Path, required=True, help='NEW directory only')
    args = parser.parse_args(argv)
    try:
        if args.mode == 'results':
            report = results(args.run, args.output, args.figure, args.config)
        elif args.mode == 'chart':
            report = chart(args.input, args.output, args.figure)
        else:
            report = historical(args.mode, args.snapshot, args.output)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f'plot entry failed: {exc}\n')
    print(json.dumps({'mode': report['mode'], 'passed': report['passed'],
                      'output': str(args.output.absolute()), 'new_searches': 0, 'gpu_execution': False}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
