"""Read-only audit of a source-and-input release, not a performance certification."""
from __future__ import annotations
import argparse
from dataclasses import asdict, dataclass
import json
import sys
from .config import load_yaml, repo_root
from .verify_preparation import verify


@dataclass(frozen=True)
class ReleaseCheck:
    name: str
    status: str
    detail: str


def run_release_checks(repository=None):
    repository = repository or repo_root()
    manifest = load_yaml(repository / 'artifact/manifests/release.yaml')
    checks = [ReleaseCheck('distribution_scope',
        'ok' if manifest.get('kind') == 'source_and_inputs' and manifest.get('experimental_results_included') is False else 'error',
        'Source, experiment tools and running inputs only; no bundled experimental results.')]
    report = verify(repository)
    checks.extend(ReleaseCheck(row['name'], 'ok' if row['passed'] else 'error',
                              json.dumps({'value': row['value'], 'expected': row['expected']}))
                  for row in report['checks'])
    checks.extend(ReleaseCheck('integrity_issue', 'error', issue) for issue in report['issues'])
    checks.append(ReleaseCheck('gpu_performance', 'warning',
                               'GPU build, inference and paper performance are not certified by this CPU audit.'))
    return checks


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--json', action='store_true')
    parser.add_argument('--no-strict', action='store_true')
    args = parser.parse_args(argv)
    try:
        checks = run_release_checks()
    except (OSError, ValueError, KeyError) as error:
        print(f'release audit error: {error}', file=sys.stderr)
        return 2
    summary = {status: sum(c.status == status for c in checks) for status in ('ok', 'warning', 'error')}
    if args.json:
        print(json.dumps({'summary': summary, 'checks': [asdict(c) for c in checks]}, indent=2))
    else:
        for check in checks:
            print(f'[{check.status.upper()}] {check.name}: {check.detail}')
    return 0 if args.no_strict or summary['error'] == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
