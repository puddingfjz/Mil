"""Verify source, retained running inputs and synthetic CPU behavior; no result baselines."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from .config import artifact_root, load_yaml
from .cpu_reference import read_pinned, validate_request
from .hashing import sha256_file
from .paper_suite import compile_suite
from .schema import atomic_write_json
from .source_contract import source_path, verify_source_contract


def verify(repository=None):
    repository = Path(repository) if repository is not None else artifact_root().parent
    root = repository / 'artifact'
    manifest = load_yaml(root / 'manifests/cpu_preparation.yaml')
    issues, checks = [], []
    def check(name, value, expected=True):
        passed = value == expected
        checks.append({'name': name, 'value': value, 'expected': expected, 'passed': passed})
        if not passed:
            issues.append(name)
    check('manifest_schema', (manifest.get('schema_version'), manifest.get('kind')), (1, 'cpu_preparation'))
    for relative, digest in manifest['files_sha256'].items():
        try:
            actual = sha256_file(source_path(repository, relative))
        except (OSError, ValueError) as error:
            actual = None
            issues.append(f'{relative}: {error}')
        check(relative, actual, digest)
    try:
        contract = read_pinned(repository, manifest['source_contract'], manifest['source_contract_sha256'])
        result = verify_source_contract(repository, contract)
        check('source_semantics', result['passed'])
        check('source_contract_file_count', result['checked_files'], manifest['source_files'])
        issues.extend(f"source/{row['path']}: {row['error']}" for row in result['issues'])
    except (OSError, ValueError, KeyError, SyntaxError) as error:
        check('source_semantics', False)
        issues.append(f'source_semantics: {error}')
    methods = load_yaml(root / 'configs/methods.yaml')['methods']
    check('registered_methods', len(methods), manifest['registered_methods'])
    try:
        _, _, coverage, _ = compile_suite()
        check('paper_matrix', {'items': len(coverage['items']), 'settings': coverage['unique_cases'],
                              'expanded_tasks': coverage['unique_scheduling_tasks']}, manifest['paper_matrix'])
        check('paper_methods', sorted({m for c in coverage['cases'].values() for m in c['methods']}),
              manifest['paper_methods'])
    except (OSError, ValueError, KeyError) as error:
        check('paper_matrix', False)
        issues.append(f'paper_matrix: {error}')
    try:
        for role, spec in manifest['resources'].items():
            check('resource/' + role, sha256_file(source_path(repository, spec['path'])), spec['sha256'])
        for relative in manifest['request_sets']:
            request = read_pinned(repository, relative, manifest['files_sha256'][relative])
            validate_request(request)
        check('frozen_request_sets', len(manifest['request_sets']), manifest['frozen_request_sets'])
        cases_valid = all(method in methods and
                          f'artifact/inputs/frozen/{app}.json' in manifest['request_sets']
                          for app, method in (name.split('/') for name in manifest['representative_cases']))
        check('representative_inputs', cases_valid)
    except (OSError, ValueError, KeyError, TypeError) as error:
        check('input_integrity', False)
        issues.append(f'input_integrity: {error}')
    try:
        from .runtime_events import run_cases
        events = run_cases()
        check('runtime_event_cases', len(events['cases']), manifest['runtime_event_cases'])
        check('runtime_event_behavior', events['passed'] and not events['gpu_execution'])
    except (OSError, ValueError, KeyError, AssertionError) as error:
        check('runtime_event_behavior', False)
        issues.append(f'runtime_events: {error}')
    distribution = load_yaml(root / 'manifests/source_distribution.yaml')
    check('experimental_results_excluded', distribution.get('experimental_results_included'), False)
    forbidden = [p for p in distribution['files'] if
                 p.endswith('.log') or p.startswith(('artifact/results/', 'results/'))
                 or (p.startswith('artifact/expected/') and p != manifest['source_contract'])]
    check('result_payloads_absent', forbidden, [])
    check('baseline_measurements_absent', load_yaml(root / 'manifests/expected_results.yaml')['results'], [])
    return {'schema_version': 3, 'passed': not issues, 'issues': issues, 'checks': checks,
            'gpu_execution': False, 'new_scheduling_searches': 0, 'historical_audits_read': False,
            'scope': 'Source/input integrity and synthetic functional checks; no comparison to historical experiment results.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from .plot_entry import new_output_path
    output = new_output_path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    report = verify()
    atomic_write_json(output / 'verification.json', report)
    print(f"CPU bundle checks: {len(report['checks'])}; issues: {len(report['issues'])}")
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
