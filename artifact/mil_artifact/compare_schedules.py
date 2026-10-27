"""Compare normal-input CPU plans with every matching archived initial plan."""
from __future__ import annotations
import argparse
import ast
import json
from pathlib import Path
import re
from .config import artifact_root, load_yaml, repo_root
from .schema import atomic_write_json
from .hashing import sha256_file
from .scheduling_replay import compare, parse_plan


def read_initial(path, method):
    lengths, expected = {}, None
    with path.open(encoding='utf-8', errors='replace') as stream:
        for line in stream:
            match = re.match(r'len\(inp_lens\) of model (\d+): (.*)', line)
            if match:
                lengths[match[1]] = ast.literal_eval(match[2])
            if line.startswith('Best group seq: '):
                raw = line.split(': ', 1)[1]
                if method == 'light_rule':
                    expected = {'plans': ast.literal_eval(raw), 'predicted_seconds': None}
                else:
                    expected = parse_plan(raw)
                break
            if line.startswith('best_group_seq: '):
                expected = {'plans': ast.literal_eval(line.split(': ', 1)[1])}
            if line.startswith('other hybridflow res infor: ') and expected:
                expected['predicted_seconds'] = float(line.split('(', 1)[1].split(',', 1)[0])
                break
    return lengths, expected


def compare_result(requests, result, source_index):
    # Import here: history reuses the format parser above, but both entrypoints
    # must share stage validation, hash checks and all-candidate comparison.
    from .history import snapshot_history, compare_history
    bundle = json.loads(requests.read_text())
    actual = json.loads(result.read_text())
    method_id = actual['method']
    method = load_yaml(artifact_root() / 'configs/methods.yaml')['methods'][method_id]
    history = compare_history(snapshot_history(bundle, method, source_index, repo_root()), actual['actual'])
    reports = [{**row, 'source_sha256': row['sha256'],
                'historical_execution_status': row['indexed_status'],
                'random_method_rule': 'Seed not logged historically; exact equality is diagnostic, not a required invariant.'
                if method_id == 'min_r' else None} for row in history['candidates']]
    return {'schema_version': 1, 'application': bundle['application'], 'method': method_id,
            'request_set_sha256': actual['request_set_sha256'], 'actual': actual['actual'], 'sources': reports,
            'selection_policy': history['policy'], 'all_historical_comparisons_match': history['passed'],
            'scope': 'Initial allocation/order/cost only; later errors do not certify GPU success. Versions remain unverified.',
            'wall_clock_rule': 'Search wall-clock time is not compared across machines/CPU counts.'}


def compare_frozen(requests, result, reference):
    """Validate a fresh CPU result without requiring historical raw logs."""
    actual = json.loads(result.read_text())
    frozen = json.loads(reference.read_text())
    digest = sha256_file(requests)
    if actual['request_set_sha256'] != digest:
        raise ValueError('result belongs to a different request set')
    portable = frozen.get('kind') == 'cpu_regression_reference'
    if portable:
        from .cpu_reference import validate_references
        validate_references(frozen)
        candidates = [case for case in frozen['cases']
                      if case['request_set_sha256'] == digest and case['method'] == actual['method']]
    else:
        # Compatibility for explicitly supplied historical reference files only.
        candidates = [case['report'] for case in frozen['final_cases'].values()
                      if case['exit_code'] == 0 and case['report']['request_set_sha256'] == digest
                      and case['method'] == actual['method']]
    if len(candidates) != 1:
        raise ValueError('exactly one frozen CPU reference must match method and request-set hash')
    expected = candidates[0]
    if expected['actual'].get('predicted_seconds') is None:
        comparison = {'plans_exact_match': actual['actual']['plans'] == expected['actual']['plans'],
                      'cost_comparison': 'not_available_for_this_method'}
        comparison['passed'] = comparison['plans_exact_match']
    else:
        comparison = compare(expected['actual'], actual['actual'])
    if portable:
        expected_resources = {role: spec['sha256'] for role,spec in frozen['resources'].items()}
        resources_match = actual.get('resource_hashes') == expected_resources
    else:
        from collections import Counter
        resources_match = not (Counter(expected['resources'].values()) - Counter(actual['resources'].values()))
    parameters_match = actual['parameters'] == expected['parameters']
    return {'schema_version': 1, 'method': actual['method'], 'request_set_sha256': digest,
            'reference_sha256': sha256_file(reference), 'comparison': comparison,
            'reference_format': 'portable_cpu_v2' if portable else 'explicit_legacy_file',
            'parameters_match': parameters_match, 'resources_match': resources_match,
            'passed': comparison['passed'] and parameters_match and resources_match,
            'scope': 'Frozen CPU initial-plan regression; historical paper membership and GPU timing are separate.',
            'wall_clock_rule': 'Search wall-clock time is excluded.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--requests', type=Path, required=True)
    parser.add_argument('--result', type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--sources', type=Path, help='Source index; original logs must be locally available')
    source.add_argument('--reference', type=Path, help='Frozen CPU validation evidence; no raw logs needed')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    if args.reference:
        report = compare_frozen(args.requests, args.result, args.reference)
    else:
        records = [json.loads(line) for line in args.sources.read_text().splitlines()]
        report = compare_result(args.requests, args.result, records)
    atomic_write_json(args.output / 'comparison.json', report)
    print(json.dumps({'passed': report['passed']} if args.reference else
                     {'historical_sources_compared': len(report['sources'])}))
    return 0 if not args.reference or report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
