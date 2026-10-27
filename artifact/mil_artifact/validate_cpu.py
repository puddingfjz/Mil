"""Run representative real CPU searches, not the full performance matrix."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys

from .compare_schedules import compare_frozen
from .config import artifact_root, load_yaml, repo_root
from .hashing import sha256_file
from .cpu_reference import load_reference, validate_request
from .plot_entry import safe_path
import math
from .source_contract import source_path
from .schema import atomic_write_json


CASES = [('ensemble', 'mil'), ('routing', 'mil'), ('chain', 'mil_chain_summary'),
         ('mixed', 'mil'), ('ensemble', 'max'), ('routing', 'hybridflow_star'),
         ('chain', 'light_rule_star'), ('mixed', 'min')]
SOURCE_FILES = {'search_exec_plans.py', 'fake_scheduling.py', 'schedule_multi_model.py', 'output_length_sampler.py'}
IMPORTED_MODULES = {'vllm', 'search_exec_plans', 'fake_scheduling'}


def verify_result(requests, result_path, reference, source_root, expected_method, history=None):
    result = json.loads(result_path.read_text())
    method_matches = result.get('method') == expected_method
    bundle = json.loads(requests.read_text())
    validate_request(bundle)
    methods = load_yaml(source_root / 'artifact/configs/methods.yaml')['methods']
    resource_spec = load_yaml(source_root / 'artifact/manifests/cpu_preparation.yaml')['resources']
    inputs_match = (
        result.get('status') == 'completed_cpu_scheduling'
        and result.get('request_set_sha256') == sha256_file(requests)
        and result.get('parameters') == {**bundle['parameters'], **methods[expected_method].get('args', {})}
        and result.get('resource_hashes') == {
            role: sha256_file(source_path(source_root, spec['path']))
            for role, spec in resource_spec.items()})
    actual = result.get('actual', {})
    plans = actual.get('plans')
    cost = actual.get('predicted_seconds')
    numeric = lambda n: type(n) in (int, float) and math.isfinite(n) and n >= 0
    structure_valid = (isinstance(plans, list) and bool(plans)
                       and all(isinstance(group, list) and bool(group) for group in plans)
                       and (cost is None or numeric(cost)))
    if cost is None and expected_method not in {'light_rule', 'light_rule_no_preemption', 'saturn_star_no_preemption'}:
        structure_valid = False
    if 'stage_times' in actual:
        stages = actual['stage_times']
        structure_valid = (structure_valid and isinstance(stages, list) and len(stages) == len(plans)
                           and all(numeric(n) for n in stages) and numeric(cost)
                           and math.isclose(sum(stages), cost, rel_tol=1e-9, abs_tol=1e-9))
    regression = None
    if reference is not None and method_matches:
        regression = compare_frozen(requests, result_path, reference)
    source_hashes_match = set(result.get('source_hashes', {})) == SOURCE_FILES
    if source_hashes_match:
        source_hashes_match = all(sha256_file(source_path(source_root, 'benchmarks/' + name)) == digest
                                 for name, digest in result['source_hashes'].items())
    origins_match = set(result.get('imported_modules', {})) == IMPORTED_MODULES
    if origins_match:
        try:
            expected_imports = {'vllm':'vllm/__init__.py',
                                'search_exec_plans':'benchmarks/search_exec_plans.py',
                                'fake_scheduling':'benchmarks/fake_scheduling.py'}
            origins_match = all(Path(result['imported_modules'][name]) == source_path(source_root, path)
                                for name,path in expected_imports.items())
        except ValueError:
            origins_match = False
    guarded = (result.get('cuda_initialized') is False
               and result.get('compiled_gpu_extension_loaded') is False
               and result.get('gpu_operator_module') == '<CPU-only forbidden GPU operators>')
    historical = None
    if history is not None and method_matches:
        from .history import compare_history
        historical = compare_history(history, result['actual'])
    passed = (method_matches and inputs_match and structure_valid
              and (regression is None or regression['passed']) and source_hashes_match and origins_match and guarded
              and (historical is None or historical['passed']))
    return {'passed': passed, 'regression': regression, 'history': historical,
            'inputs_match': inputs_match, 'schedule_structure_valid': bool(structure_valid),
            'comparison_to_prior_results': reference is not None or history is not None,
            'expected_method': expected_method, 'reported_method': result.get('method'),
            'method_matches': method_matches,
            'source_hashes_match': source_hashes_match, 'import_origins_match': origins_match,
            'gpu_guard_verified': guarded, 'result': result}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=2, choices=(1, 2))
    parser.add_argument('--reference', type=Path, help='Optional external CPU regression reference; none is bundled.')
    parser.add_argument('--sources', type=Path, help='External log index, required with --history-root.')
    parser.add_argument('--history-root', type=Path,
                        help='Optional active source repository containing original logs; never a backup.')
    parser.add_argument('--cases', nargs='+', choices=[a + '/' + m for a, m in CASES])
    args = parser.parse_args(argv)
    if bool(args.history_root) != bool(args.sources):
        parser.error('--history-root and --sources must be provided together')
    reference = load_reference(args.reference)[1] if args.reference else None
    from .plot_entry import new_output_path
    output = new_output_path(args.output)
    if args.history_root:
        args.history_root = safe_path(args.history_root)
        args.sources = safe_path(args.sources)
    output.mkdir(parents=True, exist_ok=False)
    root, artifact = repo_root(), artifact_root()
    methods = load_yaml(artifact / 'configs/methods.yaml')['methods']
    records = [json.loads(line) for line in args.sources.read_text().splitlines()] if args.history_root else []
    tasks = []
    for app, method in CASES:
        name = app + '/' + method
        if args.cases and name not in args.cases:
            continue
        requests = artifact / 'inputs/frozen' / (app + '.json')
        history = None
        if args.history_root:
            from .history import snapshot_history
            history = snapshot_history(json.loads(requests.read_text()), {**methods[method], 'id': method},
                                       records, args.history_root)
        tasks.append((name, requests, method, history))
    # Freeze source choices before starting any real searches.
    atomic_write_json(output / 'selection.json', {name: history for name, _, _, history in tasks})
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'PYTHONDONTWRITEBYTECODE': '1',
           'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
           'HF_HOME': str(output / 'empty-model-cache'), 'MIL_CPU_SCHEDULING': '1'}
    def run(task):
        name, requests, method, history = task
        destination = output / name.replace('/', '_')
        command = [sys.executable, '-B', '-m', 'artifact.mil_artifact.cpu_schedule', 'run',
                   '--requests', str(requests), '--method', method, '--cpus', '2',
                   '--output', str(destination)]
        print('START ' + name, flush=True)
        with destination.with_suffix('.stdout').open('x') as log:
            code = subprocess.call(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
        report = {'case': name, 'command': command, 'exit_code': code, 'passed': False,
                  'outer_timeout_seconds': None}
        if code == 0:
            try:
                report.update(verify_result(requests, destination / 'result.json', reference, root, method, history))
            except (ValueError, KeyError, OSError) as error:
                report['verification_error'] = str(error)
        atomic_write_json(output / (name.replace('/', '_') + '.check.json'), report)
        print('DONE ' + name + ' passed=' + str(report['passed']), flush=True)
        return report
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        reports = list(pool.map(run, tasks))
    report = {'schema_version': 1, 'scope': 'CPU scheduling usability and structural integrity; historical equality is checked only with explicit external references.',
              'source_root': str(root), 'outer_timeout_seconds': None, 'gpu_execution': False,
              'passed': bool(reports) and all(r['passed'] for r in reports), 'cases': reports,
              'excluded': ['Full search matrix', 'Saturn optimization/license changes', 'MIN-R statistical repetitions', 'GPU inference']}
    atomic_write_json(output / 'validation.json', report)
    print(json.dumps({'passed': report['passed'], 'cases': len(reports), 'output': str(output)}))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
