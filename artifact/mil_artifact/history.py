"""Stage-specific historical evidence; never infer source versions from results."""
from __future__ import annotations

import hashlib
import json
import math
import mmap
from pathlib import Path
import re
import tokenize

from .archive import ERROR_MARKERS
from .compare_schedules import read_initial
from .paper_results import matches
from .scheduling_replay import compare


POLICY = ('All parameter/method matches, in path order; no mtime/version ranking, '
          'no result-based selection. Unknown source versions remain unresolved.')


def safe_log_path(root, relative):
    root, relative = Path(root).absolute(), Path(relative)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('history source must be repository-relative')
    path = root / relative
    if any('backup' in p.lower() or 'camera-ready' in p.lower() for p in path.parts):
        raise ValueError('backup paths are forbidden')
    # System prefixes such as /tmp may be symlinks. Within the explicitly
    # supplied root, reject every symlink; also reject resolved backup targets.
    if any('backup' in p.lower() or 'camera-ready' in p.lower() for p in root.resolve().parts):
        raise ValueError('backup paths are forbidden')
    within_root = [root]
    current = root
    for part in relative.parts:
        current = current / part
        within_root.append(current)
    if any(p.is_symlink() for p in within_root):
        raise ValueError('history source and ancestors must not be symlinks')
    return path


def version_evidence(record):
    name = Path(record['path']).stem
    suffix = name[name.index('bugfix'):] if 'bugfix' in name else 'unlabelled'
    return {'source_commit': None, 'status': 'unverified', 'filename_hint': suffix,
            'method_header': {k: record['parameters'].get(k) for k in
                              ('gen_execplans_baseline', 'search_method_baseline')},
            'paper_membership': 'unresolved',
            'note': 'Names, comments and equal outputs are clues, not commit identity.'}


def finite_nonnegative(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def inspect_log(record, root):
    """Hash the full file; parse the FIRST final plan and locate fatal diagnostics.

    Later diagnostics do not invalidate the already completed initial CPU stage,
    but never establish GPU success. No original script is imported/executed.
    """
    path = safe_log_path(root, record['path'])
    before = path.stat()
    digest = hashlib.sha256()
    first_errors, completion = {}, None
    with path.open('rb') as stream:
        while chunk := stream.read(8 * 1024**2):
            digest.update(chunk)
        if before.st_size:
            with mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as data:
                for marker, label in ERROR_MARKERS.items():
                    pos = data.find(marker)
                    if pos >= 0:
                        first_errors[label] = pos
                marker = (rb'(?m)^other hybridflow res infor: .*' if
                          record['parameters'].get('search_method_baseline') == 'hybridflow_model_parallel'
                          else rb'(?m)^Best group seq: .*')
                found = re.search(marker, data)
                if found:
                    completion = found.end()
    if digest.hexdigest() != record['sha256']:
        raise ValueError(f"historical log changed since indexing: {record['path']}")
    method = ('light_rule' if record['parameters'].get('search_method_baseline')
              in ('LightRule', 'LightRule_no_preemption') else 'other')
    error, lengths, expected = None, {}, None
    try:
        lengths, expected = read_initial(path, method)
    except (ValueError, SyntaxError, IndexError, TypeError, KeyError, tokenize.TokenError) as exc:
        error = type(exc).__name__  # Do not copy raw request text into evidence.
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError('history source changed during inspection')
    models = record.get('model_paths') or []
    inputs_complete = bool(models) and set(lengths) == {str(i) for i in range(len(models))}
    inputs_complete = inputs_complete and all(isinstance(v, list) and v and
        all(type(n) is int and n >= 0 for n in v) for v in lengths.values())
    plan_complete = bool(expected and isinstance(expected.get('plans'), list) and expected['plans'])
    if expected:
        cost = expected.get('predicted_seconds')
        plan_complete = plan_complete and (finite_nonnegative(cost) or
                                           (method == 'light_rule' and cost is None))
        if 'stage_times' in expected:
            times = expected['stage_times']
            plan_complete = plan_complete and isinstance(times, list) and bool(times) and all(
                finite_nonnegative(t) for t in times) and len(times) == len(expected['plans'])
    early = [label for label, pos in first_errors.items() if completion is None or pos < completion]
    late = [label for label, pos in first_errors.items() if completion is not None and pos >= completion]
    usable = bool(inputs_complete and plan_complete and completion is not None and not early)
    metrics = record.get('metrics', {})
    timing = all(finite_nonnegative(metrics.get(k)) for k in
                 ('search_time_seconds', 'preparation_time_seconds'))
    gpu_complete = bool(record.get('complete_timing') and not first_errors)
    if expected and not plan_complete:
        expected = None
        error = error or 'invalid_or_incomplete_plan'
    return {'source': record['path'], 'sha256': digest.hexdigest(), 'full_hash_verified': True,
            'indexed_status': record['status'], 'mtime_ns': record['mtime_ns'],
            'version': version_evidence(record), 'cpu_plan_usable': usable,
            'cpu_timing_available': timing, 'gpu_timing_complete': gpu_complete,
            'cpu_status': ('complete_with_later_diagnostics' if usable and late else
                           'complete' if usable else 'incomplete_or_failed'),
            'diagnostics_before_cpu_completion': early, 'diagnostics_after_cpu_completion': late,
            'initial_plan_end_byte': completion, 'parse_error': error,
            'inputs_present': bool(inputs_complete),
            'input_lengths': lengths, 'expected': expected}


def matching_records(bundle, method, records):
    case = {'test_case': bundle['parameters']['test_case'],
            'parameters': bundle['parameters'], 'model_paths': bundle['model_paths']}
    return sorted((r for r in records if matches(r, case, method, {})), key=lambda r: r['path'])


def snapshot_history(bundle, method, records, root, cache=None):
    cache = {} if cache is None else cache
    candidates = []
    for record in matching_records(bundle, method, records):
        if record['path'] not in cache:
            cache[record['path']] = inspect_log(record, root)
        row = dict(cache[record['path']])
        row['input_lengths_equal'] = row.pop('input_lengths') == bundle['input_lengths']
        row['model_order_equal'] = record.get('model_paths') == bundle['model_paths']
        row['eligible_for_initial_comparison'] = bool(row['cpu_plan_usable'] and
            row['input_lengths_equal'] and row['model_order_equal'])
        candidates.append(row)
    return {'policy': POLICY, 'available': any(r['eligible_for_initial_comparison'] for r in candidates),
            'source_commit': 'not_recorded', 'version_certified': False, 'candidates': candidates,
            'scope': 'Initial CPU regression only. GPU completeness and paper/version attribution are separate.'}


def compare_history(snapshot, actual):
    rows = []
    for candidate in snapshot['candidates']:
        row = dict(candidate)
        expected = row.get('expected')
        row['comparison'] = None
        if row['eligible_for_initial_comparison']:
            if expected.get('predicted_seconds') is None or actual.get('predicted_seconds') is None:
                equal = json.loads(json.dumps(expected['plans'])) == json.loads(json.dumps(actual['plans']))
                row['comparison'] = {'passed': equal, 'plans_exact_match': equal,
                                     'cost_comparison': 'unavailable'}
            else:
                row['comparison'] = compare(expected, actual)
        rows.append(row)
    compared = [r for r in rows if r['comparison'] is not None]
    return {**snapshot, 'candidates': rows, 'compared': len(compared),
            'matched': sum(r['comparison']['passed'] for r in compared),
            'passed': bool(compared) and all(r['comparison']['passed'] for r in compared),
            'interpretation': 'A mismatch is retained, not a proof of a current-code defect or permission to select another log.'}


def compact_snapshot(snapshot):
    """Keep reproducible digests and numeric evidence, not giant input/plan arrays."""
    result = dict(snapshot)
    for key in ('input_lengths', 'expected'):
        value = result.pop(key, None)
        if value is not None:
            result[key + '_sha256'] = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
            if key == 'expected':
                result['initial_predicted_seconds'] = value.get('predicted_seconds')
    if 'candidates' in result:
        result['candidates'] = [compact_snapshot(row) for row in result['candidates']]
    return result
