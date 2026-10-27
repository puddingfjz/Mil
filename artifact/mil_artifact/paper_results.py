"""Recover paper-coordinate candidates from raw-log indexes; plot frozen data.

Candidate membership is not proof of inclusion in the published paper. All
matching runs and exclusions are retained; no minimum-time selection is used.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path
import re
import statistics

from .archive import read_execution_events, final_models_and_counters
from .config import artifact_root, load_yaml, repo_root
from .hashing import sha256_file
from .paper_suite import compile_suite
from .schema import atomic_write_json


DEFAULTS = {'outlen_known': False, 'noise_level': 0, 'oracle_model_num': -1,
            'no_runtime_adjustment': False, 'model_replicate_num': 1, 'router_replicate_num': 1}
CONTROL_KEYS = {'reqnum', 'max_token_num', 'specify_outlen', 'outlen_known', 'noise_level',
                'oracle_model_num', 'no_runtime_adjustment', 'max_group_seq_num', 'top_k',
                'similar_threshold', 'ratio_seed', 'ratio_set'}
APP_KEYS = {'general': set(), 'router': {'router_replicate_num', 'router_question_version'},
            'chain-summary': {'evaluator_num', 'summarize_model', 'evaluator_model'},
            'mixed': {'evaluator_num', 'summarize_model', 'evaluator_model', 'reqnum_mixed_blender',
                      'max_token_num_mixed_blender', 'model_replicate_num'}}


def matches(record, case, method, defaults):
    observed = record['parameters']
    wanted = {**defaults, **case['parameters'], **method.get('args', {})}
    if observed.get('test_case') != case['test_case']:
        return False
    dispatch = dict(method['legacy'])
    if dispatch['search_method_baseline'] == 'hybridflow_model_parallel_old':
        dispatch['search_method_baseline'] = 'hybridflow_model_parallel'
        if 'Round2' in record['path']:
            return False
    elif dispatch['search_method_baseline'] == 'hybridflow_model_parallel':
        if 'Round2' not in record['path']:
            return False
    if any(observed.get(k) != v for k, v in dispatch.items()):
        return False
    for key in CONTROL_KEYS | APP_KEYS[case['test_case']]:
        value = observed.get(key, DEFAULTS.get(key))
        # Older LightRule explicitly replaces noise with expected length after
        # printing arguments. Its registry records the effective setting.
        if key == 'noise_level' and dispatch['search_method_baseline'] == 'LightRule':
            value = -2
        if key in wanted and value != wanted[key]:
            return False
    hardware = {'tot_gpu_num': 8, 'machine_name': 'zxcpu', 'gpu_name': 'A100-80G',
                'byte_per_gpu': 80 * 1024**3, 'fully_connected_gpu_unit': 2}
    if any(observed.get(key) != value for key, value in hardware.items()):
        return False
    if record.get('model_paths') and set(record['model_paths']) != set(case['model_paths']):
        return False
    return True


def recover(index_path, metadata_path, output, timelines=False):
    records = [json.loads(line) for line in index_path.read_text().splitlines()]
    metadata = {r['path']: r for r in map(json.loads, metadata_path.read_text().splitlines())}
    if not records or set(metadata) != {r['path'] for r in records}:
        raise ValueError('nonempty log index and matching per-file metadata required')
    catalog, _, coverage, _ = compile_suite()
    methods = load_yaml(artifact_root() / 'configs/methods.yaml')['methods']
    output.mkdir(parents=True, exist_ok=False)
    for record in records:
        current = (repo_root() / record['path']).stat()
        if (current.st_size, current.st_mtime_ns) != (record['size_bytes'], record['mtime_ns']):
            raise ValueError(f"source changed since hashing: {record['path']}")
        record.update(metadata[record['path']])
        prefix = final_models_and_counters(repo_root() / record['path'])
        if prefix['model_paths'] is not None:
            record['model_paths'] = prefix['model_paths']
        record['metrics'].update(prefix['search_counters'])
        metrics = record['metrics']
        if record['complete_timing']:
            metrics['extra_time_seconds'] = metrics['search_time_seconds']
            metrics['total_time_seconds'] = metrics['real_running_time_seconds'] + metrics['search_time_seconds']
            metrics['search_after_preparation_seconds'] = metrics['search_time_seconds'] - metrics['preparation_time_seconds']
        record['timing_convention'] = 'v2_search_includes_preparation'
        record['status'] = ('failed' if record['diagnostics'] else
                            'complete_timing' if record['complete_timing'] else 'incomplete')
    cells = {}
    for case_id, case in coverage['cases'].items():
        for method_id in case['methods']:
            found = [r for r in records if matches(r, case, methods[method_id], catalog['defaults'])]
            valid = [r for r in found if r['status'] == 'complete_timing']
            key = case_id + '/' + method_id
            cell = {'case_id': case_id, 'method': method_id, 'parameters': case['parameters'],
                    'application': case['workload'], 'records': [r['path'] for r in found],
                    'included': [r['path'] for r in valid],
                    'excluded': [{'path': r['path'], 'status': r['status'], 'diagnostics': r['diagnostics']}
                                 for r in found if r not in valid],
                    'status': 'historical_candidates' if valid else 'missing_complete_timing',
                    'paper_membership': 'unresolved', 'seed_membership': 'not_logged',
                    'selection': 'All parameter-matching complete-timing runs without fatal markers; no performance ranking.',
                    'aggregates': {}}
            for field in ('real_running_time_seconds', 'search_time_seconds', 'preparation_time_seconds',
                          'extra_time_seconds', 'search_after_preparation_seconds', 'total_time_seconds', 'initial_predicted_seconds',
                          'incremental_predicted_seconds'):
                values = [r['metrics'][field] for r in valid if field in r['metrics']]
                if values:
                    cell['aggregates'][field] = {'n': len(values), 'mean': statistics.mean(values),
                        'median': statistics.median(values), 'minimum': min(values), 'maximum': max(values)}
            if method_id == 'min_r':
                cell['paper_repetition_count'] = 3 if len(case['model_paths']) >= 8 else None
                cell['protocol_status'] = 'unresolved_seed_and_run_membership'
            cells[key] = cell
    items = {}
    for item_id, item_report in coverage['items'].items():
        members = item_report['members']
        keys = list(dict.fromkeys(m['case_id'] + '/' + method for m in members for method in m['methods']))
        items[item_id] = {'kind': catalog['items'][item_id]['kind'], 'cells': keys,
                         'cells_with_candidates': sum(bool(cells[k]['included']) for k in keys),
                         'missing_cells': [k for k in keys if not cells[k]['included']],
                         'status': 'partial_historical_recovery' if any(cells[k]['included'] for k in keys)
                                   else 'supporting_analysis' if not keys else 'no_matching_records'}
    selected = {p for c in cells.values() for p in c['records']}
    with (output / 'sources.jsonl').open('x') as stream:
        for record in records:
            stream.write(json.dumps(record, allow_nan=False) + '\n')
    atomic_write_json(output / 'results.json', {'schema_version': 1, 'cells': cells, 'items': items,
        'source_count': len(records), 'matched_source_count': len(selected),
        'source_index_sha256': sha256_file(index_path), 'metadata_sha256': sha256_file(metadata_path),
        'aggregation': 'Descriptive mean/median/range of all candidates, not asserted paper aggregates.',
        'hybridflow_caveat': 'Round2 filename distinguishes candidate generations only; historical commit is not in logs.'})
    if timelines:
        events = {}
        for key in items['figure_9']['cells']:
            for source in cells[key]['included']:
                event = read_execution_events(repo_root() / source)
                if event['complete']:
                    event['source'] = source
                    event['source_sha256'] = next(r['sha256'] for r in records if r['path'] == source)
                    event['selection'] = 'First complete event trace in lexicographic path order; illustrative, paper membership unresolved.'
                    events[key] = event
                    break
        atomic_write_json(output / 'timelines.json', events)
    checks = []
    for record in records:
        metrics = record['metrics']
        if record['status'] == 'complete_timing':
            expected = sum(metrics[k] for k in ('real_running_time_seconds', 'search_time_seconds'))
            error = abs(metrics['total_time_seconds'] - expected)
            checks.append({'path': record['path'], 'value': metrics['total_time_seconds'],
                           'expected_from_logged_components': expected, 'absolute_error': error,
                           'tolerance': 1e-9, 'passed': error <= 1e-9 and all(math.isfinite(x) and x >= 0 for x in (
                               metrics['real_running_time_seconds'], metrics['search_time_seconds'], metrics['preparation_time_seconds']))})
    atomic_write_json(output / 'numerical_checks.json', {'checks': checks,
        'passed': all(c['passed'] for c in checks), 'scope': 'Logged component consistency; published figure equality remains separate.'})
    analyses = {'prediction_errors': {}, 'runtime_adjustment': [], 'search_statistics': [],
                'scope': 'Descriptive historical candidates; unresolved run/seed membership prevents paper-equality claims.'}
    by_path = {r['path']: r for r in records}
    for key in items['estimation_error']['cells']:
        cell = cells[key]
        errors = []
        for source in cell['included']:
            metrics = by_path[source]['metrics']
            predicted = metrics.get('incremental_predicted_seconds', metrics.get('initial_predicted_seconds'))
            if predicted is not None and metrics['real_running_time_seconds'] > 0:
                errors.append({'source': source, 'predicted_seconds': predicted,
                               'measured_seconds': metrics['real_running_time_seconds'],
                               'relative_absolute_error': abs(predicted - metrics['real_running_time_seconds']) / metrics['real_running_time_seconds']})
        analyses['prediction_errors'][key] = errors
    grouped = {}
    for key, rows in analyses['prediction_errors'].items():
        grouped.setdefault(cells[key]['application'], []).extend(r['relative_absolute_error'] for r in rows)
    analyses['prediction_error_by_application'] = {app: {'n': len(values), 'mean': statistics.mean(values),
        'maximum': max(values)} for app, values in grouped.items() if values}
    for key in items['search_efficiency']['cells']:
        for source in cells[key]['included']:
            metrics = by_path[source]['metrics']
            analyses['search_statistics'].append({'cell': key, 'source': source, 'statistics': {
                k: v for k, v in metrics.items() if k in ('search_time_seconds', 'preparation_time_seconds',
                'search_after_preparation_seconds') or k.startswith(('Cost_Query', 'Plan_Group', 'Select_Best', 'Iter_Search'))}})
    controls = {}
    for key in items['runtime_adjustment']['cells']:
        cell = cells[key]
        params = {**catalog['defaults'], **cell['parameters']}
        disabled = params.pop('no_runtime_adjustment')
        identity = json.dumps([cell['application'], params], sort_keys=True)
        controls.setdefault(identity, {})[disabled] = key
    for pair in controls.values():
        if set(pair) == {False, True}:
            enabled, disabled = cells[pair[False]], cells[pair[True]]
            on = enabled['aggregates'].get('total_time_seconds', {}).get('mean')
            off = disabled['aggregates'].get('total_time_seconds', {}).get('mean')
            analyses['runtime_adjustment'].append({'enabled': pair[False], 'disabled': pair[True],
                'descriptive_speedup': off / on if on and off else None,
                'status': 'candidate_aggregate_ratio' if on and off else 'missing_comparison'})
    atomic_write_json(output / 'analyses.json', analyses)
    return {'cells': len(cells), 'with_candidates': sum(bool(c['included']) for c in cells.values()),
            'sources': len(records), 'matched': len(selected)}


def plot(snapshot, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    import numpy as np
    data = json.loads((snapshot / 'results.json').read_text())
    output.mkdir(parents=True, exist_ok=False)
    artifacts = []
    main_axes = {'figure_6a': ('reqnum', 'Ensembling requests'),
        'figure_6b': ('router_replicate_num', 'Routing replication (6856 base requests)'),
        'figure_6c': ('router_replicate_num', 'Routing replication (known output lengths)'),
        'figure_6d': ('reqnum', 'Documents'), 'figure_6e': ('evaluator_num', 'Evaluations per document'),
        'figure_6f': ('max_token_num', 'Maximum output tokens'), 'figure_7': ('reqnum', 'Summary documents'),
        'figure_8': ('model_replicate_num', 'Applications (8 logical GPUs)')}
    method_registry = load_yaml(artifact_root() / 'configs/methods.yaml')['methods']
    # A reviewable table for every scheduling/analysis item, including missing
    # cells. Empty numeric fields remain empty rather than becoming zeros.
    for item_id, item in data['items'].items():
        if not item['cells']:
            continue
        rows = []
        for key in item['cells']:
            cell = data['cells'][key]
            row = {'case_id': cell['case_id'], 'application': cell['application'],
                   'method': cell['method'], 'parameters': json.dumps(cell['parameters'], sort_keys=True),
                   'status': cell['status'], 'candidate_runs': len(cell['included']),
                   'excluded_runs': len(cell['excluded']), 'paper_membership': cell['paper_membership']}
            for field in ('real_running_time_seconds', 'extra_time_seconds', 'total_time_seconds'):
                for statistic in ('mean', 'minimum', 'maximum'):
                    row[field + '_' + statistic] = cell['aggregates'].get(field, {}).get(statistic)
            rows.append(row)
        with (output / f'{item_id}.csv').open('x', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    for item_id, item in data['items'].items():
        if item['kind'] in ('output_distribution', 'latency_profile', 'inference_trace_comparison', 'input_statistics'):
            continue
        if item_id == 'figure_9':
            continue
        cells = [data['cells'][key] for key in item['cells']]
        if item_id in main_axes:
            parameter, xlabel = main_axes[item_id]
            coordinates = list(dict.fromkeys(c['parameters'][parameter] for c in cells))
            methods = list(dict.fromkeys(c['method'] for c in cells))
            fig, ax = plt.subplots(figsize=(max(9, len(coordinates) * 2.3), 5))
            width = .8 / len(methods)
            for j, method in enumerate(methods):
                selected = {c['parameters'][parameter]: c for c in cells if c['method'] == method}
                positions = np.arange(len(coordinates)) + (j - (len(methods) - 1) / 2) * width
                lower, overhead = [], []
                for coordinate in coordinates:
                    metrics = selected[coordinate]['aggregates']
                    lower.append(metrics.get('real_running_time_seconds', {}).get('mean', np.nan))
                    overhead.append(metrics.get('extra_time_seconds', {}).get('mean', np.nan))
                color = plt.get_cmap('tab10')(j % 10)
                ax.bar(positions, lower, width, color=color, label=method_registry[method]['paper_label'])
                ax.bar(positions, overhead, width, bottom=lower, color=color, alpha=.45, hatch='///')
                for pos, coordinate, value in zip(positions, coordinates, lower):
                    cell = selected[coordinate]
                    if not math.isfinite(value):
                        ax.plot([pos], [0], marker='x', color='crimson', markersize=5)
                    elif method == 'min_r':
                        total = cell['aggregates']['total_time_seconds']
                        ax.errorbar([pos], [total['mean']], yerr=[[total['mean'] - total['minimum']],
                            [total['maximum'] - total['mean']]], fmt='none', color='black', capsize=2)
            labels = [2 * v for v in coordinates] if item_id == 'figure_8' else coordinates
            ax.set_xticks(range(len(coordinates)))
            ax.set_xticklabels(labels)
            ax.set(xlabel=xlabel, ylabel='Seconds (inference + inclusive search)',
                   title=f'{item_id}: archived candidates, exact paper membership unresolved')
            ax.legend(ncol=3, fontsize=8)
            ax.grid(axis='y', alpha=.2)
            fig.text(.01, .005, 'Cross = missing/failed; MIN-R bars show all candidate min/max, not a reconstructed three-run paper sample.', fontsize=7)
            fig.tight_layout(rect=(0, .025, 1, 1))
            fig.savefig(output / f'{item_id}.pdf')
            fig.savefig(output / f'{item_id}.png', dpi=160)
            plt.close(fig)
            artifacts.append(item_id)
            continue
        if item['kind'] == 'prediction_error':
            metrics = ('initial_predicted_seconds', 'real_running_time_seconds')
        elif item['kind'] == 'search_statistics':
            metrics = ('search_after_preparation_seconds', 'preparation_time_seconds')
        else:
            metrics = ('real_running_time_seconds', 'extra_time_seconds')
        fig, ax = plt.subplots(figsize=(max(9, len(cells) * .27), 5))
        x = np.arange(len(cells))
        bottom = np.zeros(len(cells))
        for metric_index, metric in enumerate(metrics):
            values = np.asarray([c['aggregates'].get(metric, {}).get('mean', np.nan) for c in cells])
            if item['kind'] == 'prediction_error':
                ax.bar(x + (metric_index - .5) * .4, values, width=.4, label=metric.replace('_seconds', ''))
            else:
                ax.bar(x, values, bottom=bottom, label=metric.replace('_seconds', ''))
                bottom += np.nan_to_num(values)
        for i, cell in enumerate(cells):
            if not cell['included']:
                ax.text(i, 0, 'missing', rotation=90, va='bottom', fontsize=7, color='crimson')
        labels = []
        for cell in cells:
            p = cell['parameters']
            scale = (f"rep={p.get('router_replicate_num', 1)}" if cell['application'] == 'router'
                     else f"n={p['reqnum']}")
            variant = ''
            if item_id == 'table_3':
                variant = '\nExact' if p.get('outlen_known') else f"\nnoise={p.get('noise_level', 0)}"
            elif item_id == 'runtime_adjustment':
                variant = '\nadjust=off' if p.get('no_runtime_adjustment') else '\nadjust=on'
            labels.append(f"{cell['application']} {scale}{variant}\n{cell['method']}")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=90, fontsize=7)
        ax.set_ylabel('seconds')
        ax.set_title(f'{item_id}: historical candidates; paper membership unresolved')
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output / f'{item_id}.pdf')
        fig.savefig(output / f'{item_id}.png', dpi=140)
        plt.close(fig)
        artifacts.append(item_id)
    timelines = json.loads((snapshot / 'timelines.json').read_text()) if (snapshot / 'timelines.json').exists() else {}
    for key, event in timelines.items():
        fig, ax = plt.subplots(figsize=(11, 4))
        start = 0.0
        models = {}
        for stage, duration in zip(event['stages'], event['durations_seconds']):
            for plan in stage:
                match = re.search(r'model_id: (\d+), comp_gpus: (\[.*\])', plan)
                if not match:
                    continue
                model = int(match[1])
                models[model] = plan.split(', ', 1)[0]
                for gpu in json.loads(match[2]):
                    ax.broken_barh([(start, duration)], (gpu - .4, .8), facecolors=plt.get_cmap('tab10')(model % 10))
            start += duration
        name = 'figure_9_' + key.split('/')[1]
        ax.set(xlabel='Actual stage elapsed seconds', ylabel='Logical GPU', yticks=list(range(8)),
               title=f'{name}: observed runtime events (illustrative archived run)')
        ax.legend(handles=[Patch(facecolor=plt.get_cmap('tab10')(model % 10), label=models[model])
                           for model in sorted(models)], loc='upper center',
                  bbox_to_anchor=(.5, -.17), ncol=3, fontsize=7)
        ax.grid(axis='x', alpha=.2)
        fig.tight_layout()
        fig.savefig(output / f'{name}.pdf')
        fig.savefig(output / f'{name}.png', dpi=160)
        plt.close(fig)
        artifacts.append(name)
    atomic_write_json(output / 'plot_manifest.json', {'artifacts': artifacts,
        'input_sha256': sha256_file(snapshot / 'results.json'), 'gpu_execution': False,
        'missing_and_conflicting_evidence': 'See results.json cells/items and supporting.json.'})
    return artifacts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    recover_parser = sub.add_parser('recover')
    recover_parser.add_argument('--index', type=Path, required=True)
    recover_parser.add_argument('--metadata', type=Path, required=True)
    recover_parser.add_argument('--output', type=Path, required=True)
    recover_parser.add_argument('--timelines', action='store_true')
    plot_parser = sub.add_parser('plot')
    plot_parser.add_argument('--snapshot', type=Path, required=True)
    plot_parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = recover(args.index, args.metadata, args.output, args.timelines) if args.command == 'recover' else plot(args.snapshot, args.output)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
