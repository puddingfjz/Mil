"""CPU analyses for Figures 3/4 and Table 1, with explicit missing-evidence states."""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from .config import artifact_root, load_yaml, repo_root
from .hashing import sha256_file
from .schema import atomic_write_json


def js_divergence(p, q):
    p, q = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
    if p.shape != q.shape or p.sum() <= 0 or q.sum() <= 0 or np.any(p < 0) or np.any(q < 0):
        raise ValueError('JSD requires equal-sized nonnegative nonempty histograms')
    p, q = p / p.sum(), q / q.sum()
    middle = (p + q) / 2
    def kl(a):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / middle[mask])))
    return (kl(p) + kl(q)) / 2


def distributions(records):
    groups, categories, bins = defaultdict(list), defaultdict(list), defaultdict(list)
    for row in records:
        model, dataset = row['model'], row['dataset']
        inp, out = row['input_length'], row['output_length']
        if type(inp) is not int or type(out) is not int or min(inp, out) <= 0:
            raise ValueError('request lengths must be positive integers')
        groups[(model, dataset)].append(out)
        if model == 'vicuna-13b-v1.5' and dataset == 'no_robot':
            bins[str((inp // 100) * 100)].append(out)
            if row.get('category'):
                categories[row['category']].append(out)
    maximum = max((max(v) for v in groups.values()), default=0)
    histograms = {k: np.bincount(v, minlength=maximum + 1).tolist() for k, v in groups.items()}
    jsd = []
    for model in sorted({k[0] for k in groups}):
        datasets = sorted(k[1] for k in groups if k[0] == model)
        for i, a in enumerate(datasets):
            for b in datasets[i + 1:]:
                jsd.append({'model': model, 'datasets': [a, b],
                            'js_divergence_bits': js_divergence(histograms[(model, a)], histograms[(model, b)])})
    return {'histograms': [{'model': k[0], 'dataset': k[1], 'counts': v, 'requests': len(groups[k])}
                           for k, v in histograms.items()],
            'input_length_bins': {k: np.bincount(v, minlength=maximum + 1).tolist() for k, v in bins.items()},
            'categories': {k: np.bincount(v, minlength=maximum + 1).tolist() for k, v in categories.items()},
            'js_divergence': jsd, 'bin_width_tokens': 100,
            'note': 'Input-bin width is an explicit analysis choice; category metadata must be supplied, never inferred from length.'}


def profile_points(path, skip):
    rows = []
    with path.open() as stream:
        for line in stream:
            if line.startswith('[['):
                record = json.loads(line)
                meta, tflops, times = record[:3]
                rows.append({'batch': meta[0], 'context_tokens': meta[1], 'max_length': meta[3],
                             'phase': 'prefill' if meta[4] else 'decode', 'tflops': tflops,
                             'compute_seconds': times[1], 'sampling_seconds': times[2],
                             'preparation_seconds': times[3]})
    rows = rows[skip:]
    fits = []
    for phase in ('prefill', 'decode'):
        values = [r for r in rows if r['phase'] == phase]
        if len(values) < 2:
            continue
        x = np.asarray([r['tflops'] for r in values])
        y = np.asarray([r['compute_seconds'] for r in values])
        slope, intercept = np.polyfit(x, y, 1)
        predicted = slope * x + intercept
        denominator = float(np.sum((y - y.mean())**2))
        fits.append({'phase': phase, 'n': len(values), 'slope': float(slope), 'intercept': float(intercept),
                     'r_squared': 1 - float(np.sum((y - predicted)**2)) / denominator if denominator else None})
    return {'points': rows, 'fits': fits}


def build(config, output, routing_requests=None):
    cfg = load_yaml(config)
    output.mkdir(parents=True, exist_ok=False)
    sources, records = [], []
    for name in cfg['distribution_records']:
        path = repo_root() / name
        sources.append({'path': name, 'sha256': sha256_file(path)})
        records.extend(json.loads(line) for line in path.read_text().splitlines())
    dist = distributions(records)
    observed = {(r['model'], r['dataset']) for r in dist['histograms']}
    missing_pairs = [[m, d] for m in cfg['distribution_requirements']['models']
                     for d in cfg['distribution_requirements']['datasets'] if (m, d) not in observed]
    items = {
        'figure_3ab': {'status': 'partial' if records else 'missing_raw_records',
                      'missing': 'Exact temperature-1 Vicuna paired lengths and ten request categories not recovered from active source tree.'},
        'figure_3c_jsd': {'status': 'partial' if records else 'missing_raw_records', 'missing_model_datasets': missing_pairs}}
    latency = []
    for entry in cfg['latency_profiles']:
        path = repo_root() / entry['path']
        if path.is_file():
            sources.append({'path': entry['path'], 'sha256': sha256_file(path)})
            latency.append({**entry, **profile_points(path, entry.get('skip_records', 0))})
    items['figure_3d'] = {'status': 'historical_candidate' if latency else 'missing_raw_records',
                         'paper_membership': 'unresolved', 'profiles': len(latency)}
    trace_config = cfg['trace_comparison']
    traces = {}
    for kind in ('real_trace', 'simulated_trace'):
        if trace_config.get(kind):
            path = repo_root() / trace_config[kind]
            trace = json.loads(path.read_text())
            if any(trace.get(k) != trace_config['required_' + k] for k in ('model', 'dataset', 'requests')):
                raise ValueError('Figure 4 trace workload mismatch')
            if trace.get('kind') != kind:
                raise ValueError('real and simulated traces must be explicitly distinguished')
            if not trace.get('running_requests_by_iteration'):
                raise ValueError('missing iteration events')
            traces[kind] = trace
            sources.append({'path': str(path), 'sha256': sha256_file(path)})
    items['figure_4'] = {'status': 'paired_traces' if len(traces) == 2 else 'missing_paired_traces',
                         'missing': [kind for kind in ('real_trace', 'simulated_trace') if kind not in traces],
                         'note': 'Llama-2-7b baseline trace is not the required Vicuna/ShareGPT/1000 trace.'}
    if routing_requests:
        bundle = json.loads(routing_requests.read_text())
        if bundle['application'] != 'router':
            raise ValueError('Table 1 requires Routing requests')
        counts = [len(bundle['input_lengths'][str(i)]) for i in range(len(bundle['model_paths']))]
        expected = [408, 1267, 2068, 456, 2657]
        items['table_1'] = {'status': 'verified' if counts == expected else 'mismatch',
                            'model_paths': bundle['model_paths'], 'counts': counts, 'expected': expected,
                            'total_requests': sum(counts), 'request_set_sha256': sha256_file(routing_requests)}
        items['table_1']['length_statistics'] = [
            {'model': model, 'input_mean': float(np.mean(bundle['input_lengths'][str(i)])),
             'input_maximum': max(bundle['input_lengths'][str(i)]),
             'dataset_response_mean': float(np.mean(bundle['known_output_lengths'][str(i)])),
             'dataset_response_maximum': max(bundle['known_output_lengths'][str(i)])}
            for i, model in enumerate(bundle['model_paths'])]
    else:
        items['table_1'] = {'status': 'missing_declared_request_set'}
    atomic_write_json(output / 'supporting.json', {'schema_version': 1, 'items': items, 'sources': sources,
        'distributions': dist, 'latency': latency, 'traces': traces, 'gpu_execution': False})
    return {k: v['status'] for k, v in items.items()}


def plot(snapshot, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    data = json.loads((snapshot / 'supporting.json').read_text())
    output.mkdir(parents=True, exist_ok=False)
    for i, profile in enumerate(data['latency']):
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        for ax, metric in zip(axes, ('compute_seconds', 'sampling_seconds', 'preparation_seconds')):
            for phase in ('prefill', 'decode'):
                rows = [r for r in profile['points'] if r['phase'] == phase]
                ax.scatter([r['tflops'] for r in rows], [r[metric] for r in rows], s=3, label=phase, rasterized=True)
            ax.set(xlabel='TFLOPs per iteration', ylabel=metric)
            ax.legend()
        fig.suptitle('Figure 3(d) candidate profile; exact paper run unresolved')
        fig.tight_layout()
        fig.savefig(output / f'figure_3d_candidate_{i}.pdf')
        plt.close(fig)
    dist = data['distributions']
    for model in sorted({r['model'] for r in dist['histograms']}):
        rows = [r for r in dist['histograms'] if r['model'] == model]
        fig, ax = plt.subplots(figsize=(8, 4))
        for row in rows:
            v = np.asarray(row['counts'])
            ax.plot(v / v.sum(), label=row['dataset'])
        ax.set(xlabel='Output tokens', ylabel='Empirical probability', title=model)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(output / f'figure_3c_{model}.pdf')
        plt.close(fig)
        matrix = np.asarray([[js_divergence(a['counts'], b['counts']) for b in rows] for a in rows])
        fig, ax = plt.subplots(figsize=(7, 6))
        plot = ax.imshow(matrix, vmin=0, vmax=1)
        labels = [r['dataset'] for r in rows]
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=90)
        ax.set_yticklabels(labels)
        ax.set_title(f'{model}: Jensen-Shannon divergence (bits)')
        fig.colorbar(plot, ax=ax)
        fig.tight_layout()
        fig.savefig(output / f'figure_3c_jsd_{model}.pdf')
        plt.close(fig)
    for field in ('input_length_bins', 'categories'):
        if dist[field]:
            fig, ax = plt.subplots(figsize=(8, 4))
            for name, values in sorted(dist[field].items()):
                v = np.asarray(values)
                ax.plot(np.cumsum(v) / v.sum(), label=name)
            ax.set(xlabel='Output tokens', ylabel='Empirical cumulative probability', title=field)
            ax.legend(fontsize=6)
            fig.tight_layout()
            fig.savefig(output / f'figure_3_{field}.pdf')
            plt.close(fig)
    if data['traces']:
        fig, ax = plt.subplots(figsize=(9, 4))
        for kind, trace in data['traces'].items():
            ax.plot(trace['running_requests_by_iteration'], label=kind)
        ax.set(xlabel='Iteration', ylabel='Running requests', title='Figure 4: observed vs simulated')
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / 'figure_4.pdf')
        plt.close(fig)
    atomic_write_json(output / 'coverage.json', data['items'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('build')
    prep.add_argument('--config', type=Path, default=artifact_root() / 'configs/supporting_analyses.yaml')
    prep.add_argument('--routing-requests', type=Path)
    prep.add_argument('--output', type=Path, required=True)
    plots = sub.add_parser('plot')
    plots.add_argument('--snapshot', type=Path, required=True)
    plots.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'build':
        print(json.dumps(build(args.config, args.output, args.routing_requests), indent=2))
    else:
        plot(args.snapshot, args.output)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
