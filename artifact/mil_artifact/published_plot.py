"""Render paper-style bar charts from explicitly supplied numeric inputs."""
from __future__ import annotations

import copy
import hashlib
import math


METHOD_LABELS = ('Mil', 'MAX', 'MIN', 'MIN-R', 'Saturn*', 'LightRule',
                 'LightRule*', 'HybridFlow', 'HybridFlow*')
METHOD_KEYS = ('max_gpu', 'min_gpu_useAll', 'min_gpu_useAll_sampleModels',
               'saturnLP', 'LightRule', 'LightRule_our_cost_model',
               'hybridflow_model_parallel', 'hybridflow_model_parallel_new')
STAT_FIELDS = {'run_time', 'tot_time', 'run_time_min', 'run_time_max',
               'tot_time_min', 'tot_time_max'}


def pixels(path):
    from PIL import Image
    with Image.open(path) as image:
        rgba = image.convert('RGBA')
        return {'size': list(image.size),
                'rgb_sha256': hashlib.sha256(rgba.convert('RGB').tobytes()).hexdigest(),
                'alpha_sha256': hashlib.sha256(rgba.getchannel('A').tobytes()).hexdigest()}


def validate_chart(chart):
    """Reject incomplete, nonfinite or inconsistent inputs before drawing."""
    coordinates, bars, aggregates = chart['coordinates'], chart['bars'], chart['min_r']
    n = len(coordinates)
    if not n or len(set(coordinates)) != n or len(bars) != 9 or len(aggregates) != n:
        raise ValueError('invalid chart dimensions')
    if len(chart['displayed_coordinates']) != n or len(chart['style']['tick_labels']) != n:
        raise ValueError('invalid displayed coordinates')
    if tuple(b['label'] for b in bars) != METHOD_LABELS:
        raise ValueError('published method order changed')
    if bars[0]['method_key'] not in (['greedy_saturn_LPT', 1], ['greedy_saturn', 1]):
        raise ValueError('unknown Mil implementation')
    if [b['method_key'] for b in bars[1:]] != [[k, 1] for k in METHOD_KEYS]:
        raise ValueError('published method identities changed')

    def number(value):
        return type(value) in (int, float) and math.isfinite(value) and value >= 0

    def close(a, b):
        return math.isclose(a, b, rel_tol=0, abs_tol=1e-8)

    for bar in bars:
        if any(len(bar[field]) != n or not all(number(v) for v in bar[field])
               for field in ('runtime', 'total')):
            raise ValueError('invalid bar values')
        if any(t < r for r, t in zip(bar['runtime'], bar['total'])):
            raise ValueError('total is smaller than runtime')
    if any(v <= 0 for field in ('runtime', 'total') for v in bars[0][field]):
        raise ValueError('Mil normalization baseline must be positive')
    for i, aggregate in enumerate(aggregates):
        kind = aggregate['aggregation']
        if kind == 'copied_from_MIN':
            if aggregate.get('samples') or aggregate.get('statistics'):
                raise ValueError('copied MIN point must not invent independent samples')
            if any(bars[3][k][i] != bars[2][k][i] for k in ('runtime', 'total')):
                raise ValueError('copied MIN value differs from MIN')
            continue
        if kind not in ('arithmetic_mean', 'provided_statistics'):
            raise ValueError('unknown MIN-R aggregation')
        stats, samples = aggregate['statistics'], aggregate['samples']
        if set(stats) != STAT_FIELDS or not all(number(v) for v in stats.values()) or not samples:
            raise ValueError('incomplete MIN-R statistics')
        if aggregate['sample_count'] != len(samples):
            raise ValueError('MIN-R sample count mismatch')
        for sample in samples:
            if not all(number(sample[k]) for k in ('runtime', 'total')) or sample['total'] < sample['runtime']:
                raise ValueError('invalid MIN-R sample')
        for source, target in [('runtime', 'run_time'), ('total', 'tot_time')]:
            values = [s[source] for s in samples]
            if not all(close(stats[field], value) for field, value in (
                    (target, sum(values) / len(values)), (target + '_min', min(values)),
                    (target + '_max', max(values)), (target, bars[3][source][i]))):
                raise ValueError('MIN-R statistics disagree with selected samples or bars')
        if not stats['tot_time_min'] <= stats['tot_time'] <= stats['tot_time_max']:
            raise ValueError('invalid MIN-R error range')


def render(chart, output):
    """Draw numeric bars and return the actual bar heights used by Matplotlib."""
    validate_chart(chart)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    style = chart['style']
    rc = {'font.size': style['font_size'], 'legend.fontsize': 22,
          'axes.labelsize': 28, 'xtick.labelsize': 22, 'ytick.labelsize': 22,
          'figure.dpi': 100, 'savefig.dpi': 'figure'}
    bars, drawn = chart['bars'], []
    baseline_runtime = np.asarray(bars[0]['runtime'])
    baseline_total = np.asarray(bars[0]['total'])
    index, width = np.arange(len(chart['coordinates'])), 0.1

    with matplotlib.rc_context(rc):
        fig = plt.figure(figsize=style['figsize'])
        try:
            colors = plt.get_cmap('Set3')
            for i, bar in enumerate(bars):
                runtime, total = np.asarray(bar['runtime']), np.asarray(bar['total'])
                positions = index + i * width
                lower = plt.bar(positions, runtime, width, label=bar['label'], color=colors(i))
                upper = plt.bar(positions, total - runtime, width, bottom=runtime,
                                hatch='///', color=colors(i), alpha=0.5)
                if style['missing_markers']:
                    for x, value in zip(positions, runtime):
                        if value == 0:
                            plt.text(x, 100, '×', ha='center', va='bottom', fontsize=40,
                                     color=colors(i), fontweight='bold', alpha=1.0)
                for container, values, baseline, label_type in (
                        (lower, runtime, baseline_runtime, 'center'),
                        (upper, total, baseline_total, 'edge')):
                    # Match the paper's positive-value, half-up ratio labels.
                    ratios = (values / baseline * 10 + 0.5).astype(int) / 10
                    labels = ([str(v) if v > 0 else '' for v in ratios]
                              if style['missing_markers'] else ratios)
                    plt.bar_label(container, labels=labels, label_type=label_type)
                if i == 3 and chart['min_r'][0]['aggregation'] != 'copied_from_MIN':
                    minimum = np.asarray([a['statistics']['tot_time_min'] for a in chart['min_r']])
                    maximum = np.asarray([a['statistics']['tot_time_max'] for a in chart['min_r']])
                    plt.errorbar(positions, total, yerr=[total - minimum, maximum - total],
                                 fmt='none', ecolor='crimson', elinewidth=1.5, capsize=3, capthick=1.5)
                drawn.append({'method_key': bar['method_key'], 'label': bar['label'],
                              'runtime': [float(p.get_height()) for p in lower],
                              'total': [float(p.get_y() + p.get_height()) for p in upper]})
            # Zero-height legend entry for the stacked extra-time component.
            plt.bar(index + 8 * width, total - total, width,
                    label='extra time', hatch='///', color='white', alpha=0)
            ylabel = {} if style['ylabel_font_size'] is None else {'fontsize': style['ylabel_font_size']}
            plt.ylabel('Running Time (s)', **ylabel)
            plt.xticks(index + 4 * width, style['tick_labels'])
            plt.legend(ncol=2)
            plt.tight_layout()
            plt.savefig(output, format='png')
        finally:
            plt.close(fig)
    return {'bars': drawn, 'min_r': copy.deepcopy(chart['min_r']), 'rendered_pixels': pixels(output)}
