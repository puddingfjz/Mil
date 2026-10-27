"""Numeric candidate attribution; equality never establishes a source version."""
import math
import statistics

METRICS = {"runtime": "real_running_time_seconds", "extra": "search_time_seconds", "total": "total_time_seconds"}


def close(a, b, tolerance=1e-9):
    return (isinstance(a, (int, float)) and isinstance(b, (int, float))
            and math.isfinite(a) and math.isfinite(b) and abs(a - b) <= tolerance)


def attribute(runtime, total, candidates, rounded=False):
    values = {'runtime': runtime, 'extra': total - runtime, 'total': total}
    if runtime == total == 0:
        return {'status': 'missing_placeholder', 'values': values, 'components': {},
                'same_run_candidates': [], 'reason': 'Zero/cross in the plot, not a zero-second measurement.'}
    parts = {}
    for name, field in METRICS.items():
        exact = [r['id'] for r in candidates if close(r['metrics'].get(field), values[name])]
        compatible = [r['id'] for r in candidates
                      if rounded and float(values[name]).is_integer()
                      and isinstance(r['metrics'].get(field), (int, float))
                      and abs(r['metrics'][field] - values[name]) <= 0.5 and r['id'] not in exact]
        parts[name] = {'value_seconds': values[name], 'exact_candidates': exact,
                       'rounding_compatible_candidates': compatible}
    same = sorted(set.intersection(*(set(p['exact_candidates']) for p in parts.values())))
    if same:
        status = 'exact_unique_same_run_candidate' if len(same) == 1 else 'exact_ambiguous_same_run_candidates'
    elif parts['runtime']['exact_candidates'] and parts['extra']['exact_candidates']:
        status = 'separate_component_candidates'
    elif any(p['exact_candidates'] for p in parts.values()):
        status = 'partial_exact_components'
    elif any(p['rounding_compatible_candidates'] for p in parts.values()):
        status = 'rounding_compatible_only'
    else:
        status = 'unresolved'
    return {'status': status, 'values': values, 'components': parts, 'same_run_candidates': same}


def statistics_match(samples, stats):
    for key, metric in [('run_time', 'runtime'), ('tot_time', 'total')]:
        values = [s[metric] for s in samples]
        if not values or not all(close(a, b, 5.1e-9) for a, b in [
                (statistics.mean(values), stats[key]), (min(values), stats[key + '_min']),
                (max(values), stats[key + '_max'])]):
            return False
    return True
