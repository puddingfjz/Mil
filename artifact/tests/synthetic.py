"""Artificial test data only; no experimental measurements or saved schedules."""
import json
from artifact.mil_artifact.config import repo_root, artifact_root, load_yaml
from artifact.mil_artifact.hashing import sha256_file


def cpu_reference(method='mil'):
    root = repo_root()
    request_path = 'artifact/inputs/frozen/ensemble.json'
    request = json.loads((root / request_path).read_text())
    resources = {role: {'path': path, 'sha256': sha256_file(root / path)} for role, path in {
        'cost_table': 'benchmarks/get_my_cost_table_directly_zxcpu.py',
        'output_lengths': 'benchmarks/collect_output_lengths/no_robot/out_len_sampler_2.py',
        'model_configurations': 'artifact/inputs/model_configurations.json',
    }.items()}
    case = {'case_id': 'ensemble/' + method, 'method': method,
            'request_path': request_path, 'request_set_sha256': sha256_file(root / request_path),
            'parameters': {**request['parameters'],
                           **load_yaml(artifact_root() / 'configs/methods.yaml')['methods'][method].get('args', {})},
            'recorded_resource_hashes': {k: v['sha256'] for k, v in resources.items()},
            'actual': {'plans': [['synthetic-stage-a'], ['synthetic-stage-b']],
                       'stage_times': [1.0, 2.0], 'predicted_seconds': 3.0}}
    return {'schema_version': 2, 'kind': 'cpu_regression_reference',
            'resources': resources, 'cases': [case]}


def chart_data():
    from artifact.mil_artifact.published_plot import METHOD_LABELS, METHOD_KEYS
    keys = ['greedy_saturn_LPT', *METHOD_KEYS]
    bars = [{'label': label, 'method_key': [key, 1],
             'runtime': [float(i + 10)], 'total': [float(i + 12)]}
            for i, (label, key) in enumerate(zip(METHOD_LABELS, keys))]
    samples = [{'runtime': 12., 'total': 14.}, {'runtime': 14., 'total': 16.}]
    aggregation = {'aggregation': 'arithmetic_mean', 'sample_count': 2, 'samples': samples,
                   'statistics': {'run_time': 13., 'tot_time': 15., 'run_time_min': 12.,
                                  'run_time_max': 14., 'tot_time_min': 14., 'tot_time_max': 16.}}
    return {'png': 'synthetic.png', 'axis_parameter': 'synthetic_coordinate',
            'coordinates': [1], 'displayed_coordinates': [1],
            'bars': bars, 'min_r': [aggregation],
            'style': {'font_size': 12, 'figsize': [12, 6], 'tick_labels': ['synthetic'],
                      'missing_markers': False, 'ylabel_font_size': None}}
