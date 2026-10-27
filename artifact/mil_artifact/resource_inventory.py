"""Inventory retained measurements without traversing backup directories."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re
import yaml
from .config import artifact_root, load_yaml, repo_root
from .hashing import sha256_file
from .resource_build import read_cost_table, read_literal_module, read_lengths, distribution_from_pairs
from .schema import atomic_write_json


def inventory(output):
    output.mkdir(parents=True, exist_ok=False)
    root = repo_root()
    catalog = load_yaml(artifact_root() / 'configs/paper_experiments.yaml')
    models = sorted({m for values in catalog['model_sets'].values() for m in values})
    by_short = {m.rsplit('/', 1)[-1]: m for m in models}
    table = read_cost_table(root / 'benchmarks/get_my_cost_table_directly_zxcpu.py')
    pdfs = read_literal_module(root / 'benchmarks/collect_output_lengths/no_robot/out_len_sampler_2.py', ('pdf_dict',))['pdf_dict']
    records, cost_entries, output_entries = [], [], []
    # Explicit nonrecursive allowlist. Never inspect sibling/copy/backup trees.
    for directory in ('benchmarks/Cost_Model_per_iter', 'benchmarks/Cost_Model_per_iter_zxcpu',
                      'benchmarks/collect_output_lengths/no_robot'):
        for path in sorted((root / directory).glob('*.log')):
            if path.is_symlink():
                continue
            relative = path.relative_to(root).as_posix()
            record = {'path': relative, 'sha256': sha256_file(path), 'size_bytes': path.stat().st_size,
                      'kind': 'output_lengths' if 'collect_output_lengths' in directory else 'cost_profile'}
            with path.open('rb') as stream:
                head = stream.read(16384).decode(errors='replace')
            match = re.search(r"\bmodel='([^']+)'", head)
            record['model'] = match[1] if match else None
            record['hardware_identity'] = 'zxcpu' if directory.endswith('_zxcpu') else 'historical_other_or_unrecorded_host'
            records.append(record)
            if record['kind'] == 'output_lengths':
                short = next((name for name in by_short if path.name.startswith('NEWROUND_' + name + '_')), None)
                if short is None:
                    continue
                temp = re.search(r'_temp([\d.]+)_', path.name)
                temperature = float(temp[1]) if temp else 1.0
                key = short if temperature == 1 else (short, temperature)
                if key not in pdfs:
                    record['status'] = 'no_frozen_distribution_for_context_limit'
                    continue
                try:
                    pairs = read_lengths(path, None, None)
                    distribution_from_pairs(pairs, len(pdfs[key]))
                except ValueError as exc:
                    record['status'] = 'retained_not_rebuilt'
                    record['reason'] = str(exc)
                    continue
                record.update(request_count=len(pairs), temperature=temperature, status='rebuildable')
                output_entries.append({'model': by_short[short], 'log': relative,
                    'max_model_len': len(pdfs[key]), 'num_prompts': 10000, 'expected_requests': len(pairs),
                    'temperature': temperature, 'tensor_parallel_size': 2, 'dataset': 'no_robot.parquet'})
            elif directory == 'benchmarks/Cost_Model_per_iter':
                match = re.match(r'NEWROUND_(.+)_tp([12])_0728_temp1.0_wldeg2_1.log$', path.name)
                if match and match[1] in by_short:
                    cost_entries.append({'model': by_short[match[1]], 'exec_plan': [int(match[2]), .9, 2, 0], 'log': relative})
    coverage = []
    for model in models:
        plans = [list(plan) for m, plan in table['prefill_table'] if m == model]
        sources = [r for r in records if r['model'] == model]
        coverage.append({'model': model, 'frozen_cost_plans': plans,
            'frozen_output_distribution': model.rsplit('/', 1)[-1] in pdfs,
            'resource_status': 'frozen_inherited',
            'raw_cost_sources': [r['path'] for r in sources if r['kind'] == 'cost_profile'],
            'raw_distribution_sources': [r['path'] for r in sources if r['kind'] == 'output_lengths'],
            'reconstructed_target_host': model in ('THUDM/chatglm3-6b', 'stabilityai/stablelm-tuned-alpha-7b')})
    config = {'schema_version': 1, 'name': 'retained_other_host_measurements',
        'description': 'Other/unrecorded-host profiles exported separately; never replace zxcpu cost resources.',
        'cost': {'base_table': 'benchmarks/get_my_cost_table_directly.py',
                 'coefficient_table': 'benchmarks/model_coeff_database.py', 'entries': cost_entries},
        'output_lengths': {'base_distribution': 'benchmarks/collect_output_lengths/no_robot/out_len_sampler_2.py',
                          'compare_distribution': 'benchmarks/collect_output_lengths/no_robot/out_len_sampler_2.py',
                          'entries': output_entries}}
    (output / 'additional_measurements.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    report = {'schema_version': 1, 'coverage': coverage, 'sources': records,
              'additional_cost_entries': len(cost_entries), 'additional_output_distributions': len(output_entries),
              'policy': 'Other-host cost logs remain separate. Context limit from frozen PDF is declared; no GPU acquisition.'}
    atomic_write_json(output / 'inventory.json', report)
    return {'models': len(coverage), 'logs': len(records), 'cost_entries': len(cost_entries),
            'output_distributions': len(output_entries)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(inventory(args.output), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
