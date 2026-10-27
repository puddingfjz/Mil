"""Prepare numeric request sets and run the real scheduler, without GPU inference.

Preparation calls the original workload builders/tokenizers on declared datasets.
Scheduling consumes the resulting numeric request set, not an execution log.
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import signal
import subprocess
import tempfile
import time
from types import SimpleNamespace

from .config import artifact_root, load_yaml, repo_root
from .hashing import sha256_file
from .resource_build import COST_FIELDS, native, read_literal_module
from .schema import atomic_write_json
from .scheduling_replay import parse_plan


FIELDS = ('model_paths', 'check_gap', 'sort_input', 'input_edges', 'num_prompts',
          'inp_seq_ids_dict', 'inp_generator', 'inp_merger', 'outlen_generator',
          'prompt_templates_lens', 'node_dataset_chunk_mapping', 'inp_req_ids',
          'inp_req_from_which_model_which_out_reqs', 'out_req_id_mapping',
          'new_out_req_part_num', 'independent_srcs', 'prompt_template_args',
          'sampling_args_dict', 'seq_outlen_dict', 'test_cases', 'versions',
          'max_token_nums', 'specify_outlens')
KEEP = set(FIELDS) - {'inp_generator', 'inp_merger', 'outlen_generator',
                     'sampling_args_dict', 'prompt_template_args'}


def cpu_environment(seed):
    os.environ.update(CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                      PYTHONDONTWRITEBYTECODE='1', MIL_EXPERIMENT_SEED=str(seed),
                      MIL_CPU_SCHEDULING='1',
                      OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    paths = [str(artifact_root() / 'cpu_bootstrap'), str(repo_root() / 'benchmarks'), str(repo_root())]
    sys.path[:0] = paths
    os.environ['PYTHONPATH'] = os.pathsep.join(paths + [os.environ.get('PYTHONPATH', '')])
    from .cpu_guard import install
    install()
    import torch
    return torch


def integer_keys(value):
    if isinstance(value, dict):
        return {int(k) if str(k).lstrip('-').isdigit() else k: integer_keys(v) for k, v in value.items()}
    if isinstance(value, list):
        return [integer_keys(v) for v in value]
    return value


def prepare(spec_path, output):
    spec = load_yaml(spec_path)
    catalog = load_yaml(artifact_root() / 'configs/paper_experiments.yaml')
    app = spec['application']
    definition = catalog['workloads'][app]
    p = dict(router_replicate_num=1, router_question_version='not_multiple_choice_question',
             evaluator_num=4, summarize_model='lmsys/vicuna-13b-v1.5',
             evaluator_model='meta-llama/Llama-2-70b-chat-hf', model_replicate_num=1,
             reqnum_mixed_blender=5000, max_token_num_mixed_blender=256)
    p.update(catalog['defaults'])
    p.update(definition['fixed'])
    p.update(spec.get('parameters', {}))
    p['test_case'] = definition['test_case']
    if 'reqnum' not in p or p['reqnum'] < 1:
        raise ValueError('positive reqnum required')
    output.mkdir(parents=True, exist_ok=False)
    torch = cpu_environment(p['seed'])
    benchmarks = Path(spec.get('benchmark_inputs', repo_root() / 'benchmarks')).resolve()
    data_root = Path(spec.get('data_root', repo_root().parent)).resolve()
    os.environ['MIL_DATA_ROOT'] = str(data_root)
    os.chdir(output)
    (output / 'my_dummy_requests').mkdir()
    sources = []
    if app in ('llm_blender', 'mixed'):
        sources.append(benchmarks / 'train_data_prepared.jsonl')
    if app == 'router':
        sources.append(benchmarks / f"router_bench_{p['router_question_version']}_dataset_with_responses.json")
    if app in ('chain_summary', 'mixed'):
        sources += [data_root / 'BooookScore' / name for name in
                    ('all_books_chunked_2048.pkl', 'booksum_data_chunked_2048.pkl')]
    for source in sources:
        if not source.is_file():
            raise FileNotFoundError(source)
    with (output / 'prepare.log').open('x') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        import schedule_multi_model as legacy
        legacy.BENCHMARK_ROOT, legacy.DATA_ROOT = benchmarks, data_root
        # Output-length records must be an explicit input; no implicit search in cwd.
        declared_outlens = spec.get('known_output_lengths')
        if declared_outlens:
            declared_outlens = (repo_root() / declared_outlens).resolve()
            sources.append(declared_outlens)
            legacy.get_output_len_file_name = lambda args: str(declared_outlens)
        else:
            legacy.get_output_len_file_name = lambda args: str(output / 'no_declared_output_lengths.json')
        args = SimpleNamespace(**p)
        values = legacy.get_schedule_setting(args, test_case=p['test_case'],
            version=p['router_question_version'], max_token_num=p['max_token_num'],
            specify_outlen=p['specify_outlen'], use_real_dataset=True,
            ratio_seed=p['ratio_seed'], ratio_set=p['ratio_set'], reqnum=p['reqnum'],
            evaluator_num=p['evaluator_num'], summarize_model=p['summarize_model'],
            evaluator_model=p['evaluator_model'])
        if len(values) != len(FIELDS):
            raise ValueError('legacy builder return contract changed')
        built = dict(zip(FIELDS, values))
        payload = {k: built[k] for k in KEEP}
        if spec.get('model_set') and set(spec['model_set']) != set(built['model_paths']):
            raise ValueError('declared model set disagrees with workload builder')
        lengths = {}
        for i, model in enumerate(built['model_paths']):
            funcs = legacy._get_req_len_funcs(built['test_cases'][i], built['versions'][i],
                built['max_token_nums'][i], built['specify_outlens'][i], p['noise_level'])
            lengths[i] = funcs[0](built['num_prompts'], i, model, built['inp_seq_ids_dict'][i])
        payload['input_lengths'] = lengths
        outpath = output / 'my_dummy_requests/my_dummy_requests_outlen.json'
        payload['known_output_lengths'] = json.loads(outpath.read_text()) if outpath.exists() else {}
        for i, known in enumerate(payload['specify_outlens']):
            if known and len(payload['known_output_lengths'].get(str(i), [])) != len(lengths[i]):
                raise ValueError(f'known output lengths missing for model node {i}')
        assert not torch.cuda.is_initialized()
    payload = native(payload)
    payload.update(schema_version=1, kind='normal_request_set', application=app, parameters=p,
                   source_files=[{'path': str(f), 'sha256': sha256_file(f)} for f in sources],
                   builder_sha256=sha256_file(repo_root() / 'benchmarks/schedule_multi_model.py'),
                   cuda_initialized=False,
                   successor_content='Estimated by the original cost simulator; not generated text.',
                   preparation='Original dataset sampling, tokenization, chunking and graph builders; no run logs.')
    atomic_write_json(output / 'requests.json', payload)
    return {'request_set': str(output / 'requests.json'), 'application': app,
            'model_nodes': len(payload['model_paths']), 'requests': payload['num_prompts'],
            'cuda_initialized': False}


def output_functions(bundle, distribution_path, noise):
    """Use original function bodies in a serializable namespace with explicit PDF."""
    import numpy as np
    from typing import List
    seed = bundle['parameters']['seed']
    pdf = read_literal_module(distribution_path, ('pdf_dict',))['pdf_dict']
    namespace = {'np': np, 'List': List, 'experiment_seed': lambda: seed,
                 'out_len_sampler2': SimpleNamespace(pdf_dict=pdf)}
    tree = ast.parse((repo_root() / 'benchmarks/output_length_sampler.py').read_text())
    tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    exec(compile(tree, '<frozen-output-sampler>', 'exec'), namespace)
    known = bundle['known_output_lengths']
    funcs_ns = {'np': np, 'List': List, 'output_length_sampler': SimpleNamespace(**{
        k: v for k, v in namespace.items() if callable(v) and k not in ('List',)}),
        'get_inplens': None, 'get_inplens_chain_summary': None,
        'get_outlens_router_bench': lambda model_id, name, lengths: np.asarray(known[model_id])}
    source = ast.parse((repo_root() / 'benchmarks/schedule_multi_model.py').read_text())
    source.body = [n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == '_get_req_len_funcs']
    exec(compile(source, '<original-length-functions>', 'exec'), funcs_ns)
    return [funcs_ns['_get_req_len_funcs'](test, version, maximum, known_flag, noise)
            for test, version, maximum, known_flag in zip(bundle['test_cases'], bundle['versions'],
                bundle['max_token_nums'], bundle['specify_outlens'])]


def serialize_schedule(best, method):
    """Preserve the scheduler's costed versus plan-only return contracts."""
    search = method['legacy']['search_method_baseline']
    if 'hybridflow' in search:
        return {'plans': [[str(plan) for plan in group.exec_plans] for group in best.plan_group_seq],
                'predicted_seconds': float(best.hybridflow_result[0])}
    if search in ('LightRule', 'LightRule_no_preemption', 'saturnLP_no_preemption'):
        return {'plans': [[str(plan) for plan in group.exec_plans] for group in best.plan_group_seq],
                'predicted_seconds': None,
                'cost_status': f'{search} does not expose a costed initial stage sequence'}
    return parse_plan(best.get_str_using_model_ids())


def schedule(request_set, method_id, output, cpus=2, cost_path=None, distribution_path=None, model_configs=None):
    bundle = integer_keys(json.loads(request_set.read_text()))
    if bundle.get('kind') != 'normal_request_set':
        raise ValueError('requires a normal request set')
    output.mkdir(parents=True, exist_ok=False)
    root, p = repo_root(), bundle['parameters']
    method = load_yaml(artifact_root() / 'configs/methods.yaml')['methods'][method_id]
    p = {**p, **method.get('args', {})}
    if p.get('outlen_known') and not all(bundle['specify_outlens']):
        raise ValueError('prepare a request set with known output lengths for this method')
    torch = cpu_environment(p['seed'])
    def stop_worker(signum, frame):
        raise TimeoutError('CPU scheduling time limit reached')
    signal.signal(signal.SIGTERM, stop_worker)
    os.environ['SAMPLE_SIZE'] = str(p.get('sample_size', 8))
    cost_path = cost_path or root / 'benchmarks/get_my_cost_table_directly_zxcpu.py'
    distribution_path = distribution_path or root / 'benchmarks/collect_output_lengths/no_robot/out_len_sampler_2.py'
    model_configs = model_configs or artifact_root() / 'inputs/model_configurations.json'
    os.chdir(output)
    started = time.monotonic()
    with (output / 'scheduler.log').open('x', buffering=1) as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        import ray
        import search_exec_plans as scheduler
        from experiment_reproducibility import seed_process
        from my_per_iter_latency_estimator import get_cost_table_from_serialized_data
        if model_configs.is_file():
            from .cpu_models import load_configurations
            configurations = load_configurations(model_configs)
            missing = [model for model in bundle['model_paths'] if (model, True, None) not in configurations]
            if missing:
                raise ValueError(f'missing frozen model configuration: {missing}')
            scheduler._hf_config_DICT.update(configurations)
            def frozen_config(model, trust_remote_code, revision=None):
                key = (model, trust_remote_code, revision)
                if key not in configurations:
                    raise ValueError(f'undeclared model configuration: {key}')
                return copy.deepcopy(configurations[key])
            import vllm.config as engine_config
            engine_config.get_config = frozen_config
            scheduler.get_config = frozen_config
        else:
            raise FileNotFoundError(f'freeze declared model metadata first: {model_configs}')
        data = read_literal_module(cost_path, COST_FIELDS)
        scheduler.get_my_cost_table_directly_zxcpu.cost_table = get_cost_table_from_serialized_data(
            *(data[k] for k in COST_FIELDS))
        funcs = output_functions(bundle, distribution_path, p['noise_level'])
        lengths = bundle['input_lengths']
        def inputs(num_prompts, model_id, model, ids):
            if ids != bundle['inp_seq_ids_dict'][model_id]:
                raise ValueError('request ID order changed')
            return lengths[model_id]
        out_edges = {}
        for target, sources in bundle['input_edges'].items():
            for source in sources:
                if source >= 0:
                    out_edges.setdefault(source, []).append(target)
        seed_process(p['seed'])
        ray.init(address='local', num_cpus=cpus, num_gpus=0, include_dashboard=False,
                 object_store_memory=256 * 1024**2, _temp_dir=tempfile.mkdtemp(prefix='mil-cpu-ray-'),
                 log_to_driver=False)
        try:
            best = scheduler.get_best_model_schedule(**method['legacy'],
                check_gap=bundle['check_gap'], sort_input=bundle['sort_input'], model_paths=bundle['model_paths'],
                num_prompts=bundle['num_prompts'], inp_seq_ids_dict=bundle['inp_seq_ids_dict'],
                out_req_id_mapping=bundle['out_req_id_mapping'], inp_req_from_which_models=bundle['inp_req_ids'],
                inp_req_from_which_model_which_out_reqs=bundle['inp_req_from_which_model_which_out_reqs'],
                independent_srcs=bundle['independent_srcs'], inp_generators=[inputs] * len(funcs),
                inp_mergers=[f[1] for f in funcs], outlen_generators=[f[2] for f in funcs],
                prompt_templates_lens=bundle['prompt_templates_lens'], out_edge_dict=out_edges,
                sample_config=(1, 1, -1, 0), trust_remote_code=True, revision=None,
                gpu_name='A100-80G', tot_gpu_num=8, byte_per_gpu=80 * 1024**3, data_byte=2,
                max_group_seq_num=p['max_group_seq_num'], top_k=p['top_k'],
                similar_threshold=p['similar_threshold'], fully_connected_gpu_unit=2, machine_name='zxcpu')
            actual = serialize_schedule(best, method)
            assert not torch.cuda.is_initialized()
            extension_origin = getattr(sys.modules.get('vllm._C'), '__file__', None)
            if extension_origin != '<CPU-only forbidden GPU operators>':
                raise RuntimeError('CPU scheduling GPU-operator guard was replaced')
        finally:
            ray.shutdown()
    report = {'schema_version': 1, 'status': 'completed_cpu_scheduling', 'application': bundle['application'],
              'method': method_id, 'parameters': p, 'actual': actual, 'cpu_workers': cpus,
              'cuda_initialized': False, 'wall_seconds': time.monotonic() - started,
              'compiled_gpu_extension_loaded': False,
              'gpu_operator_module': extension_origin,
              'imported_modules': {name: str(Path(sys.modules[name].__file__).resolve())
                                   for name in ('vllm', 'search_exec_plans', 'fake_scheduling')},
              'request_set_sha256': sha256_file(request_set),
              'resources': {str(path): sha256_file(path) for path in (cost_path, distribution_path, model_configs)},
              'resource_hashes': {'cost_table': sha256_file(cost_path),
                                  'output_lengths': sha256_file(distribution_path),
                                  'model_configurations': sha256_file(model_configs)},
              'source_hashes': {name: sha256_file(root / 'benchmarks' / name) for name in
                  ('search_exec_plans.py', 'fake_scheduling.py', 'schedule_multi_model.py', 'output_length_sampler.py')},
              'scope': 'Initial plans and predicted costs; no inference or runtime speedup claim.'}
    atomic_write_json(output / 'result.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--spec', type=Path, required=True)
    prep.add_argument('--output', type=Path, required=True)
    for name in ('run', '_worker'):
        run = sub.add_parser(name)
        run.add_argument('--requests', type=Path, required=True)
        run.add_argument('--method', default='mil')
        run.add_argument('--output', type=Path, required=True)
        run.add_argument('--cpus', type=int, default=2)
        run.add_argument('--cost-table', type=Path)
        run.add_argument('--distribution', type=Path)
        run.add_argument('--model-configs', type=Path)
        run.add_argument('--timeout', type=int, default=None,
                         help='Optional diagnostic time limit in seconds; default: no limit.')
    args = parser.parse_args()
    if args.command == 'prepare':
        result = prepare(args.spec.resolve(), args.output.resolve())
    else:
        if not 1 <= args.cpus <= 8:
            parser.error('cpus must be between 1 and 8')
        if args.timeout is not None and args.timeout < 1:
            parser.error('timeout must be positive when explicitly supplied')
        if args.command == 'run':
            if args.output.exists():
                raise FileExistsError(args.output)
            command = [sys.executable, '-m', 'artifact.mil_artifact.cpu_schedule', '_worker'] + sys.argv[2:]
            process = subprocess.Popen(command, start_new_session=True)
            timed_out = False
            try:
                code = process.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                process.send_signal(signal.SIGTERM)
                try:
                    code = process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    code = process.wait()
            if code:
                args.output.mkdir(parents=True, exist_ok=True)
                atomic_write_json(args.output / 'failure.json', {
                    'status': 'timeout' if timed_out else 'failed', 'exit_code': code,
                    'method': args.method, 'timeout_seconds': args.timeout,
                    'request_set_sha256': sha256_file(args.requests),
                    'automatic_solver_fallback': False})
            return 1 if code else 0
        result = schedule(args.requests.resolve(), args.method, args.output.resolve(), args.cpus,
                          args.cost_table.resolve() if args.cost_table else None,
                          args.distribution.resolve() if args.distribution else None,
                          args.model_configs.resolve() if args.model_configs else None)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
