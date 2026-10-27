"""CPU event simulation of the original runtime placement functions."""
from __future__ import annotations
import argparse
import ast
import contextlib
import io
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from .config import repo_root
from .hashing import sha256_file
from .schema import atomic_write_json


class PlanState:
    def __init__(self, model, gpus, stage, dependencies=()):
        self.exec_plan = SimpleNamespace(model=SimpleNamespace(model_id=model, input_model_ids=list(dependencies)),
                                         num_worker=gpus, dp_size=1, get_key=lambda: (gpus,))
        self.stage_i, self.last_exec_plan_for_the_model = stage, True
        self.need_prepare_infer_env, self.launched, self.comp_gpus = True, False, []

    def get_comp_gpus(self):
        return self.comp_gpus

    def set_comp_gpus(self, gpus):
        self.comp_gpus = list(gpus)


def original_functions(finished):
    source = repo_root() / 'benchmarks/schedule_multi_model.py'
    names = {'_get_the_first_plan_state_when_sorted_by_gpu_num_and_topology',
             '_get_the_first_plan_state_when_sorted_by_topology', '_try_to_load_exec_plans',
             '_has_model_finished', '_get_the_next_round_exec_plan_schedule',
             '_get_the_next_round_exec_plan_schedule_no_runtime_adjustment'}
    tree = ast.parse(source.read_text())
    tree.body = [ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)] + [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    ast.fix_missing_locations(tree)
    namespace = {'np': np, 'SHARED_CONTECT': SimpleNamespace(query_finish_status=lambda model: model in finished)}
    exec(compile(tree, str(source), 'exec'), namespace)
    return namespace


def run_cases():
    cases = []
    for adjustment in (True, False):
        finished = {1}
        functions = original_functions(finished)
        a, b, c = PlanState(0, 4, 0), PlanState(1, 4, 0), PlanState(2, 4, 1)
        for plan, gpus in ((a, range(4)), (b, range(4, 8))):
            plan.launched = True
            plan.set_comp_gpus(gpus)
        name = '_get_the_next_round_exec_plan_schedule' + ('' if adjustment else '_no_runtime_adjustment')
        with contextlib.redirect_stdout(io.StringIO()):
            launched, candidates, stage = functions[name]([a, b], [c], 1, 8, [[a, b], [c]], {},
                                                         {0: 0, 1: 1, 2: 2}, 'greedy_saturn_LPT')
        actual = [p.exec_plan.model.model_id for p in launched]
        expected = [0, 2] if adjustment else [0]
        cases.append({'case': 'early_finish_adjustment_' + str(adjustment).lower(),
                      'finished_models': [1], 'launched': actual, 'expected': expected,
                      'gpu_assignments': {p.exec_plan.model.model_id: p.comp_gpus for p in launched},
                      'passed': actual == expected})
        if not adjustment:
            finished.add(0)
            with contextlib.redirect_stdout(io.StringIO()):
                launched, _, _ = functions[name](launched, candidates, stage, 8, [[a, b], [c]], {},
                                                 {0: 0, 1: 1, 2: 2}, 'greedy_saturn_LPT')
            actual = [p.exec_plan.model.model_id for p in launched]
            cases.append({'case': 'stage_barrier_released', 'finished_models': [0, 1],
                          'launched': actual, 'expected': [2], 'passed': actual == [2]})
    for method, expected in [('greedy_saturn_LPT', [1, 0]), ('LightRule', [0, 1]),
                             ('LightRule_our_cost_model', [1, 0])]:
        functions = original_functions(set())
        plans = [PlanState(0, 2, 0), PlanState(1, 4, 0), PlanState(2, 4, 0, [0])]
        launched, pending = [], []
        with contextlib.redirect_stdout(io.StringIO()):
            functions['_try_to_load_exec_plans'](plans, launched, pending, {}, list(range(6)), [], method)
        actual = [p.exec_plan.model.model_id for p in launched]
        cases.append({'case': 'placement_' + method, 'launched': actual, 'expected': expected,
                      'pending': [p.exec_plan.model.model_id for p in pending],
                      'passed': actual == expected and [p.exec_plan.model.model_id for p in pending] == [2]})
    return {'schema_version': 1, 'kind': 'synthetic_completion_events', 'gpu_execution': False,
            'source_sha256': sha256_file(repo_root() / 'benchmarks/schedule_multi_model.py'),
            'cases': cases, 'passed': all(c['passed'] for c in cases),
            'scope': 'Actual placement branches with synthetic completion events; not a measured speedup.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    result = run_cases()
    atomic_write_json(args.output / 'events.json', result)
    print(f"{len(result['cases'])} cases; passed={result['passed']}")
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
