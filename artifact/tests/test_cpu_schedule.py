import ast
import json
from pathlib import Path
import unittest
import tempfile
import hashlib
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from artifact.mil_artifact.config import artifact_root, load_yaml, repo_root
from artifact.mil_artifact.cpu_schedule import integer_keys, output_functions, serialize_schedule
from artifact.mil_artifact.runtime_events import run_cases
from artifact.mil_artifact.compare_schedules import compare_frozen


class CPUScheduleTests(unittest.TestCase):
    def test_all_registered_methods_serialize_original_return_shapes(self):
        # Execute only the original return container, with no scheduler/GPU imports.
        source = repo_root() / 'benchmarks/search_exec_plans.py'
        tree = ast.parse(source.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'MyExecPlanGroupSeq')
        module = ast.Module(body=[ast.ImportFrom(module='__future__',
            names=[ast.alias(name='annotations')], level=0), cls], type_ignores=[])
        ast.fix_missing_locations(module)
        namespace = {}
        exec(compile(module, str(source), 'exec'), namespace)
        plan = SimpleNamespace(model=SimpleNamespace(get_base_model_ids=lambda: [0]),
                               get_key=lambda: (1, 0.9, 2, 0, 1))
        group = SimpleNamespace(exec_plans=[plan])
        methods = load_yaml(artifact_root() / 'configs/methods.yaml')['methods']
        no_cost = {'light_rule', 'light_rule_no_preemption', 'saturn_star_no_preemption'}
        for method_id, method in methods.items():
            with self.subTest(method=method_id):
                best = namespace['MyExecPlanGroupSeq'](6.0, [group], [3.0], [])
                if method_id in no_cost:
                    best.set_plan_group_and_time([group], None, None)
                    # This is the crash the adapter must avoid, not a mocked exception.
                    with self.assertRaises(TypeError):
                        best.get_str_using_model_ids()
                    with patch.object(best, 'get_str_using_model_ids', side_effect=AssertionError('plan only')):
                        actual = serialize_schedule(best, method)
                    self.assertIsNone(actual['predicted_seconds'])
                    self.assertNotIn('stage_times', actual)
                    self.assertIn('cost_status', actual)
                    self.assertEqual(actual['plans'], [[str(plan)]])
                elif 'hybridflow' in method['legacy']['search_method_baseline']:
                    best.hybridflow_result = (4.0,)
                    actual = serialize_schedule(best, method)
                    self.assertEqual(actual['predicted_seconds'], 4.0)
                else:
                    actual = serialize_schedule(best, method)
                    self.assertEqual(actual['predicted_seconds'], 3.0)
                    self.assertEqual(actual['stage_times'], [3.0])
                    self.assertEqual(actual['plans'], [[([0], (1, 0.9, 2, 0, 1))]])
                json.dumps(actual, allow_nan=False)

    def test_portable_cpu_reference_checks_inputs_costs_and_resources(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request, result, reference = (root / name for name in ('requests.json', 'result.json', 'reference.json'))
            request.write_text('{}')
            digest = hashlib.sha256(request.read_bytes()).hexdigest()
            expected = {'method': 'mil', 'request_set_sha256': digest, 'parameters': {'seed': 0},
                        'resources': {'/original/cost.py': 'cost-hash'},
                        'actual': {'plans': [[1]], 'stage_times': [3.0], 'predicted_seconds': 3.0}}
            reference.write_text(json.dumps({'final_cases': {'case': {'method': 'mil', 'exit_code': 0, 'report': expected}}}))
            actual = json.loads(json.dumps(expected))
            actual['resources'] = {'/new/renamed-table.py': 'cost-hash'}
            result.write_text(json.dumps(actual))
            self.assertTrue(compare_frozen(request, result, reference)['passed'])
            actual['actual']['stage_times'] = [4.0]
            result.write_text(json.dumps(actual))
            self.assertFalse(compare_frozen(request, result, reference)['passed'])
            request.write_text('{"other_requests": true}')
            with self.assertRaisesRegex(ValueError, 'different request set'):
                compare_frozen(request, result, reference)

    def test_completion_events(self):
        report = run_cases()
        self.assertTrue(report['passed'], report)
        self.assertEqual(len(report['cases']), 6)

    def test_graph_key_roundtrip(self):
        graph = {0: {-1: [0, 1]}, 1: {0: [1]}}
        self.assertEqual(integer_keys(json.loads(json.dumps(graph))), graph)

    def test_original_sampler_and_known_lengths(self):
        bundle = {'parameters': {'seed': 0}, 'test_cases': ['general', 'chain-summary', 'router'],
                  'versions': [None] * 3, 'max_token_nums': [256, 900, 4096],
                  'specify_outlens': [False, False, True], 'known_output_lengths': {2: [12, 40]}}
        path = repo_root() / 'benchmarks/collect_output_lengths/no_robot/out_len_sampler_2.py'
        funcs = output_functions(bundle, path, 0)
        lengths = [100, 200]
        sampled = funcs[0][2](0, 'vicuna-13b-v1.5', lengths)
        np.testing.assert_array_equal(sampled, funcs[0][2](0, 'vicuna-13b-v1.5', lengths))
        self.assertTrue(all(0 < x <= 256 for x in sampled))
        self.assertTrue(all(0 < x <= 900 for x in funcs[1][2](1, 'vicuna-13b-v1.5', lengths)))
        np.testing.assert_array_equal(funcs[2][2](2, 'unused', lengths), [12, 40])

    def test_rand_uses_legal_uniform_lengths_and_keeps_seed(self):
        bundle = {'parameters': {'seed': 0}, 'test_cases': ['general'],
                  'versions': [None], 'max_token_nums': [4096],
                  'specify_outlens': [False], 'known_output_lengths': {}}
        path = repo_root() / 'benchmarks/collect_output_lengths/no_robot/out_len_sampler_2.py'
        sampler = output_functions(bundle, path, -1)[0][2]
        lengths = [100] * 100
        first = sampler(0, 'vicuna-13b-v1.5', lengths)
        np.testing.assert_array_equal(first, sampler(0, 'vicuna-13b-v1.5', lengths))
        self.assertGreater(len(set(first)), 20)
        self.assertTrue(all(1 <= x <= 3996 for x in first))
