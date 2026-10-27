"""Exercise the original dispatcher with stubbed searches, without GPU or MILP work."""
from __future__ import annotations

import ast
import unittest
from unittest import mock

from artifact.mil_artifact.config import artifact_root, load_yaml, repo_root


class MethodDispatchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = repo_root() / 'benchmarks/search_exec_plans.py'
        tree = ast.parse(source.read_text())
        owner = next(node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
                     and any(isinstance(child, ast.FunctionDef)
                             and child.name == 'get_candidate_plan_groups_dispatch'
                             for child in node.body))
        cls.implementations = [node.name for node in owner.body if isinstance(node, ast.FunctionDef)]
        function = next(node for node in owner.body if isinstance(node, ast.FunctionDef)
                        and node.name == 'get_candidate_plan_groups_dispatch')
        module = ast.Module(body=[ast.ImportFrom(module='__future__',
            names=[ast.alias(name='annotations')], level=0), function], type_ignores=[])
        ast.fix_missing_locations(module)
        namespace = {}
        exec(compile(module, str(source), 'exec'), namespace)
        cls.dispatch = staticmethod(namespace[function.name])
        cls.methods = load_yaml(artifact_root() / 'configs/methods.yaml')['methods']

    def test_paper_methods_reach_expected_original_implementations(self):
        expected = {
            'mil': ('ours', 'greedy_saturn_LPT', 'get_candidate_plan_groups_greedy_best_exec_plan_first_LPT'),
            'mil_chain_summary': ('ours', 'greedy_saturn', 'get_candidate_plan_groups_greedy_best_exec_plan_first'),
            'max': ('max_gpu', 'ours', 'get_candidate_plan_groups'),
            'min': ('ours', 'min_gpu_useAll', 'get_candidate_plan_groups_min_gpu_use_all_gpus'),
            'min_r': ('ours', 'min_gpu_useAll_sampleModels', 'get_candidate_plan_groups_min_gpu_use_all_gpus'),
            'saturn_star': ('ours', 'saturnLP', 'get_candidate_plan_groups_saturn_LP'),
            'light_rule': ('ours', 'LightRule', 'get_candidate_plan_groups_LightRule'),
            'light_rule_star': ('ours', 'LightRule_our_cost_model', 'get_candidate_plan_groups_LightRule_our_cost_model'),
            'hybridflow': ('ours', 'hybridflow_model_parallel_old', 'get_candidate_plan_groups_HybridFlow_old'),
            'hybridflow_star': ('ours', 'hybridflow_model_parallel', 'get_candidate_plan_groups_HybridFlow'),
            'oracle': ('ours', 'oracle', 'get_candidate_plan_groups_oracle'),
            'mil_uniform_output': ('ours', 'greedy_saturn_LPT', 'get_candidate_plan_groups_greedy_best_exec_plan_first_LPT'),
            'mil_no_preemption': ('ours', 'greedy_saturn_LPT_no_preemption', 'get_candidate_plan_groups_greedy_best_exec_plan_first_LPT'),
        }
        for method, (generator, search, implementation) in expected.items():
            with self.subTest(method=method):
                legacy = self.methods[method]['legacy']
                self.assertEqual(legacy, {'gen_execplans_baseline': generator,
                                          'search_method_baseline': search})
                self.assertIn(implementation, self.implementations)
                target = mock.Mock(spec=self.implementations)
                cost, plans = object(), []
                result = self.dispatch(target, **legacy, check_gap=1600, sort_input=True,
                    last_stage_exec_plans=plans, cost_table=cost, tot_gpu_num=8,
                    byte_per_gpu=80 * 1024**3, top_k=20, fully_connected_gpu_unit=2)
                forwarded = [generator]
                if method != 'max':
                    forwarded.append('hybridflow_model_parallel' if method == 'hybridflow' else search)
                forwarded += [1600, True, plans, cost, 8, 80 * 1024**3, 20, 2]
                self.assertEqual(target.mock_calls, [getattr(mock.call, implementation)(*forwarded)])
                self.assertIs(result, getattr(target, implementation).return_value)

    def test_unknown_method_cannot_silently_fall_back(self):
        target = mock.Mock(spec=self.implementations)
        with self.assertRaises(AssertionError):
            self.dispatch(target, 'ours', 'not_a_registered_method', 1600, True, [], object())
        self.assertEqual(target.mock_calls, [])


if __name__ == '__main__':
    unittest.main()
