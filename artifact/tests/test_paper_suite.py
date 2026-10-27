from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from artifact.mil_artifact.config import expand_experiment, repo_root
from artifact.mil_artifact.paper_suite import compile_suite, MAIN_ITEMS, REQUIRED_ITEMS


class PaperSuiteTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog, cls.profile, cls.report, cls.expanded = compile_suite()

    def test_complete_inventory_and_unique_main_coordinates(self):
        self.assertEqual(set(self.report["items"]), REQUIRED_ITEMS)
        main = {m["case_id"] for name in MAIN_ITEMS for m in self.report["items"][name]["members"]}
        self.assertEqual(len(main), 28)
        self.assertEqual(len(self.report["items"]["figure_9"]["members"]), 1)
        self.assertEqual(len(self.report["items"]["runtime_adjustment"]["members"]), 56)

    def test_component_settings_are_not_old_ablation_aliases(self):
        items, cases = self.report["items"], self.report["cases"]
        self.assertEqual(len(items["table_2"]["members"]), 3)
        self.assertEqual(len(items["table_3"]["members"]), 14)
        self.assertEqual({m["variant"] for m in items["table_3"]["members"]},
                         {"Exact", "0", "1e-5", "1e-4", "1e-3", "1e-2", "Rand"})
        for member in items["table_3"]["members"]:
            self.assertEqual(cases[member["case_id"]]["parameters"]["max_token_num"], 4096)
        for member in items["table_4"]["members"]:
            case = cases[member["case_id"]]
            self.assertEqual(case["model_paths"], self.catalog["model_sets"]["oracle"])
            self.assertEqual(case["parameters"]["oracle_model_num"], 3)

    def test_paper_and_oracle_have_no_outer_time_limit(self):
        self.assertIsNone(self.profile['task_timeout_seconds'])
        self.assertIsNone(expand_experiment('paper').profile['task_timeout_seconds'])
        self.assertIsNone(self.catalog['items']['table_4']['cpu_timeout_seconds'])

    def test_min_r_three_seeds_only_when_at_least_eight_models(self):
        by_case = {}
        for task in self.expanded.tasks:
            if task.method_id == "min_r":
                by_case.setdefault(task.workload_id, []).append(task.parameters["seed"])
        for case_id, seeds in by_case.items():
            expected = [0, 1, 2] if len(self.report["cases"][case_id]["model_paths"]) >= 8 else [0]
            self.assertEqual(seeds, expected)

    def test_paper_profile_is_executable_as_dry_run_and_extensions_remain(self):
        direct = expand_experiment("paper")
        self.assertEqual([t.task_id for t in direct.tasks], [t.task_id for t in self.expanded.tasks])
        self.assertTrue(all(t.executable for t in direct.tasks))
        self.assertEqual(len(expand_experiment("main").tasks), 1359)
        self.assertEqual(len(expand_experiment("ablation").tasks), 18)

    def test_rand_samples_legal_interval_instead_of_clipping_a_larger_distribution(self):
        source = ast.parse((repo_root() / "benchmarks/output_length_sampler.py").read_text())
        function = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == "sample_uniform_output_lengths")
        namespace = {"np": np, "experiment_seed": lambda: 7,
                     "out_len_sampler2": SimpleNamespace(pdf_dict={"model": [0.01] * 100})}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "<uniform-output>", "exec"), namespace)
        sample = namespace[function.name]("model", [10] * 10000, 5)
        self.assertEqual(set(sample), {1, 2, 3, 4, 5})
        self.assertLess(max(Counter(sample).values()), 2300)
        np.testing.assert_array_equal(sample, namespace[function.name]("model", [10] * 10000, 5))


if __name__ == "__main__":
    unittest.main()
