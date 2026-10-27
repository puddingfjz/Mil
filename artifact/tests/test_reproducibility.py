from __future__ import annotations

import ast
import importlib.util
import random
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from artifact.mil_artifact.config import repo_root


def _load_module():
    path = repo_root() / "benchmarks" / "experiment_reproducibility.py"
    specification = importlib.util.spec_from_file_location(
        "artifact_experiment_reproducibility", path
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


class ExperimentReproducibilityTest(unittest.TestCase):
    def test_seed_process_replays_python_and_numpy_rngs(self) -> None:
        module = _load_module()
        with mock.patch.dict("os.environ", {"MIL_EXPERIMENT_SEED": "17"}):
            self.assertEqual(module.seed_process(), 17)
            first = (random.random(), np.random.random())
            module.seed_process()
            second = (random.random(), np.random.random())
        self.assertEqual(first, second)

    def test_seed_validation_fails_closed(self) -> None:
        module = _load_module()
        with self.assertRaisesRegex(ValueError, "must be an integer"):
            module.experiment_seed({"MIL_EXPERIMENT_SEED": "not-a-seed"})
        with self.assertRaisesRegex(ValueError, "between 0"):
            module.experiment_seed({"MIL_EXPERIMENT_SEED": "-1"})

    def test_critical_legacy_execution_handlers_propagate_failures(self) -> None:
        source_path = repo_root() / "benchmarks" / "schedule_multi_model.py"
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        critical_functions = {
            "start_a_model_inference_child_process",
            "start_a_model_inference",
            "_search_best_scheduling_with_another_process",
            "main_with_preemption",
        }
        functions = {
            node.name: node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name in critical_functions
        }
        self.assertEqual(set(functions), critical_functions)
        for function_name, function in functions.items():
            handlers = [
                node for node in ast.walk(function) if isinstance(node, ast.ExceptHandler)
            ]
            self.assertTrue(handlers, function_name)
            for handler in handlers:
                self.assertTrue(
                    any(isinstance(node, ast.Raise) for node in ast.walk(handler)),
                    f"{function_name} swallows an execution failure",
                )


if __name__ == "__main__":
    unittest.main()
