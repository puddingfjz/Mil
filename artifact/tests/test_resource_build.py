from __future__ import annotations

import ast
import contextlib
import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from artifact.mil_artifact.config import repo_root
from artifact.mil_artifact.resource_build import (
    COST_FIELDS, TABLES, build_cost_entry, build_resources, compare_values,
    cost_module_text, distribution_from_pairs, load_spec, main,
    plan_acquisition, read_cost_table, read_lengths, read_literal_module,
    select_endpoints, write_bundle,
)


ROOT = repo_root()
FIXTURES = ROOT / "artifact/tests/fixtures/resources"
SIMULATED = ROOT / "artifact/configs/resources/simulated.yaml"


def legacy_cpu_class():
    """Compile only trusted CPU definitions for an independent parity oracle.

    Never execute top-level legacy imports or instantiate model/engine objects.
    Production resource construction does not use exec or import legacy code.
    """
    source = ast.parse((ROOT / "benchmarks/my_per_iter_latency_estimator.py").read_text())
    definitions = [n for n in source.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))
                   and n.name in {"CostTable", "get_tp_size_from_logfile"}]
    metrics = ast.parse((ROOT / "vllm/engine/metrics.py").read_text())
    flops = next(n for n in metrics.body if isinstance(n, ast.FunctionDef) and n.name == "cal_flops")
    logger = next(n for n in metrics.body if isinstance(n, ast.ClassDef) and n.name == "MyThroughputLogger")
    logger.body = [n for n in logger.body if isinstance(n, ast.FunctionDef) and n.name.startswith("get_seqnum_")]
    module = ast.Module(body=[flops, logger] + definitions, type_ignores=[])
    namespace = {"np": np, "defaultdict": __import__("collections").defaultdict,
                 "model_coeff_database": SimpleNamespace(model_coeffs={("synthetic/toy-model", 1): 1e12})}
    exec(compile(module, "<legacy-cpu-definitions>", "exec"), namespace)
    return namespace["CostTable"]


class ResourceBuildTest(unittest.TestCase):
    def test_distribution_censoring_and_probability_mass(self):
        pdf, details = distribution_from_pairs([[2, 1], [2, 2], [2, 6], [4, 2]], 8)
        self.assertEqual(pdf, [0.25, 0.5, 0, 0, 0, 0, 0, 0.25])
        self.assertEqual(details["censored_requests"], 1)
        # All requests hit the context limit: no false observed completion.
        censored, _ = distribution_from_pairs([[2, 6], [4, 4]], 8)
        self.assertEqual(censored, [0] * 7 + [1])

    def test_invalid_length_records_are_rejected(self):
        for pairs in ([], [[2, 0]], [[2, 7]], [[True, 2]], [[1.5, 2]]):
            with self.subTest(pairs=pairs), self.assertRaises(ValueError):
                distribution_from_pairs(pairs, 8)

    def test_ambiguous_and_truncated_output_logs_require_explicit_selection(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "lengths.txt"
            path.write_text("output_lens = [[2, 1]]\noutput_lens = [[2, 2]]\n")
            with self.assertRaisesRegex(ValueError, "select record_block"):
                read_lengths(path, None, 1)
            self.assertEqual(read_lengths(path, 1, 1), [[2, 2]])
            with self.assertRaisesRegex(ValueError, "expected 2 requests"):
                read_lengths(path, 1, 2)

    def test_endpoint_selection_matches_legacy_linear_and_piecewise(self):
        legacy_type = legacy_cpu_class()
        legacy = legacy_type.__new__(legacy_type)
        # Repeats have different x values; preserve the measured fastest pair.
        rows = [(1, 2), (1.01, 3), (2, 2), (2.01, 2.5), (3, 4), (3.01, 5), (4, 6), (4.01, 7)]
        for piece in (1, 2):
            expected = {1: copy.deepcopy(rows)}
            legacy._select_cost_model_samples_from_records(expected, piece, 2, 0.001, 0)
            actual = select_endpoints(rows, 2, piece, 0.001)
            self.assertEqual(expected[1], actual)
        with self.assertRaises(ValueError):
            select_endpoints(rows[:-1], 2, 1, 0.001)

    def test_cost_log_reconstruction_matches_unmodified_legacy_math(self):
        entry = load_spec(SIMULATED)["cost"]["entries"][0]
        actual, _ = build_cost_entry(FIXTURES / "cost.txt", entry, [2, 1], 1e12)
        legacy_type = legacy_cpu_class()
        metadata = [{} for _ in COST_FIELDS]
        metadata[-2][("synthetic/toy-model", True, None)] = (2, 1)
        legacy = legacy_type([], [], [], None, metadata=metadata)
        with contextlib.redirect_stdout(io.StringIO()):
            legacy.init_table("synthetic/toy-model", (1, 0.9, 2, 0), (1, 1, -1, 0),
                              str(FIXTURES / "cost.txt"), 2, 0.001, True)
        for field in TABLES:
            self.assertEqual(next(iter(getattr(legacy, field).values())), actual[field])
        # Check actual estimator consumption of all three latency components.
        expected_seconds = 0.4 + 0.01 + 0.02
        self.assertAlmostEqual(float(legacy.estimate_cost([1], [4], [4], [1], True,
                              "synthetic/toy-model", (1, 0.9, 2, 0), (1, 1, -1, 0), True)[0]), expected_seconds)

    def test_tensor_parallel_mismatch_is_rejected(self):
        entry = load_spec(SIMULATED)["cost"]["entries"][0]
        entry["exec_plan"][0] = 2
        with self.assertRaisesRegex(ValueError, "tensor parallel"):
            build_cost_entry(FIXTURES / "cost.txt", entry, [2, 1], 1e12)

    def test_literal_reader_preserves_updates_but_never_executes_imports(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "resource.py"
            path.write_text("pdf_dict = {}\npdf_dict.update({'model': [0.25, 0.75]})\n")
            self.assertEqual(read_literal_module(path, ("pdf_dict",))["pdf_dict"]["model"], [0.25, 0.75])
            path.write_text("import torch\npdf_dict = {}\n")
            with self.assertRaisesRegex(ValueError, "unsupported resource syntax"):
                read_literal_module(path, ("pdf_dict",))

    def test_new_bundle_roundtrips_without_appending_or_changing_inputs(self):
        spec = load_spec(SIMULATED)
        original = (FIXTURES / "cost.txt").read_bytes()
        files, report = build_resources(spec, ROOT)
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "new"
            write_bundle(output, files, report)
            data = read_cost_table(output / "cost_table.py")
            self.assertEqual(next(iter(data["prefill_table"].values()))[1][0][1], 0.4)
            pdfs = read_literal_module(output / "output_lengths.py", ("pdf_dict",))["pdf_dict"]
            self.assertEqual(pdfs["toy-model"], [0.25, 0.5, 0, 0, 0, 0, 0, 0.25])
            before = (output / "report.json").read_bytes()
            with self.assertRaises(FileExistsError):
                write_bundle(output, files, report)
            self.assertEqual((output / "report.json").read_bytes(), before)
            roundtrip = Path(temp) / "roundtrip.py"
            roundtrip.write_text(cost_module_text(data))
            self.assertEqual(read_cost_table(roundtrip), data)
        self.assertEqual(original, (FIXTURES / "cost.txt").read_bytes())

    def test_missing_logs_fail_before_any_output_is_created(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "missing"
            with contextlib.redirect_stderr(io.StringIO()):
                status = main(["build", "--config", str(SIMULATED), "--input-root", temp, "--output", str(output)])
            self.assertEqual(status, 2)
            self.assertFalse(output.exists())

    def test_plan_is_data_only_and_needs_no_model_or_measurement_files(self):
        spec = load_spec(SIMULATED, ["synthetic/toy-model"])
        with mock.patch("subprocess.Popen", side_effect=AssertionError("must never launch a process")):
            plan = plan_acquisition(spec, ROOT, "python3")
        self.assertEqual(len(plan["jobs"]), 2)
        self.assertTrue(all(job["executed"] is False for job in plan["jobs"]))
        self.assertTrue(all(job["stdout"] != job["source_log"] for job in plan["jobs"]))
        self.assertIn("--max-model-len", plan["jobs"][1]["argv"])
        with self.assertRaisesRegex(ValueError, "unknown models"):
            load_spec(SIMULATED, ["missing/model"])

    def test_plan_writes_followup_config_pointing_only_to_new_logs(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "plan"
            with contextlib.redirect_stdout(io.StringIO()):
                status = main(["plan", "--config", str(SIMULATED), "--output", str(output)])
            self.assertEqual(status, 0)
            planned = load_spec(output / "build_config.json")
            for section in ("cost", "output_lengths"):
                log = Path(planned[section]["entries"][0]["log"])
                self.assertEqual(log.parent, output.resolve() / "measurements")
                self.assertFalse(log.exists())

    def test_plan_does_not_silently_ignore_unsupported_profiler_settings(self):
        spec = load_spec(SIMULATED)
        spec["cost"]["entries"][0]["profile_num"] = 3
        with self.assertRaisesRegex(ValueError, "profiler fixes"):
            plan_acquisition(spec, ROOT, "python3")

    def test_request_limit_is_distinct_from_retained_count(self):
        spec = load_spec(SIMULATED)
        spec["output_lengths"]["entries"][0].update(num_prompts=10000, expected_requests=4)
        job = plan_acquisition(spec, ROOT, "python3")["jobs"][1]
        self.assertEqual(job["argv"][job["argv"].index("--num-prompts") + 1], "10000")

    def test_cli_cannot_import_gpu_frameworks_even_with_visible_devices(self):
        # A subprocess isolates the guard from unrelated test modules.
        script = """
import sys
class NoGpuImports:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'vllm', 'ray'}:
            raise AssertionError('GPU/runtime import attempted: ' + fullname)
sys.meta_path.insert(0, NoGpuImports())
from artifact.mil_artifact.resource_build import main
raise SystemExit(main(sys.argv[1:]))
"""
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "cpu"
            result = subprocess.run([sys.executable, "-B", "-c", script, "build", "--config", str(SIMULATED),
                                     "--output", str(output)], cwd=ROOT, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads((output / "report.json").read_text())
            self.assertEqual(report["gpu_modules_imported"], [])

    def test_comparison_detects_missing_keys_and_changed_values(self):
        self.assertFalse(compare_values({"a": [1]}, {})["passed"])
        self.assertFalse(compare_values([1, 2], [1, 3])["passed"])
        self.assertFalse(compare_values([1], [float("nan")])["passed"])
        self.assertTrue(compare_values([1, 2], [1, 2])["passed"])


if __name__ == "__main__":
    unittest.main()
