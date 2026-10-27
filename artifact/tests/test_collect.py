from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from artifact.mil_artifact.collect import collect_run, parse_legacy_log
from artifact.mil_artifact.hashing import sha256_file
from artifact.mil_artifact.schema import read_json, read_jsonl


class LegacyLogParserTest(unittest.TestCase):
    def test_parses_primary_timing_fields(self) -> None:
        log = """\
begin search: --abs: 100.0
finish fusing similar models: --abs: 103.5
Total search time: 9.25
Total time for preparation before search: 3.5
Best group seq: [[example]] [1.0] 42.75 3.5
total running time: 51.125s ---abs: 999.0
"""
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "stdout.log"
            path.write_text(log, encoding="utf-8")
            metrics, errors = parse_legacy_log(path)
        self.assertEqual(errors, [])
        self.assertEqual(metrics["predicted_running_time_seconds"], 42.75)
        self.assertEqual(metrics["real_running_time_seconds"], 51.125)
        self.assertEqual(metrics["search_time_seconds"], 9.25)
        self.assertEqual(metrics["preparation_time_seconds"], 3.5)
        self.assertEqual(metrics["total_time_seconds"], 60.375)

    def test_detects_runtime_errors_without_inventing_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "stdout.log"
            path.write_text(
                "Traceback (most recent call last):\nRuntimeError: CUDA out of memory\n",
                encoding="utf-8",
            )
            metrics, errors = parse_legacy_log(path)
        self.assertIn("traceback", errors)
        self.assertIn("cuda_out_of_memory", errors)
        self.assertIsNone(metrics["real_running_time_seconds"])

    def test_parses_request_completion_and_gpu_idle_evidence(self) -> None:
        log = """\
finish one req: model id: 0 dp id: 0, output.request_id: 7
finish one req: model id: 0 dp id: 1, output.request_id: 7
finish one req: model id: 1 dp id: 0, output.request_id: 8
Total search time: 1
Total time for preparation before search: 1
total running time: 10s
non_computation_ranges={'0': [[0, [[1.0, 4.0], [3.0, 5.0]]]], '1': [[0, [[2.0], [4.0]]]]}
"""
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "stdout.log"
            path.write_text(log, encoding="utf-8")
            metrics, errors = parse_legacy_log(path)

        self.assertEqual(errors, [])
        self.assertEqual(metrics["completed_request_count"], 2)
        self.assertEqual(metrics["completion_event_count"], 3)
        self.assertIsNone(metrics["failed_request_count"])
        self.assertEqual(metrics["gpu_idle_time_seconds"], 5.0)
        self.assertEqual(metrics["gpu_capacity_time_seconds"], 20.0)
        self.assertEqual(metrics["gpu_idle_ratio"], 0.25)

    def test_collection_is_idempotent_and_records_input_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_directory = Path(temporary)
            task_directory = run_directory / "tasks" / "task-1"
            task_directory.mkdir(parents=True)
            run_metadata = {
                "run_id": "run-1",
                "profile": "smoke",
                "execution_protocol": "camera_ready_reconstruction_v1",
                "git": {"commit": "deadbeef"},
                "artifact_input_digest_sha256": "a" * 64,
            }
            (run_directory / "run.json").write_text(
                json.dumps(run_metadata), encoding="utf-8"
            )
            metadata = {
                "task_id": "task-1",
                "profile": "smoke",
                "workload_id": "router",
                "method_id": "mil",
                "status": "succeeded",
                "comparison_to_control": {
                    "control_task_id": "task-0",
                    "parameter_diff": {"noise_level": {"control": -2, "variant": 0}},
                },
                "parameters": {"ratio_seed": 0},
                "seed_provenance": {"experiment_seed": 0},
                "legacy": {"search_method_baseline": "greedy_saturn_LPT"},
                "stdout_log": "stdout.attempt-1.log",
                "exit_code": 0,
                "task_timeout_seconds": 12.5,
            }
            metadata_path = task_directory / "metadata.json"
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            resolved_path = task_directory / "resolved_config.yaml"
            resolved_path.write_text("schema_version: 1\n", encoding="utf-8")
            log_path = task_directory / "stdout.attempt-1.log"
            log_path.write_text(
                "Total search time: 2\n"
                "Total time for preparation before search: 1\n"
                "Best group seq: [] 3 0\n"
                "total running time: 4s\n",
                encoding="utf-8",
            )

            first = collect_run(run_directory)
            first_bytes = (run_directory / "metrics.jsonl").read_bytes()
            second = collect_run(run_directory)
            second_bytes = (run_directory / "metrics.jsonl").read_bytes()
            summary = read_json(run_directory / "summary.json")
            serialized = read_jsonl(run_directory / "metrics.jsonl")
            log_checksum = sha256_file(log_path)
            resolved_checksum = sha256_file(resolved_path)
            metrics_checksum = sha256_file(run_directory / "metrics.jsonl")

        self.assertEqual(first, second)
        self.assertEqual(first_bytes, second_bytes)
        self.assertEqual(serialized, second)
        self.assertEqual(first[0]["git_commit"], "deadbeef")
        self.assertEqual(
            first[0]["execution_protocol"], "camera_ready_reconstruction_v1"
        )
        self.assertEqual(first[0]["seed_provenance"]["experiment_seed"], 0)
        self.assertEqual(first[0]["task_timeout_seconds"], 12.5)
        self.assertTrue(first[0]["provenance_required"])
        self.assertEqual(first[0]["source_checksum_sha256"], log_checksum)
        self.assertEqual(first[0]["resolved_config_checksum_sha256"], resolved_checksum)
        self.assertEqual(first[0]["comparison_to_control"]["control_task_id"], "task-0")
        self.assertEqual(summary["metrics_checksum_sha256"], metrics_checksum)

    def test_nominal_success_with_traceback_is_classified_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_directory = Path(temporary)
            task_directory = run_directory / "tasks" / "task-1"
            task_directory.mkdir(parents=True)
            (run_directory / "run.json").write_text(
                json.dumps({"run_id": "run-1", "profile": "smoke"}),
                encoding="utf-8",
            )
            (task_directory / "metadata.json").write_text(
                json.dumps(
                    {
                        "task_id": "task-1",
                        "profile": "smoke",
                        "workload_id": "router",
                        "method_id": "mil",
                        "status": "succeeded",
                        "exit_code": 0,
                        "stdout_log": "stdout.attempt-1.log",
                        "legacy": {"search_method_baseline": "greedy_saturn_LPT"},
                    }
                ),
                encoding="utf-8",
            )
            (task_directory / "resolved_config.yaml").write_text(
                "schema_version: 1\n", encoding="utf-8"
            )
            (task_directory / "stdout.attempt-1.log").write_text(
                "Traceback (most recent call last):\nRuntimeError: failed\n",
                encoding="utf-8",
            )
            records = collect_run(run_directory)
            summary = read_json(run_directory / "summary.json")

        self.assertEqual(records[0]["status"], "invalid")
        self.assertIn("traceback", records[0]["errors"])
        self.assertEqual(summary["invalid"], 1)


if __name__ == "__main__":
    unittest.main()
