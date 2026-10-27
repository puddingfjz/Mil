from __future__ import annotations

import tempfile
import json
import unittest
from pathlib import Path

import yaml

from artifact.mil_artifact.hashing import sha256_file
from artifact.mil_artifact.schema import SCHEMA_VERSION, atomic_write_json, write_jsonl
from artifact.mil_artifact.collect import collect_run
from artifact.mil_artifact.config import artifact_root
from artifact.mil_artifact.verify import _provenance_issues, verify_run


def _record(runtime: float = 10.0) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": "run-1",
        "task_id": "smoke.router.mil.test",
        "profile": "smoke",
        "workload_id": "router",
        "method_id": "mil",
        "status": "succeeded",
        "parameters": {"reqnum": 5},
        "metrics": {
            "predicted_running_time_seconds": 9.0,
            "real_running_time_seconds": runtime,
            "search_time_seconds": 1.0,
            "preparation_time_seconds": 0.5,
            "total_time_seconds": runtime + 1.0,
        },
        "errors": [],
    }


class VerificationTest(unittest.TestCase):
    def test_collected_run_must_cover_all_planned_tasks(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            metadata = {'run_id':'incomplete-smoke', 'profile':'smoke', 'status':'interrupted',
                        'tasks':['one', 'two'], 'git':{'commit':'a'*40},
                        'artifact_input_digest_sha256':'b'*64}
            atomic_write_json(run/'run.json', metadata)
            def verify():
                collect_run(run)
                return verify_run(run, artifact_root()/'manifests/expected_results.yaml')
            for completed in range(3):
                if completed:
                    name = metadata['tasks'][completed - 1]
                    task = run/'tasks'/name
                    atomic_write_json(task/'metadata.json', {'task_id':name, 'status':'succeeded',
                        'exit_code':0, 'profile':'smoke', 'workload_id':'router', 'method_id':'mil',
                        'parameters':{}, 'stdout_log':'stdout.attempt-1.log'})
                    (task/'resolved_config.yaml').write_text('schema_version: 1\n')
                    (task/'stdout.attempt-1.log').write_text('Total search time: 2\n'
                        'Total time for preparation before search: 1\n'
                        'Best group seq: [] 3 0\ntotal running time: 4s\n')
                with self.subTest(completed=completed):
                    self.assertGreater(verify()['issue_count'], 0)
                    metadata['status'] = 'succeeded'
                    atomic_write_json(run/'run.json', metadata)
                    report = verify()
                    self.assertEqual(report['issue_count'] == 0, completed == 2)
                    if completed < 2:
                        self.assertTrue(any('missing planned task' in row['message'] for row in report['issues']))
                    metadata['status'] = 'interrupted'
                    atomic_write_json(run/'run.json', metadata)
            metadata['status'] = 'succeeded'
            atomic_write_json(run/'run.json', metadata)
            self.assertEqual(verify()['issue_count'], 0)

    def test_empty_duplicate_and_unexpected_records_are_not_accepted(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            expected = run/'expected.yaml'
            expected.write_text('schema_version: 1\nresults: []\n')
            for records in ([], [_record(), _record()]):
                write_jsonl(run/'metrics.jsonl', records)
                self.assertGreater(verify_run(run, expected)['issue_count'], 0)
            atomic_write_json(run/'run.json', {'status':'succeeded', 'tasks':['different-task']})
            write_jsonl(run/'metrics.jsonl', [_record()])
            self.assertGreater(verify_run(run, expected)['issue_count'], 0)

    def test_historical_reference_without_task_manifest_stays_supported(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            expected = run/'expected.yaml'
            expected.write_text('schema_version: 1\nresults: []\n')
            write_jsonl(run/'metrics.jsonl', [_record()])
            atomic_write_json(run/'run.json', {'status':'published_historical_reference', 'record_count':1})
            self.assertEqual(verify_run(run, expected)['issue_count'], 0)

    def test_snapshot_identity_cannot_be_an_unchecked_replacement_for_git(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            record = {'provenance_required': True, 'git_commit': None}
            self.assertIn('missing provenance field git_commit or source_snapshot_sha256',
                          _provenance_issues(run, record))
            record['source_snapshot_sha256'] = 'not-a-hash'
            self.assertIn('invalid source_snapshot_sha256', _provenance_issues(run, record))
            record['source_snapshot_sha256'] = 'a' * 64
            self.assertIn('missing source snapshot evidence', _provenance_issues(run, record))
            path = run / 'inputs/artifact/source_snapshot.json'
            path.parent.mkdir(parents=True)
            path.write_text('{}')
            record['source_snapshot_sha256'] = sha256_file(path)
            self.assertIn('invalid source snapshot evidence', _provenance_issues(run, record))
            path.write_text(json.dumps({'schema_version': 1, 'kind': 'source_snapshot',
                                        'files_sha256': {'final.py': 'a'*64}}))
            record['source_snapshot_sha256'] = sha256_file(path)
            self.assertFalse([i for i in _provenance_issues(run, record) if 'snapshot' in i])
            # Other required provenance is not waived by a valid snapshot.
            self.assertIn('missing provenance field task_metadata_checksum_sha256',
                          _provenance_issues(run, record))

    def test_compares_expected_metric(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / "run"
            run.mkdir()
            write_jsonl(run / "metrics.jsonl", [_record()])
            expected = Path(temporary) / "expected.yaml"
            expected.write_text(
                yaml.safe_dump(
                    {
                        "schema_version": 1,
                        "results": [
                            {
                                "selector": {"task_id": "smoke.router.mil.test"},
                                "metrics": {"real_running_time_seconds": 10.1},
                                "tolerances": {
                                    "real_running_time_seconds": {"relative": 0.02, "absolute": 0}
                                },
                            }
                        ],
                        "default_tolerances": {
                            "deterministic_absolute": 0,
                            "runtime_relative": 0.02,
                        },
                    }
                ),
                encoding="utf-8",
            )
            report = verify_run(run, expected)
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["issue_count"], 0)

    def test_empty_expectations_are_a_warning_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / "run"
            run.mkdir()
            write_jsonl(run / "metrics.jsonl", [_record()])
            expected = Path(temporary) / "expected.yaml"
            expected.write_text(
                yaml.safe_dump({"schema_version": 1, "results": [], "default_tolerances": {}}),
                encoding="utf-8",
            )
            report = verify_run(run, expected)
        self.assertEqual(report["status"], "passed_with_warnings")

    def test_unrelated_profile_expectation_is_a_warning_by_default(self) -> None:
        expected_config = {
            "schema_version": 1,
            "results": [
                {
                    "selector": {"profile": "reference", "method_id": "mil"},
                    "metrics": {"total_time_seconds": 15.0},
                }
            ],
            "default_tolerances": {},
        }
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / "run"
            run.mkdir()
            write_jsonl(run / "metrics.jsonl", [_record()])
            expected = Path(temporary) / "expected.yaml"
            expected.write_text(yaml.safe_dump(expected_config), encoding="utf-8")
            report = verify_run(run, expected)

        self.assertEqual(report["status"], "passed_with_warnings")
        self.assertEqual(report["issue_count"], 0)
        self.assertEqual(report["warning_count"], 1)

    def test_median_aggregation_enforces_minimum_samples(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / "run"
            run.mkdir()
            records = []
            for index, runtime in enumerate((9.0, 10.0, 11.0, 12.0, 13.0), 1):
                record = _record(runtime)
                record["task_id"] = f"run-{index}"
                records.append(record)
            write_jsonl(run / "metrics.jsonl", records)
            expected = Path(temporary) / "expected.yaml"
            expected.write_text(
                yaml.safe_dump(
                    {
                        "schema_version": 1,
                        "results": [
                            {
                                "reference_id": "median-test",
                                "selector": {"workload_id": "router", "method_id": "mil"},
                                "aggregation": "median",
                                "minimum_samples": 5,
                                "metrics": {"real_running_time_seconds": 11.0},
                                "tolerances": {
                                    "real_running_time_seconds": {"relative": 0, "absolute": 0}
                                },
                            }
                        ],
                        "default_tolerances": {},
                    }
                ),
                encoding="utf-8",
            )
            report = verify_run(run, expected)

        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["comparisons"][0]["sample_count"], 5)
        self.assertEqual(report["comparisons"][0]["actual"], 11.0)

    def test_rejects_inconsistent_total_and_nonzero_success_exit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / "run"
            run.mkdir()
            record = _record()
            record["metrics"]["total_time_seconds"] = 999.0
            record["exit_code"] = 3
            write_jsonl(run / "metrics.jsonl", [record])
            expected = Path(temporary) / "expected.yaml"
            expected.write_text(
                yaml.safe_dump(
                    {
                        "schema_version": 1,
                        "results": [],
                        "default_tolerances": {},
                    }
                ),
                encoding="utf-8",
            )
            report = verify_run(run, expected)

        messages = [issue["message"] for issue in report["issues"]]
        self.assertTrue(any("does not equal" in message for message in messages))
        self.assertTrue(any("nonzero exit_code" in message for message in messages))

    def test_comparison_reports_error_and_allowed_tolerance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / "run"
            run.mkdir()
            write_jsonl(run / "metrics.jsonl", [_record()])
            expected = Path(temporary) / "expected.yaml"
            expected.write_text(
                yaml.safe_dump(
                    {
                        "schema_version": 1,
                        "results": [
                            {
                                "selector": {"task_id": "smoke.router.mil.test"},
                                "metrics": {"real_running_time_seconds": 10.1},
                                "tolerances": {
                                    "real_running_time_seconds": {
                                        "relative": 0.02,
                                        "absolute": 0,
                                    }
                                },
                            }
                        ],
                        "default_tolerances": {},
                    }
                ),
                encoding="utf-8",
            )
            report = verify_run(run, expected)

        comparison = report["comparisons"][0]
        self.assertAlmostEqual(comparison["absolute_error"], 0.1)
        self.assertAlmostEqual(comparison["allowed_absolute_error"], 0.202)

    def test_provenance_check_detects_modified_source_log(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / "run"
            task_directory = run / "tasks" / "task-1"
            task_directory.mkdir(parents=True)
            metadata = task_directory / "metadata.json"
            metadata.write_text("{}\n", encoding="utf-8")
            resolved = task_directory / "resolved_config.yaml"
            resolved.write_text("schema_version: 1\n", encoding="utf-8")
            source = task_directory / "stdout.attempt-1.log"
            source.write_text("original\n", encoding="utf-8")
            record = _record()
            record.update(
                {
                    "task_id": "task-1",
                    "experiment_id": "task-1",
                    "provenance_required": True,
                    "git_commit": "deadbeef",
                    "artifact_input_digest_sha256": "a" * 64,
                    "task_metadata_checksum_sha256": sha256_file(metadata),
                    "resolved_config_checksum_sha256": sha256_file(resolved),
                    "source_log": "tasks/task-1/stdout.attempt-1.log",
                    "source_checksum_sha256": sha256_file(source),
                    "source_size_bytes": source.stat().st_size,
                }
            )
            write_jsonl(run / "metrics.jsonl", [record])
            source.write_text("tampered\n", encoding="utf-8")
            expected = Path(temporary) / "expected.yaml"
            expected.write_text(
                yaml.safe_dump(
                    {"schema_version": 1, "results": [], "default_tolerances": {}}
                ),
                encoding="utf-8",
            )
            report = verify_run(run, expected)

        messages = [issue["message"] for issue in report["issues"]]
        self.assertIn(
            "source_log checksum mismatch: tasks/task-1/stdout.attempt-1.log",
            messages,
        )


if __name__ == "__main__":
    unittest.main()
