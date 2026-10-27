from __future__ import annotations

import os
import io
import json
import subprocess
from contextlib import redirect_stdout
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from artifact.mil_artifact.config import ArtifactConfigError, expand_experiment
from artifact.mil_artifact.run import (
    _acquire_gpu_locks,
    _execute_task_matrix,
    _new_run_directory,
    _run_scoped_process,
    _snapshot_artifact_inputs,
    _successful_task,
    _validate_resume_metadata,
    artifact_input_digest,
    execute,
    git_metadata,
)
from artifact.mil_artifact.schema import read_json


class RunProvenanceTest(unittest.TestCase):
    def setUp(self):
        # These fixtures exercise Git worktrees; the package tests cover the
        # independent source-snapshot route, including its failure behavior.
        context = mock.patch('artifact.mil_artifact.run.SNAPSHOT_RECORD',
                             'artifact/.git-unit-test-no-snapshot.json')
        context.start()
        self.addCleanup(context.stop)

    def test_runner_preflight_preserves_all_filters(self):
        from artifact.mil_artifact import doctor, run
        observed = []
        def environment_checks(resolved, *_args):
            observed.extend(resolved.tasks)
            return []
        def call_doctor(command, **kwargs):
            output = io.StringIO()
            with redirect_stdout(output):
                code = doctor.main(command[3:])
            return subprocess.CompletedProcess(command, code, output.getvalue())
        record = {'revision': 'a' * 40, 'snapshot_path': '/fixture', 'config_sha256': 'b' * 64}
        with mock.patch.object(run.subprocess, 'run', side_effect=call_doctor), \
                mock.patch.object(doctor, '_environment_checks', side_effect=environment_checks), \
                mock.patch.object(doctor, '_disk_checks', return_value=[]), \
                mock.patch.object(doctor, '_cost_checks', return_value=[]), \
                mock.patch.object(doctor, '_dataset_checks', return_value=[]), \
                mock.patch.object(doctor, '_figure_source_checks', return_value=[]), \
                mock.patch.object(doctor, '_model_checks', return_value=[doctor.Check('model:org/model', 'ok', json.dumps(record))]):
            report = run._run_preflight(sys.executable, 'main', False, ['router'], ['mil'], ['reqnum=10000'])
        self.assertTrue(observed)
        self.assertTrue(all(t.method_id == 'mil' and t.workload_id == 'router'
                            and t.parameters['reqnum'] == 10000 for t in observed))
        self.assertFalse(any(t.legacy['search_method_baseline'].startswith('saturnLP') for t in observed))
        self.assertEqual(report['model_snapshots'], {'org/model': record})

    def test_main_forwards_selection_without_starting_gpu(self):
        from artifact.mil_artifact import run
        with mock.patch.object(run, 'git_metadata', return_value={'dirty': False}), \
                mock.patch.object(run, '_run_preflight', return_value={}) as preflight, \
                mock.patch.object(run, 'execute', return_value=0):
            code = run.main(['--profile', 'main', '--suite', 'router', '--method', 'mil',
                             '--parameter', 'reqnum=10000', '--execute', '--allow-provisional'])
        self.assertEqual(code, 0)
        self.assertEqual(preflight.call_args.args[3:], (['router'], ['mil'], ['reqnum=10000']))

    def test_git_metadata_combines_tracked_and_untracked_changes(self) -> None:
        def git_output(arguments: list[str]) -> str:
            if arguments[0] == "status":
                return " M tracked.txt"
            if arguments[0] == "ls-files":
                return "new.txt"
            if arguments[0] == "rev-parse":
                return "deadbeef"
            if arguments[0] == "branch":
                return "artifact"
            return "deadbeef-dirty"

        with mock.patch("artifact.mil_artifact.run._git_output", side_effect=git_output):
            metadata = git_metadata()
        self.assertTrue(metadata["dirty"])
        self.assertEqual(
            metadata["status_porcelain"], [" M tracked.txt", "?? new.txt"]
        )

    @mock.patch("artifact.mil_artifact.run._git_output", return_value=None)
    def test_git_metadata_fails_closed_when_inspection_fails(
        self, _git: mock.Mock
    ) -> None:
        metadata = git_metadata()
        self.assertTrue(metadata["dirty"])
        self.assertIn("unable to inspect", metadata["status_porcelain"][0])

    def test_input_digest_is_stable_and_complete(self) -> None:
        resolved = expand_experiment("smoke")
        first = artifact_input_digest(resolved)
        second = artifact_input_digest(resolved)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 64)

    @mock.patch("artifact.mil_artifact.run._git_output", return_value="deadbeef")
    def test_run_directory_contains_commit_and_input_digest(self, _git: mock.Mock) -> None:
        resolved = expand_experiment("smoke")
        with tempfile.TemporaryDirectory() as temporary:
            directory = _new_run_directory(Path(temporary), resolved)
        self.assertRegex(directory.name, r"^\d{8}T\d{6}Z-smoke-deadbeef-[0-9a-f]{8}$")

    def test_snapshot_checksums_every_declared_input(self) -> None:
        resolved = expand_experiment("smoke")
        with tempfile.TemporaryDirectory() as temporary:
            entries = _snapshot_artifact_inputs(Path(temporary), resolved)
        paths = {entry["path"] for entry in entries}
        self.assertIn("artifact/configs/experiments/smoke.yaml", paths)
        self.assertIn("artifact/configs/figures.yaml", paths)
        self.assertIn("artifact/manifests/models.yaml", paths)
        self.assertTrue(all(len(entry["checksum_sha256"]) == 64 for entry in entries))

    @mock.patch("artifact.mil_artifact.run._git_output", return_value="commit-1")
    def test_resume_rejects_changed_inputs(self, _git: mock.Mock) -> None:
        resolved = expand_experiment("smoke")
        metadata = {
            "profile": "smoke",
            "artifact_input_digest_sha256": "wrong",
            "git": {"commit": "commit-1"},
            "tasks": [task.task_id for task in resolved.tasks],
            "host": {"task_python": "python3"},
        }
        with self.assertRaisesRegex(ArtifactConfigError, "inputs differ"):
            _validate_resume_metadata(metadata, resolved, "python3")

    @mock.patch("artifact.mil_artifact.run._git_output", return_value="commit-1")
    def test_resume_accepts_identical_provenance(self, _git: mock.Mock) -> None:
        resolved = expand_experiment("smoke")
        metadata = {
            "profile": "smoke",
            "artifact_input_digest_sha256": artifact_input_digest(resolved),
            "git": {"commit": "commit-1"},
            "tasks": [task.task_id for task in resolved.tasks],
            "host": {"task_python": "python3"},
        }
        _validate_resume_metadata(metadata, resolved, "python3")

    @mock.patch("artifact.mil_artifact.run._git_output", return_value="commit-1")
    def test_resume_rejects_changed_profile_mismatch_policy(self, _git: mock.Mock) -> None:
        resolved = expand_experiment("smoke")
        metadata = {
            "profile": "smoke",
            "artifact_input_digest_sha256": artifact_input_digest(resolved),
            "git": {"commit": "commit-1"},
            "tasks": [task.task_id for task in resolved.tasks],
            "host": {"task_python": "python3"},
            "profile_mismatch_allowed": True,
        }
        with self.assertRaisesRegex(ArtifactConfigError, "mismatch policy"):
            _validate_resume_metadata(metadata, resolved, "python3", False)

    @mock.patch("artifact.mil_artifact.run._git_output", return_value="commit-1")
    def test_resume_rejects_changed_task_timeout(self, _git: mock.Mock) -> None:
        resolved = expand_experiment("smoke")
        metadata = {
            "profile": "smoke",
            "artifact_input_digest_sha256": artifact_input_digest(resolved),
            "git": {"commit": "commit-1"},
            "tasks": [task.task_id for task in resolved.tasks],
            "host": {"task_python": "python3"},
            "profile_mismatch_allowed": False,
            "task_timeout_seconds": 30.0,
        }
        with self.assertRaisesRegex(ArtifactConfigError, "task-timeout policy"):
            _validate_resume_metadata(
                metadata,
                resolved,
                "python3",
                False,
                task_timeout_seconds=60.0,
            )

    def test_gpu_lock_rejects_a_second_holder(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with _acquire_gpu_locks(root, [0], "first", 0):
                with self.assertRaisesRegex(ArtifactConfigError, "locked"):
                    with _acquire_gpu_locks(root, [0], "second", 0):
                        self.fail("second holder unexpectedly acquired the lock")

    def test_scoped_process_records_output_and_exit_code(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            log_path = Path(temporary) / "stdout.log"
            with log_path.open("w", encoding="utf-8") as log:
                return_code, received_signal, timed_out = _run_scoped_process(
                    [sys.executable, "-c", "print('scoped')"],
                    Path(temporary),
                    dict(os.environ),
                    log,
                )
            output = log_path.read_text(encoding="utf-8")

        self.assertEqual(return_code, 0)
        self.assertIsNone(received_signal)
        self.assertFalse(timed_out)
        self.assertEqual(output.strip(), "scoped")

    def test_scoped_process_timeout_terminates_only_its_process_group(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            log_path = Path(temporary) / "stdout.log"
            with log_path.open("w", encoding="utf-8") as log:
                return_code, received_signal, timed_out = _run_scoped_process(
                    [sys.executable, "-c", "import time; time.sleep(60)"],
                    Path(temporary),
                    dict(os.environ),
                    log,
                    timeout_seconds=0.05,
                )

        self.assertEqual(return_code, 124)
        self.assertIsNone(received_signal)
        self.assertTrue(timed_out)

    @mock.patch("artifact.mil_artifact.run.git_metadata", return_value={})
    @mock.patch(
        "artifact.mil_artifact.run._run_scoped_process",
        return_value=(124, None, True),
    )
    @mock.patch("artifact.mil_artifact.run._stage_task_workdir")
    def test_task_timeout_is_recorded_in_immutable_attempt(
        self,
        stage_workdir: mock.Mock,
        _run_process: mock.Mock,
        _git_metadata: mock.Mock,
    ) -> None:
        resolved = expand_experiment("smoke", workload_filter=["router"])
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            workdir = run / "work"
            workdir.mkdir()
            stage_workdir.return_value = workdir
            failed, skipped, interrupted = _execute_task_matrix(
                run,
                resolved,
                run / "shared",
                "python3",
                12.5,
            )
            metadata = read_json(
                run / "tasks" / resolved.tasks[0].task_id / "metadata.json"
            )

        self.assertEqual(failed, [resolved.tasks[0].task_id])
        self.assertEqual(skipped, 0)
        self.assertIsNone(interrupted)
        self.assertEqual(metadata["exit_code"], 124)
        self.assertTrue(metadata["timed_out"])
        self.assertEqual(metadata["task_timeout_seconds"], 12.5)
        self.assertTrue(metadata["attempts"][0]["timed_out"])

    def test_task_uses_clean_source_environment_and_records_model_mapping(self):
        from artifact.mil_artifact import run
        from artifact.mil_artifact.config import repo_root
        resolved = expand_experiment('smoke', workload_filter=['router'])
        records = {'org/model': {'revision': 'a' * 40, 'snapshot_path': '/fixture'}}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with mock.patch.object(run, '_stage_task_workdir', return_value=root), \
                    mock.patch.object(run, 'git_metadata', return_value={}), \
                    mock.patch.object(run, '_run_scoped_process', return_value=(0, None, False)) as process, \
                    mock.patch.dict(os.environ, {'PYTHONPATH': '/foreign', 'MIL_CPU_SCHEDULING': '1'}):
                run._execute_task_matrix(root, resolved, root / 'shared', sys.executable, None, records)
            metadata = read_json(root / 'tasks' / resolved.tasks[0].task_id / 'metadata.json')
        environment = process.call_args.args[2]
        self.assertEqual(environment['PYTHONPATH'], str(repo_root()))
        self.assertNotIn('MIL_CPU_SCHEDULING', environment)
        self.assertEqual(environment['PYTHONNOUSERSITE'], '1')
        self.assertEqual(environment['CUDA_VISIBLE_DEVICES'], resolved.tasks[0].environment()['CUDA_VISIBLE_DEVICES'])
        self.assertEqual(json.loads(environment['MIL_MODEL_SNAPSHOTS']), records)
        self.assertEqual(metadata['model_snapshots'], records)

    def test_resume_rejects_nominal_success_with_invalid_log(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            task = Path(temporary)
            (task / "resolved_config.yaml").write_text(
                "schema_version: 1\n", encoding="utf-8"
            )
            (task / "metadata.json").write_text(
                '{"status":"succeeded","exit_code":0,'
                '"stdout_log":"stdout.attempt-1.log",'
                '"legacy":{"search_method_baseline":"greedy_saturn_LPT"}}',
                encoding="utf-8",
            )
            (task / "stdout.attempt-1.log").write_text(
                "Traceback (most recent call last):\nRuntimeError: failed\n",
                encoding="utf-8",
            )
            self.assertFalse(_successful_task(task))

            (task / "stdout.attempt-1.log").write_text(
                "Total search time: 1\n"
                "Total time for preparation before search: 2\n"
                "total running time: 3s\n",
                encoding="utf-8",
            )
            self.assertTrue(_successful_task(task))

    @mock.patch("artifact.mil_artifact.run._execute_task_matrix", return_value=([], 0, None))
    @mock.patch("artifact.mil_artifact.run._stage_shared_resources")
    @mock.patch("artifact.mil_artifact.run._host_metadata", return_value={})
    @mock.patch("artifact.mil_artifact.run._snapshot_artifact_inputs", return_value=[])
    @mock.patch("artifact.mil_artifact.run.artifact_input_digest", return_value="a" * 64)
    @mock.patch("artifact.mil_artifact.run._git_output", return_value="deadbeef")
    def test_execute_automatically_collects_and_rejects_empty_results(
        self,
        _git: mock.Mock,
        _digest: mock.Mock,
        _snapshot: mock.Mock,
        _host: mock.Mock,
        stage: mock.Mock,
        _matrix: mock.Mock,
    ) -> None:
        resolved = expand_experiment("smoke", workload_filter=["router"])
        with tempfile.TemporaryDirectory() as temporary:
            results_root = Path(temporary)
            stage.return_value = results_root / "shared"
            return_code = execute(
                resolved,
                results_root,
                None,
                "python3",
                {"summary": {"error": 0}},
            )
            run_directories = [
                path
                for path in results_root.iterdir()
                if path.is_dir() and path.name != ".gpu-locks"
            ]
            self.assertEqual(len(run_directories), 1)
            run = read_json(run_directories[0] / "run.json")
            summary = read_json(run_directories[0] / "summary.json")
            verification = read_json(run_directories[0] / "verification.json")

        self.assertEqual(return_code, 1)
        self.assertEqual(run["status"], "failed")
        self.assertEqual(run["postprocessing"]["status"], "failed")
        self.assertEqual(run["postprocessing"]["collector_record_count"], 0)
        self.assertEqual(summary["record_count"], 0)
        self.assertEqual(verification["status"], "failed")
        self.assertTrue(any('missing planned task' in issue['message']
                            for issue in verification['issues']))

    @mock.patch(
        "artifact.mil_artifact.run.collect_run",
        side_effect=ValueError("parser failed"),
    )
    @mock.patch("artifact.mil_artifact.run._execute_task_matrix", return_value=([], 0, None))
    @mock.patch("artifact.mil_artifact.run._stage_shared_resources")
    @mock.patch("artifact.mil_artifact.run._host_metadata", return_value={})
    @mock.patch("artifact.mil_artifact.run._snapshot_artifact_inputs", return_value=[])
    @mock.patch("artifact.mil_artifact.run.artifact_input_digest", return_value="a" * 64)
    @mock.patch("artifact.mil_artifact.run._git_output", return_value="deadbeef")
    def test_execute_records_postprocessing_failure(
        self,
        _git: mock.Mock,
        _digest: mock.Mock,
        _snapshot: mock.Mock,
        _host: mock.Mock,
        stage: mock.Mock,
        _matrix: mock.Mock,
        _collect: mock.Mock,
    ) -> None:
        resolved = expand_experiment("smoke", workload_filter=["router"])
        with tempfile.TemporaryDirectory() as temporary:
            results_root = Path(temporary)
            stage.return_value = results_root / "shared"
            with self.assertRaisesRegex(RuntimeError, "postprocessing failed"):
                execute(
                    resolved,
                    results_root,
                    None,
                    "python3",
                    {"summary": {"error": 0}},
                )
            run_directory = next(
                path
                for path in results_root.iterdir()
                if path.is_dir() and path.name != ".gpu-locks"
            )
            run = read_json(run_directory / "run.json")

        self.assertEqual(run["status"], "failed")
        self.assertEqual(run["postprocessing"]["status"], "failed")
        self.assertEqual(run["postprocessing"]["error_type"], "ValueError")

    @mock.patch(
        "artifact.mil_artifact.run.verify_run",
        return_value={"status": "failed", "issue_count": 1, "warning_count": 0},
    )
    @mock.patch(
        "artifact.mil_artifact.run.collect_run",
        return_value=[{"task_id": "bad-task", "status": "invalid"}],
    )
    @mock.patch("artifact.mil_artifact.run._execute_task_matrix", return_value=([], 0, None))
    @mock.patch("artifact.mil_artifact.run._stage_shared_resources")
    @mock.patch("artifact.mil_artifact.run._host_metadata", return_value={})
    @mock.patch("artifact.mil_artifact.run._snapshot_artifact_inputs", return_value=[])
    @mock.patch("artifact.mil_artifact.run.artifact_input_digest", return_value="a" * 64)
    @mock.patch("artifact.mil_artifact.run._git_output", return_value="deadbeef")
    def test_verification_issue_marks_run_failed(
        self,
        _git: mock.Mock,
        _digest: mock.Mock,
        _snapshot: mock.Mock,
        _host: mock.Mock,
        stage: mock.Mock,
        _matrix: mock.Mock,
        _collect: mock.Mock,
        _verify: mock.Mock,
    ) -> None:
        resolved = expand_experiment("smoke", workload_filter=["router"])
        with tempfile.TemporaryDirectory() as temporary:
            results_root = Path(temporary)
            stage.return_value = results_root / "shared"
            return_code = execute(
                resolved,
                results_root,
                None,
                "python3",
                {"summary": {"error": 0}},
            )
            run_directory = next(
                path
                for path in results_root.iterdir()
                if path.is_dir() and path.name != ".gpu-locks"
            )
            run = read_json(run_directory / "run.json")

        self.assertEqual(return_code, 1)
        self.assertEqual(run["status"], "failed")
        self.assertEqual(run["invalid_task_ids"], ["bad-task"])
        self.assertEqual(run["postprocessing"]["status"], "failed")


if __name__ == "__main__":
    unittest.main()
