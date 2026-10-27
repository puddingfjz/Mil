"""Expand and run reproducible experiment profiles.

Dry-run is the default. Explicit execution requires source, runtime, resource,
hardware and selected-solver preflight; it does not claim measured performance.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
import platform
import signal
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Sequence, TextIO

from .collect import collect_run, parse_legacy_log, success_diagnostics
from .config import (
    ArtifactConfigError,
    ResolvedExperiment,
    TaskSpec,
    artifact_root,
    dump_yaml,
    expand_experiment,
    parse_parameter_filters,
    repo_root,
)
from .hashing import sha256_file
from .plot import plot_run
from .schema import SCHEMA_VERSION, atomic_write_json, read_json
from .verify import verify_run
from .package import SNAPSHOT_RECORD, verify_snapshot
from .runtime_identity import child_environment


def _git_output(arguments: Sequence[str]) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "--no-optional-locks", "-C", os.fspath(repo_root()), *arguments],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return completed.stdout.strip()


def git_metadata() -> dict[str, Any]:
    if (repo_root() / SNAPSHOT_RECORD).exists():
        try:
            snapshot = verify_snapshot(repo_root())
        except (OSError, ValueError, KeyError, TypeError) as error:
            return {'commit': None, 'branch': None, 'describe': None, 'dirty': True,
                    'status_porcelain': ['!! invalid source snapshot: ' + str(error)]}
        return {'commit': None, 'branch': None, 'describe': 'source-' + snapshot['sha256'][:12],
                'dirty': False, 'status_porcelain': [],
                'source_snapshot_sha256': snapshot['sha256']}
    tracked_status = _git_output(
        ["status", "--porcelain=v1", "--untracked-files=no"]
    )
    untracked = _git_output(["ls-files", "--others", "--exclude-standard"])
    status_lines = tracked_status.splitlines() if tracked_status else []
    if untracked:
        status_lines.extend(f"?? {path}" for path in untracked.splitlines() if path)
    inspection_failed = tracked_status is None or untracked is None
    if inspection_failed:
        status_lines.append("!! unable to inspect complete Git worktree state")
    return {
        "commit": _git_output(["rev-parse", "HEAD"]),
        "branch": _git_output(["branch", "--show-current"]),
        "describe": _git_output(["describe", "--always", "--dirty", "--tags"]),
        "dirty": bool(status_lines),
        "status_porcelain": status_lines,
    }


def _artifact_input_sources(resolved: ResolvedExperiment) -> list[Path]:
    artifact = repo_root() / "artifact"
    sources = [
        resolved.profile_path,
        resolved.hardware_path,
        resolved.methods_path,
        artifact / "configs" / "figures.yaml",
        artifact / "env" / "runtime.yaml",
        artifact / "env" / "runtime-freeze.txt",
        *sorted((artifact / "manifests").glob("*.yaml")),
    ]
    snapshot = repo_root() / SNAPSHOT_RECORD
    if snapshot.is_file():
        sources.append(snapshot)
    return sources


def artifact_input_digest(resolved: ResolvedExperiment) -> str:
    """Hash every versioned input that controls execution or interpretation."""

    root = repo_root().resolve()
    digest = hashlib.sha256()
    for source in _artifact_input_sources(resolved):
        resolved_source = source.resolve()
        try:
            name = resolved_source.relative_to(root).as_posix()
        except ValueError:
            name = resolved_source.as_posix()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(resolved_source).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _run_id(resolved: ResolvedExperiment) -> str:
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    commit = (_git_output(["rev-parse", "--short=8", "HEAD"]) or "nogit").strip()
    inputs = artifact_input_digest(resolved)[:8]
    return f"{timestamp}-{resolved.profile['name']}-{commit}-{inputs}"


def _default_results_root() -> Path:
    configured = os.environ.get("MIL_RESULTS_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    return repo_root() / "artifact" / "results"


@contextmanager
def _acquire_gpu_locks(
    lock_root: Path,
    devices: Sequence[int],
    run_id: str,
    timeout_seconds: float,
) -> Iterator[None]:
    """Acquire advisory, run-scoped locks without inspecting or killing peers."""

    if timeout_seconds < 0:
        raise ArtifactConfigError("GPU lock timeout must be non-negative")
    lock_root.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout_seconds
    handles: list[TextIO] = []
    try:
        for device in sorted(set(devices)):
            path = lock_root / f"gpu-{device}.lock"
            handle = path.open("a+", encoding="utf-8")
            while True:
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        handle.close()
                        raise ArtifactConfigError(
                            f"GPU {device} is locked by another Artifact run; "
                            "increase --gpu-lock-timeout to wait"
                        )
                    time.sleep(min(0.25, max(0.0, deadline - time.monotonic())))
            handle.seek(0)
            handle.truncate()
            json.dump(
                {
                    "run_id": run_id,
                    "pid": os.getpid(),
                    "device": device,
                    "acquired_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                },
                handle,
                sort_keys=True,
            )
            handle.write("\n")
            handle.flush()
            handles.append(handle)
        yield
    finally:
        for handle in reversed(handles):
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            finally:
                handle.close()


class _RunInterrupted(Exception):
    def __init__(self, signal_number: int) -> None:
        super().__init__(f"received signal {signal_number}")
        self.signal_number = signal_number


def _terminate_process_group(process: subprocess.Popen[Any], signal_number: int) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal_number)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def _run_scoped_process(
    command: Sequence[str],
    cwd: Path,
    environment: dict[str, str],
    log: TextIO,
    timeout_seconds: float | None = None,
) -> tuple[int, int | None, bool]:
    """Run one task in its own process group and clean up only that group."""

    if timeout_seconds is not None and (
        not math.isfinite(float(timeout_seconds)) or timeout_seconds <= 0
    ):
        raise ArtifactConfigError("task timeout must be a positive number")

    process = subprocess.Popen(
        list(command),
        cwd=cwd,
        env=environment,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    original_handlers: dict[int, Any] = {}

    def handle_signal(signal_number: int, _frame: Any) -> None:
        raise _RunInterrupted(signal_number)

    try:
        for signal_number in (signal.SIGINT, signal.SIGTERM):
            original_handlers[signal_number] = signal.getsignal(signal_number)
            signal.signal(signal_number, handle_signal)
        try:
            return process.wait(timeout=timeout_seconds), None, False
        except subprocess.TimeoutExpired:
            _terminate_process_group(process, signal.SIGTERM)
            return 124, None, True
        except KeyboardInterrupt:
            _terminate_process_group(process, signal.SIGINT)
            return 128 + signal.SIGINT, signal.SIGINT, False
        except _RunInterrupted as exc:
            _terminate_process_group(process, exc.signal_number)
            return 128 + exc.signal_number, exc.signal_number, False
        except BaseException:
            _terminate_process_group(process, signal.SIGTERM)
            raise
    finally:
        for signal_number, handler in original_handlers.items():
            signal.signal(signal_number, handler)


def _new_run_directory(root: Path, resolved: ResolvedExperiment) -> Path:
    base = root / _run_id(resolved)
    candidate = base
    suffix = 1
    while candidate.exists():
        candidate = Path(f"{base}-{suffix}")
        suffix += 1
    candidate.mkdir(parents=True)
    return candidate


def _cpu_model() -> str | None:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except (OSError, IndexError):
        pass
    return platform.processor() or None


def _physical_memory_bytes() -> int | None:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (OSError, ValueError):
        return None


def _task_runtime_metadata(task_python: str) -> dict[str, Any]:
    script = (
        "import json,torch;"
        "f=getattr(torch.cuda.nccl,'version',None);"
        "print(json.dumps({'torch':torch.__version__,'cuda':torch.version.cuda,"
        "'nccl':f() if f else None}))"
    )
    try:
        completed = subprocess.run(
            [task_python, "-c", script],
            env=child_environment(repo_root()),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=20,
        )
        if completed.returncode == 0:
            return json.loads(completed.stdout)
        return {"status": "error", "detail": completed.stdout.strip()}
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
        return {"status": "error", "detail": str(exc)}


def _host_metadata(
    task_python: str,
    resolved: ResolvedExperiment,
    profile_mismatch_allowed: bool,
) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,uuid,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=20,
        )
        gpu_inventory = completed.stdout.strip().splitlines()
        gpu_inventory_status = completed.returncode
    except (OSError, subprocess.TimeoutExpired) as exc:
        gpu_inventory = [str(exc)]
        gpu_inventory_status = 127
    try:
        topology_process = subprocess.run(
            ["nvidia-smi", "topo", "-m"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=20,
        )
        gpu_topology = topology_process.stdout.strip()
        gpu_topology_status = topology_process.returncode
    except (OSError, subprocess.TimeoutExpired) as exc:
        gpu_topology = str(exc)
        gpu_topology_status = 127
    return {
        "platform": platform.platform(),
        "operating_system": platform.uname()._asdict(),
        "cpu_model": _cpu_model(),
        "logical_cpu_count": os.cpu_count(),
        "physical_memory_bytes": _physical_memory_bytes(),
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "task_python": task_python,
        "task_runtime": _task_runtime_metadata(task_python),
        "gpu_inventory_status": gpu_inventory_status,
        "gpu_inventory": gpu_inventory,
        "gpu_topology_status": gpu_topology_status,
        "gpu_topology": gpu_topology.splitlines(),
        "gpu_topology_sha256": hashlib.sha256(gpu_topology.encode("utf-8")).hexdigest(),
        "hardware_profile": resolved.hardware.get("name"),
        "cost_profile": resolved.hardware.get("legacy", {}).get("cost_profile"),
        "profile_mismatch_allowed": profile_mismatch_allowed,
        "paper_comparable": not profile_mismatch_allowed and not resolved.provisional,
        "resource_roots": {
            name: os.environ.get(name)
            for name in ("MIL_MODEL_ROOT", "MIL_DATA_ROOT", "MIL_RESULTS_ROOT", "HF_HOME")
        },
    }


def _snapshot_artifact_inputs(
    run_directory: Path, resolved: ResolvedExperiment
) -> list[dict[str, str]]:
    root = repo_root()
    sources = _artifact_input_sources(resolved)
    snapshots: list[dict[str, str]] = []
    for source in sources:
        try:
            relative = source.resolve().relative_to(root.resolve())
        except ValueError:
            relative = Path("artifact/configs/external") / source.name
        destination = run_directory / "inputs" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        snapshots.append(
            {
                "path": relative.as_posix(),
                "checksum_sha256": sha256_file(destination),
            }
        )
    return snapshots


def _run_preflight(
    python_executable: str,
    profile: str,
    allow_profile_mismatch: bool,
    suites: Sequence[str] = (),
    methods: Sequence[str] = (),
    parameters: Sequence[str] = (),
) -> dict[str, Any]:
    command = [
        python_executable,
        "-m",
        "artifact.mil_artifact.doctor",
        "--profile",
        profile,
        "--json",
    ]
    for flag, values in (('--suite', suites), ('--method', methods), ('--parameter', parameters)):
        for value in values:
            command.extend([flag, value])
    if allow_profile_mismatch:
        command.append("--allow-profile-mismatch")
    try:
        completed = subprocess.run(
            command,
            cwd=repo_root(),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=600,
            env=child_environment(repo_root()),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ArtifactConfigError(f"could not run doctor with {python_executable}: {exc}") from exc
    if completed.returncode:
        raise ArtifactConfigError(
            f"doctor preflight failed with {python_executable}: {completed.stdout.strip()}"
        )
    try:
        report = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise ArtifactConfigError(f"doctor returned invalid JSON: {exc}") from exc
    if report.get("summary", {}).get("error"):
        raise ArtifactConfigError(f"doctor preflight reported errors: {report}")
    if not report.get('model_snapshots'):
        raise ArtifactConfigError('doctor did not return verified model snapshots')
    return report


def _profile_summary(
    resolved: ResolvedExperiment,
    task_timeout_seconds: float | None = None,
) -> dict[str, Any]:
    blocked = [task.task_id for task in resolved.tasks if not task.executable]
    provisional_mappings = [
        task.task_id for task in resolved.tasks if task.mapping_status not in {"verified", "internal"}
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "profile": resolved.profile["name"],
        "execution_protocol": resolved.execution_protocol["protocol_id"],
        "execution_protocol_status": resolved.execution_protocol.get("status"),
        "task_timeout_seconds": (
            task_timeout_seconds
            if task_timeout_seconds is not None
            else resolved.profile.get("task_timeout_seconds")
        ),
        "description": resolved.profile.get("description"),
        "provisional": resolved.provisional,
        "execution_ready": bool(resolved.profile.get("execution_ready", False)),
        "profile_path": os.fspath(resolved.profile_path),
        "hardware_path": os.fspath(resolved.hardware_path),
        "methods_path": os.fspath(resolved.methods_path),
        "execution_protocol_path": os.fspath(resolved.execution_protocol_path),
        "task_count": len(resolved.tasks),
        "blocked_task_count": len(blocked),
        "blocked_tasks": blocked,
        "provisional_mapping_task_count": len(provisional_mappings),
    }


def print_dry_run(
    resolved: ResolvedExperiment,
    as_json: bool = False,
    task_timeout_seconds: float | None = None,
) -> None:
    summary = _profile_summary(resolved, task_timeout_seconds)
    if as_json:
        print(
            json.dumps(
                {"summary": summary, "tasks": [task.to_dict() for task in resolved.tasks]},
                indent=2,
                sort_keys=True,
            )
        )
        return

    print(
        f"profile={summary['profile']} tasks={summary['task_count']} "
        f"provisional={str(summary['provisional']).lower()} "
        f"execution_ready={str(summary['execution_ready']).lower()}"
    )
    if summary["blocked_task_count"]:
        print(f"blocked_tasks={summary['blocked_task_count']} (see method reason below)")
    for index, task in enumerate(resolved.tasks, 1):
        state = "READY" if task.executable else "BLOCKED"
        label = f" paper_label={task.paper_label!r}" if task.paper_label else ""
        print(f"[{index:03d}/{len(resolved.tasks):03d}] {state} {task.task_id}{label}")
        print(f"  {task.display_command()}")
        if task.blocked_reason:
            print(f"  reason: {task.blocked_reason}")


def _copy_directory(source: Path, destination: Path) -> None:
    if not source.is_dir():
        raise RuntimeError(f"required legacy directory is missing: {source}")
    shutil.copytree(source, destination)


def _stage_shared_resources(run_directory: Path) -> Path:
    """Create a run-owned copy of legacy cost data.

    This intentionally avoids symlinks back into benchmarks/: a legacy write
    should never mutate the preserved source tree.
    """

    staged = run_directory / "legacy_resources"
    if staged.exists():
        return staged
    staged.mkdir()
    benchmark_root = repo_root() / "benchmarks"
    for name in ("Cost_Model_per_iter", "Cost_Model_per_iter_zxcpu"):
        source = benchmark_root / name
        if source.exists():
            _copy_directory(source, staged / name)
    return staged


def _stage_task_workdir(task_directory: Path, shared_resources: Path) -> Path:
    workdir = task_directory / "legacy_workdir"
    if workdir.exists():
        return workdir
    workdir.mkdir(parents=True)
    benchmark_root = repo_root() / "benchmarks"

    _copy_directory(benchmark_root / "my_dummy_requests", workdir / "my_dummy_requests")
    output_root = workdir / "test_end2end_schedule"
    output_root.mkdir()
    _copy_directory(
        benchmark_root / "test_end2end_schedule" / "outlen_files",
        output_root / "outlen_files",
    )
    for name in ("Cost_Model_per_iter", "Cost_Model_per_iter_zxcpu"):
        source = shared_resources / name
        if source.exists():
            (workdir / name).symlink_to(source, target_is_directory=True)
    return workdir


def _task_metadata(task: TaskSpec, run_id: str, command: list[str], workdir: Path) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "task_id": task.task_id,
        "profile": task.profile,
        "execution_protocol": task.execution_protocol,
        "workload_id": task.workload_id,
        "test_case": task.test_case,
        "method_id": task.method_id,
        "paper_label": task.paper_label,
        "mapping_status": task.mapping_status,
        "comparison_to_control": task.comparison_to_control,
        "parameters": task.parameters,
        "seed_provenance": {
            "experiment_seed": task.parameters.get("seed"),
            "workload_ratio_seed": task.parameters.get("ratio_seed"),
            "legacy_test_id": task.parameters.get("test_id"),
            "python_hash_seed": task.environment().get("PYTHONHASHSEED"),
            "note": (
                "MIL_EXPERIMENT_SEED initializes Python, NumPy, PyTorch, workload, "
                "scheduler, and output-length sampling. Deterministic GPU algorithms "
                "are not forced because doing so would change timing semantics."
            ),
        },
        "legacy": task.legacy,
        "environment": task.environment(),
        "command": command,
        "working_directory": os.fspath(workdir),
        "git": git_metadata(),
        "status": "pending",
        "attempts": [],
    }


def _successful_task(task_directory: Path) -> bool:
    metadata_path = task_directory / "metadata.json"
    if not metadata_path.is_file():
        return False
    try:
        metadata = read_json(metadata_path)
    except (OSError, ValueError):
        return False
    if metadata.get("status") != "succeeded" or metadata.get("exit_code") != 0:
        return False
    if not (task_directory / "resolved_config.yaml").is_file():
        return False
    log_name = metadata.get("stdout_log")
    if not isinstance(log_name, str) or not log_name:
        return False
    log_path = task_directory / log_name
    if not log_path.is_file():
        return False
    metrics, parser_errors = parse_legacy_log(
        log_path,
        light_rule="LightRule"
        in metadata.get("legacy", {}).get("search_method_baseline", ""),
    )
    return not success_diagnostics(metrics, parser_errors, metadata.get("exit_code"))


def _validate_resume_metadata(
    run_metadata: dict[str, Any],
    resolved: ResolvedExperiment,
    python_executable: str,
    allow_profile_mismatch: bool = False,
    task_timeout_seconds: float | None = None,
) -> None:
    expected_profile = resolved.profile["name"]
    if run_metadata.get("profile") != expected_profile:
        raise ArtifactConfigError(
            f"resume profile mismatch: {run_metadata.get('profile')!r} != {expected_profile!r}"
        )
    current_digest = artifact_input_digest(resolved)
    if run_metadata.get("artifact_input_digest_sha256") != current_digest:
        raise ArtifactConfigError(
            "resume Artifact inputs differ from the immutable run snapshot"
        )
    current_git = git_metadata()
    current_commit = current_git.get('commit')
    if run_metadata.get("git", {}).get("commit") != current_commit:
        raise ArtifactConfigError("resume Git commit differs from the original run")
    if run_metadata.get('git', {}).get('source_snapshot_sha256') != current_git.get('source_snapshot_sha256'):
        raise ArtifactConfigError('resume source snapshot differs from the original run')
    expected_tasks = [task.task_id for task in resolved.tasks]
    if run_metadata.get("tasks") != expected_tasks:
        raise ArtifactConfigError("resume task matrix differs from the original run")
    original_python = run_metadata.get("host", {}).get("task_python")
    if original_python != python_executable:
        raise ArtifactConfigError(
            f"resume Python executable differs: {python_executable!r} != {original_python!r}"
        )
    original_override = bool(run_metadata.get("profile_mismatch_allowed", False))
    if original_override != allow_profile_mismatch:
        raise ArtifactConfigError(
            "resume profile-mismatch policy differs from the original run"
        )
    if run_metadata.get("task_timeout_seconds") != task_timeout_seconds:
        raise ArtifactConfigError(
            "resume task-timeout policy differs from the original run"
        )


def _duration_text(seconds: float) -> str:
    rounded = max(0, int(seconds))
    hours, remainder = divmod(rounded, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _execute_task_matrix(
    run_directory: Path,
    resolved: ResolvedExperiment,
    shared_resources: Path,
    python_executable: str,
    task_timeout_seconds: float | None,
    model_snapshots: dict[str, Any] | None = None,
) -> tuple[list[str], int, int | None]:
    failed_task_ids: list[str] = []
    skipped = 0
    interrupted_signal: int | None = None
    durations: list[float] = []
    run_started = time.monotonic()

    for index, task in enumerate(resolved.tasks, 1):
        task_directory = run_directory / "tasks" / task.task_id
        task_directory.mkdir(parents=True, exist_ok=True)
        if _successful_task(task_directory):
            skipped += 1
            print(f"[{index}/{len(resolved.tasks)}] SKIP {task.task_id}")
            continue

        workdir = _stage_task_workdir(task_directory, shared_resources)
        command = task.command(python_executable=python_executable)
        metadata_path = task_directory / "metadata.json"
        if metadata_path.exists():
            metadata = read_json(metadata_path)
        else:
            metadata = _task_metadata(task, run_directory.name, command, workdir)
        attempt = len(metadata.get("attempts", [])) + 1
        log_path = task_directory / f"stdout.attempt-{attempt}.log"
        metadata["task_timeout_seconds"] = task_timeout_seconds
        metadata['model_snapshots'] = model_snapshots or {}
        metadata["status"] = "running"
        metadata["started_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        metadata.setdefault("attempts", []).append(
            {"attempt": attempt, "log": log_path.name, "started_at": metadata["started_at"]}
        )
        atomic_write_json(metadata_path, metadata)
        dump_yaml(
            task_directory / "resolved_config.yaml",
            task.to_dict(python_executable=python_executable),
        )
        (task_directory / "command.txt").write_text(
            task.display_command(python_executable=python_executable) + "\n",
            encoding="utf-8",
        )

        print(f"[{index}/{len(resolved.tasks)}] RUN {task.task_id}")
        environment = child_environment(repo_root())
        environment.update(task.environment())
        if model_snapshots:
            environment['MIL_MODEL_SNAPSHOTS'] = json.dumps(model_snapshots, sort_keys=True)
            environment.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        task_started = time.monotonic()
        with log_path.open("w", encoding="utf-8", errors="backslashreplace") as log:
            return_code, received_signal, timed_out = _run_scoped_process(
                command,
                workdir,
                environment,
                log,
                task_timeout_seconds,
            )
        duration = time.monotonic() - task_started
        durations.append(duration)
        metadata["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        metadata["duration_seconds"] = duration
        metadata["exit_code"] = return_code
        metadata["timed_out"] = timed_out
        metadata["status"] = "succeeded" if return_code == 0 else "failed"
        metadata["stdout_log"] = log_path.name
        metadata["attempts"][-1].update(
            {
                "finished_at": metadata["finished_at"],
                "duration_seconds": duration,
                "exit_code": return_code,
                "timed_out": timed_out,
            }
        )
        if received_signal is not None:
            metadata["interrupted_by_signal"] = received_signal
            interrupted_signal = received_signal
        atomic_write_json(metadata_path, metadata)

        remaining = len(resolved.tasks) - index
        mean_duration = sum(durations) / len(durations)
        print(
            f"  progress={index}/{len(resolved.tasks)} "
            f"elapsed={_duration_text(time.monotonic() - run_started)} "
            f"estimated_remaining={_duration_text(mean_duration * remaining)}"
        )
        if return_code:
            failed_task_ids.append(task.task_id)
            reason = "timed out" if timed_out else f"failed with exit code {return_code}"
            print(f"  {reason}; see {log_path}", file=sys.stderr)
        if interrupted_signal is not None:
            break

    return failed_task_ids, skipped, interrupted_signal


def execute(
    resolved: ResolvedExperiment,
    results_root: Path,
    resume: Path | None,
    python_executable: str,
    preflight_report: dict[str, Any],
    allow_profile_mismatch: bool = False,
    gpu_lock_timeout: float = 0.0,
    plot_after_run: bool = False,
    figure_ids: Sequence[str] = (),
    figure_output: Path | None = None,
    task_timeout_seconds: float | None = None,
) -> int:
    if task_timeout_seconds is None:
        task_timeout_seconds = resolved.profile.get("task_timeout_seconds")
    if task_timeout_seconds is not None and (
        not math.isfinite(float(task_timeout_seconds)) or task_timeout_seconds <= 0
    ):
        raise ArtifactConfigError("task timeout must be a positive number")
    run_directory = (
        resume.resolve() if resume else _new_run_directory(results_root, resolved)
    )
    if resume and not run_directory.is_dir():
        raise ArtifactConfigError(f"resume directory does not exist: {run_directory}")

    run_path = run_directory / "run.json"
    if run_path.exists():
        run_metadata = read_json(run_path)
        if run_metadata.get('model_snapshots', {}) != preflight_report.get('model_snapshots', {}):
            raise ArtifactConfigError('resume model snapshots differ from the original run')
        _validate_resume_metadata(
            run_metadata,
            resolved,
            python_executable,
            allow_profile_mismatch,
            task_timeout_seconds,
        )
    else:
        artifact_inputs = _snapshot_artifact_inputs(run_directory, resolved)
        run_metadata = {
            **_profile_summary(resolved, task_timeout_seconds),
            "run_id": run_directory.name,
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "git": git_metadata(),
            "host": _host_metadata(
                python_executable,
                resolved,
                allow_profile_mismatch,
            ),
            "preflight": preflight_report,
            "model_snapshots": preflight_report.get('model_snapshots', {}),
            "artifact_inputs": artifact_inputs,
            "artifact_input_digest_sha256": artifact_input_digest(resolved),
            "profile_mismatch_allowed": allow_profile_mismatch,
            "status": "running",
            "tasks": [task.task_id for task in resolved.tasks],
        }
        atomic_write_json(run_path, run_metadata)
        dump_yaml(run_directory / "profile.yaml", resolved.profile)
        dump_yaml(run_directory / "hardware.yaml", resolved.hardware)

    devices = sorted({device for task in resolved.tasks for device in task.gpu_devices})
    lock_root = run_directory.parent / ".gpu-locks"
    run_metadata["gpu_lock_root"] = os.fspath(lock_root)
    atomic_write_json(run_path, run_metadata)
    try:
        with _acquire_gpu_locks(
            lock_root,
            devices,
            run_directory.name,
            gpu_lock_timeout,
        ):
            shared_resources = _stage_shared_resources(run_directory)
            failed_task_ids, skipped, interrupted_signal = _execute_task_matrix(
                run_directory,
                resolved,
                shared_resources,
                python_executable,
                task_timeout_seconds,
                preflight_report.get('model_snapshots'),
            )
    except ArtifactConfigError:
        run_metadata.update(
            {
                "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                "status": "blocked",
                "blocked_reason": "GPU lock acquisition failed",
            }
        )
        atomic_write_json(run_path, run_metadata)
        raise

    run_metadata.update(
        {
            "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "status": (
                "interrupted"
                if interrupted_signal is not None
                else "failed"
                if failed_task_ids
                else "succeeded"
            ),
            "failed_tasks": len(failed_task_ids),
            "failed_task_ids": failed_task_ids,
            "skipped_tasks": skipped,
            "interrupted_signal": interrupted_signal,
        }
    )
    atomic_write_json(run_path, run_metadata)

    try:
        records = collect_run(run_directory)
        verification = verify_run(
            run_directory,
            artifact_root() / "manifests" / "expected_results.yaml",
        )
        figure_outputs: list[Path] = []
        if plot_after_run and not verification["issue_count"]:
            figure_outputs = plot_run(
                run_directory,
                figure_output or run_directory / "figures",
                figure_ids,
            )
    except Exception as exc:
        run_metadata["status"] = "failed"
        run_metadata["postprocessing"] = {
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        atomic_write_json(run_path, run_metadata)
        raise RuntimeError(f"run postprocessing failed: {exc}") from exc
    invalid_task_ids = [
        str(record.get("task_id"))
        for record in records
        if record.get("status") == "invalid"
    ]
    if verification["issue_count"]:
        run_metadata["status"] = "failed"
    run_metadata["invalid_task_ids"] = invalid_task_ids
    run_metadata["postprocessing"] = {
        "status": "failed" if verification["issue_count"] else "passed",
        "collector_record_count": len(records),
        "verification_status": verification["status"],
        "verification_issue_count": verification["issue_count"],
        "verification_warning_count": verification["warning_count"],
        "figure_outputs": [os.fspath(path) for path in figure_outputs],
    }
    atomic_write_json(run_path, run_metadata)

    if failed_task_ids:
        print("failed_task_ids=" + ",".join(failed_task_ids), file=sys.stderr)
    print(f"verification={verification['status']}")
    print(f"run_directory={run_directory}")
    if interrupted_signal is not None:
        return 128 + interrupted_signal
    return 1 if failed_task_ids or verification["issue_count"] else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="smoke", help="profile name or YAML path")
    parser.add_argument("--suite", action="append", default=[], help="workload id filter; repeatable")
    parser.add_argument("--method", action="append", default=[], help="method id filter; repeatable")
    parser.add_argument(
        "--parameter",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="exact scalar parameter filter for scale/seed selection; repeatable",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="validate and print commands (default)")
    mode.add_argument("--execute", action="store_true", help="run the expanded commands")
    mode.add_argument(
        "--plot-only",
        type=Path,
        metavar="RUN_DIR",
        help="render configured figures from structured results without GPU/model checks",
    )
    parser.add_argument("--json", action="store_true", help="emit dry-run as JSON")
    parser.add_argument(
        "--plot-after-run",
        action="store_true",
        help="render matching configured figures after collection and verification",
    )
    parser.add_argument("--figure", action="append", default=[], help="figure id; repeatable")
    parser.add_argument("--figure-output", type=Path, default=None)
    parser.add_argument("--allow-provisional", action="store_true")
    parser.add_argument(
        "--allow-profile-mismatch",
        action="store_true",
        help="allow GPU profile mismatch and mark the run non-comparable",
    )
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument(
        "--gpu-lock-timeout",
        type=float,
        default=0.0,
        help="seconds to wait for run-scoped GPU locks (default: fail immediately)",
    )
    parser.add_argument("--results-root", type=Path, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument(
        "--task-timeout-seconds",
        type=float,
        default=None,
        help=(
            "positive per-task wall-clock limit; overrides the profile value "
            "and is recorded in run provenance"
        ),
    )
    parser.add_argument("--python", default=sys.executable, help="Python executable for legacy command")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.plot_only is not None:
            if (
                args.json
                or args.plot_after_run
                or args.task_timeout_seconds is not None
            ):
                raise ArtifactConfigError(
                    "--json, --plot-after-run, and --task-timeout-seconds are not "
                    "valid with --plot-only"
                )
            run_directory = args.plot_only.expanduser().resolve()
            output = (
                args.figure_output.expanduser().resolve()
                if args.figure_output
                else run_directory / "figures"
            )
            outputs = plot_run(run_directory, output, args.figure)
            for path in outputs:
                print(path)
            return 0

        if not args.execute and (args.plot_after_run or args.figure or args.figure_output):
            raise ArtifactConfigError(
                "figure options require --execute --plot-after-run or --plot-only RUN_DIR"
            )

        parameter_filters = parse_parameter_filters(args.parameter)
        resolved = expand_experiment(
            args.profile,
            args.suite,
            args.method,
            parameter_filters,
        )
        task_timeout_seconds = (
            args.task_timeout_seconds
            if args.task_timeout_seconds is not None
            else resolved.profile.get("task_timeout_seconds")
        )
        if task_timeout_seconds is not None and (
            not math.isfinite(float(task_timeout_seconds)) or task_timeout_seconds <= 0
        ):
            raise ArtifactConfigError("--task-timeout-seconds must be positive")
        if not args.execute:
            print_dry_run(
                resolved,
                as_json=args.json,
                task_timeout_seconds=task_timeout_seconds,
            )
            return 0

        if args.json:
            raise ArtifactConfigError("--json is only valid for dry-run")
        if resolved.provisional and not args.allow_provisional:
            raise ArtifactConfigError(
                "profile or hardware mapping is provisional; review it and pass --allow-provisional"
            )
        if not resolved.profile.get("execution_ready", False):
            blockers = resolved.profile.get("readiness_blockers", [])
            detail = "; ".join(str(item) for item in blockers) or "profile has not been released"
            raise ArtifactConfigError(f"profile is not execution-ready: {detail}")
        blocked = [task for task in resolved.tasks if not task.executable]
        if blocked:
            detail = "; ".join(f"{task.method_id}: {task.blocked_reason}" for task in blocked)
            raise ArtifactConfigError(f"selected task matrix contains blocked methods: {detail}")
        git = git_metadata()
        if git["dirty"] and not args.allow_dirty:
            raise ArtifactConfigError(
                "working tree is dirty; commit the Artifact inputs or pass --allow-dirty"
            )
        preflight = _run_preflight(
            args.python,
            args.profile,
            args.allow_profile_mismatch,
            args.suite,
            args.method,
            args.parameter,
        )
        root = (args.results_root or _default_results_root()).expanduser().resolve()
        return execute(
            resolved,
            root,
            args.resume,
            args.python,
            preflight,
            args.allow_profile_mismatch,
            args.gpu_lock_timeout,
            args.plot_after_run,
            args.figure,
            args.figure_output.expanduser().resolve() if args.figure_output else None,
            task_timeout_seconds,
        )
    except (ArtifactConfigError, OSError, RuntimeError, ValueError) as exc:
        print(f"artifact runner error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
