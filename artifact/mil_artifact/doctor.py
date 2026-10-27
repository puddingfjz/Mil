"""Read-only preflight checks for the Mil artifact host and resources."""

from __future__ import annotations

import argparse
import ast
import importlib.metadata
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from .config import (
    ArtifactConfigError,
    artifact_root,
    expand_environment,
    expand_experiment,
    has_unresolved_environment,
    load_yaml,
    parse_parameter_filters,
    repo_root,
)
from .hashing import hash_manifest_path
from .package import SNAPSHOT_RECORD, verify_snapshot
from .runtime_identity import child_environment
from .model_runtime import snapshot_records


@dataclass(frozen=True)
class Check:
    name: str
    status: str
    detail: str


def _command(
    arguments: Sequence[str], cwd: Path | None = None, timeout: int = 20,
    env: dict[str, str] | None = None,
) -> tuple[int, str]:
    try:
        completed = subprocess.run(
            list(arguments),
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            timeout=timeout,
            env=env,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 127, str(exc)
    return completed.returncode, completed.stdout.strip()


def _version_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in re.findall(r"\d+", value)[:3])


def _hardware_match_check(
    name: str,
    matches: bool,
    detail: str,
    allow_profile_mismatch: bool,
) -> Check:
    if matches:
        return Check(name, "ok", detail)
    if allow_profile_mismatch:
        return Check(
            name,
            "warning",
            detail + "; mismatch explicitly allowed and results are non-comparable",
        )
    return Check(name, "error", detail)


def _custom_vllm_checks() -> Iterable[Check]:
    from .runtime_identity import verify
    try:
        report = verify(repo_root())
        yield Check('custom_vllm_identity', 'ok',
                    f"Mil-local source and build verified: {report['source_files']} source files; "
                    f"extensions={report['extensions']}; GPU inference remains separately unvalidated")
    except (OSError, ValueError, KeyError, TypeError, StopIteration, subprocess.SubprocessError) as error:
        yield Check('custom_vllm_identity', 'error',
                    f'{error}; follow artifact/env/README.md to build this repository in a fresh source directory; '
                    'an official vLLM wheel with the same version is not a substitute')


def _source_checkout_checks() -> Iterable[Check]:
    if (repo_root() / SNAPSHOT_RECORD).exists():
        try:
            snapshot = verify_snapshot(repo_root())
            yield Check('source_snapshot', 'ok',
                        f"{snapshot['files']} files verified; sha256={snapshot['sha256']}; Git not required")
        except (OSError, ValueError, KeyError, TypeError) as error:
            yield Check('source_snapshot', 'error', str(error))
        return
    code, branch = _command(
        ['git', '--no-optional-locks', 'branch', '--show-current'], cwd=repo_root())
    yield Check('git_branch', 'error' if code else 'ok', branch or 'unable to inspect repository')
    tracked_code, tracked = _command(
        ['git', '--no-optional-locks', 'status', '--porcelain=v1', '--untracked-files=no'],
        cwd=repo_root(), timeout=60)
    untracked_code, untracked = _command(
        ['git', '--no-optional-locks', 'ls-files', '--others', '--exclude-standard'],
        cwd=repo_root(), timeout=60)
    if tracked_code or untracked_code:
        yield Check('git_worktree', 'error', tracked if tracked_code else untracked)
    else:
        count = len(tracked.splitlines()) + len(untracked.splitlines())
        yield Check('git_worktree', 'warning' if count else 'ok', f'{count} changed path(s)')


def _environment_checks(
    resolved: Any,
    allow_profile_mismatch: bool = False,
) -> Iterable[Check]:
    minimum_python = str(resolved.hardware.get("software", {}).get("minimum_python", "3.9"))
    current = platform.python_version()
    status = "ok" if _version_tuple(current) >= _version_tuple(minimum_python) else "error"
    yield Check("python", status, f"{current}; required >= {minimum_python}")

    runtime = load_yaml(artifact_root() / "env" / "runtime.yaml")
    for distribution, expected_version in runtime.get("packages", {}).items():
        try:
            version = importlib.metadata.version(distribution)
            yield Check(
                f"package:{distribution}",
                "ok" if version == str(expected_version) else "error",
                f"{version}; expected {expected_version}",
            )
        except importlib.metadata.PackageNotFoundError:
            yield Check(
                f"package:{distribution}",
                "error",
                f"not installed; expected {expected_version}",
            )

    identity_checks = list(_custom_vllm_checks())
    yield from identity_checks
    if any(check.status == 'error' for check in identity_checks):
        # Do not run the inference entrypoint's --help or hardware/solver probes
        # through a foreign or unverified runtime.
        return

    requires_gurobi = any(
        task.legacy.get("search_method_baseline", "").startswith("saturnLP")
        for task in resolved.tasks
    )
    if requires_gurobi:
        code, license_output = _command(
            [
                sys.executable,
                "-c",
                (
                    "import gurobipy as gp; "
                    "env=gp.Env(empty=True); "
                    "env.setParam('OutputFlag', 0); "
                    "env.start(); "
                    "print('.'.join(map(str, gp.gurobi.version()))); "
                    "env.dispose()"
                ),
            ]
        )
        yield Check(
            "gurobi_license",
            "ok" if code == 0 else "error",
            (
                f"license initialized with Gurobi {license_output}"
                if code == 0
                else "license initialization failed: " + license_output[-500:]
            ),
        )

    expected_cuda = str(runtime.get("torch_cuda", ""))
    code, torch_cuda = _command(
        [sys.executable, "-c", "import torch; print(torch.version.cuda or '')"]
    )
    yield Check(
        "torch_cuda",
        "ok" if code == 0 and torch_cuda == expected_cuda else "error",
        f"{torch_cuda or 'unavailable'}; expected {expected_cuda}",
    )

    entrypoint = repo_root() / "benchmarks" / "schedule_multi_model.py"
    code, output = _command([sys.executable, os.fspath(entrypoint), "--help"], cwd=repo_root(),
                            env=child_environment(repo_root()))
    yield Check(
        "legacy_entrypoint",
        "ok" if code == 0 else "error",
        "--help completed without GPU work" if code == 0 else output[-500:],
    )

    yield from _source_checkout_checks()

    legacy_files = [
        repo_root() / "benchmarks" / "schedule_multi_model.py",
        repo_root() / "benchmarks" / "search_exec_plans.py",
        repo_root() / "benchmarks" / "my_per_iter_latency_estimator.py",
    ]
    absolute_references: list[str] = []
    pattern = re.compile(r"/(?:ssddata|export|home)/[^'\"\s]+")
    for path in legacy_files:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            # Any standalone string expression is documentation/no-op code, not
            # an active filesystem reference. This includes historical command
            # examples that are not the module's first docstring.
            inactive_string_nodes = {
                id(node.value)
                for node in ast.walk(tree)
                if isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
            }
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Constant)
                    and isinstance(node.value, str)
                    and id(node) not in inactive_string_nodes
                    and pattern.search(node.value)
                ):
                    absolute_references.append(
                        f"{path.relative_to(repo_root())}:{getattr(node, 'lineno', '?')}"
                    )
        except (OSError, SyntaxError):
            continue
    yield Check(
        "portable_legacy_paths",
        "error" if absolute_references else "ok",
        "active absolute paths: " + ", ".join(absolute_references[:8])
        if absolute_references
        else "no active machine-local paths found",
    )

    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        yield Check("nvidia_smi", "error", "command not found")
        return
    code, output = _command(
        [
            nvidia_smi,
            "--query-gpu=index,name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    if code:
        yield Check("nvidia_smi", "error", output or f"exit code {code}")
        return
    rows = [row.strip() for row in output.splitlines() if row.strip()]
    expected_count = int(resolved.hardware["gpu"]["count"])
    yield _hardware_match_check(
        "gpu_count",
        len(rows) == expected_count,
        f"detected {len(rows)}; expected {expected_count}",
        allow_profile_mismatch,
    )
    match_regex = resolved.hardware["gpu"].get("name_match_regex")
    names_match = bool(rows) and all(
        re.search(str(match_regex), row) is not None for row in rows
    ) if match_regex else False
    yield _hardware_match_check(
        "gpu_model",
        bool(rows and names_match),
        "; ".join(rows) if rows else "no GPUs reported",
        allow_profile_mismatch,
    )
    expected_memory_mib = int(resolved.hardware["gpu"]["memory_bytes"]) / (1024**2)
    memory_values: list[float] = []
    for row in rows:
        fields = [field.strip() for field in row.split(",")]
        if len(fields) >= 3:
            try:
                memory_values.append(float(fields[2]))
            except ValueError:
                pass
    memory_matches = len(memory_values) == len(rows) and all(
        memory >= expected_memory_mib * 0.99 for memory in memory_values
    )
    yield _hardware_match_check(
        "gpu_memory",
        bool(rows and memory_matches),
        f"detected MiB={memory_values}; expected approximately {expected_memory_mib:g}",
        allow_profile_mismatch,
    )
    code, topology = _command([nvidia_smi, "topo", "-m"])
    yield Check(
        "gpu_topology",
        "warning" if code == 0 else "error",
        "captured; checksum is not frozen" if code == 0 else topology,
    )


def _selected_resources(resolved: Any) -> dict[str, set[str]]:
    """Match resources to the legacy entrypoint's selected task parameters."""
    catalog = load_yaml(artifact_root() / 'configs/paper_experiments.yaml')
    model_sets = catalog['model_sets']
    required = {'models': set(), 'datasets': {'legacy_output_lengths'}}
    for task in resolved.tasks:
        case, params = task.test_case, task.parameters
        if case not in {'router', 'general', 'chain-summary', 'mixed'}:
            raise ArtifactConfigError(f'no published resource mapping for test_case {case!r}')
        if case == 'router':
            versions = {'multiple_choice_question': 'router_multiple_choice',
                        'not_multiple_choice_question': 'router_not_multiple_choice'}
            version = params['router_question_version']
            if version not in versions:
                raise ArtifactConfigError(f'unknown Router dataset version {version!r}')
            required['datasets'].add(versions[version])
            required['models'].update(model_sets['routing'])
        if case in {'general', 'mixed'}:
            models = model_sets['ensembling']
            oracle_count = int(params.get('oracle_model_num', -1))
            if oracle_count != -1:
                models = models[3:3 + oracle_count]
            required['models'].update(models)
            required['datasets'].add('mixinstruct_prepared')
        if case in {'chain-summary', 'mixed'}:
            required['models'].update((params['summarize_model'], params['evaluator_model']))
            required['datasets'].update(('booookscore_all_books', 'booksum'))
    return required


def _disk_checks(resolved: Any, required: dict[str, set[str]] | None = None) -> Iterable[Check]:
    storage = resolved.hardware.get("storage", {})
    roots = storage.get("roots", {})
    overhead_ratio = float(storage.get("bootstrap_overhead_ratio", 0.0))
    configured = {
        "repository": repo_root(),
        "data": Path(os.environ["MIL_DATA_ROOT"]).expanduser()
        if os.environ.get("MIL_DATA_ROOT")
        else None,
        "models": (
            Path(os.environ.get("MIL_MODEL_ROOT") or os.environ.get("HF_HUB_CACHE", ""))
            .expanduser()
            if os.environ.get("MIL_MODEL_ROOT") or os.environ.get("HF_HUB_CACHE")
            else (
                Path(os.environ["HF_HOME"]).expanduser() / "hub"
                if os.environ.get("HF_HOME")
                else Path.home() / ".cache" / "huggingface" / "hub"
            )
        ),
        "results": Path(os.environ["MIL_RESULTS_ROOT"]).expanduser()
        if os.environ.get("MIL_RESULTS_ROOT")
        else artifact_root() / "results",
    }

    missing_payload_bytes = {label: 0 for label in configured}
    selected_payload_bytes = {label: 0 for label in configured}
    uses_external_data = False
    dataset_manifest = expand_environment(
        load_yaml(artifact_root() / "manifests" / "datasets.yaml")
    )
    repository_root = repo_root().resolve()
    data_root = configured["data"].resolve() if configured["data"] else None
    for dataset in dataset_manifest.get("datasets", []):
        if required is not None and dataset.get('id') not in required['datasets']:
            continue
        raw_path = str(dataset.get("path", ""))
        if has_unresolved_environment(raw_path):
            continue
        path = Path(raw_path).expanduser().resolve()
        size_bytes = int(dataset.get("size_bytes", 0))
        if path.is_relative_to(repository_root):
            label = 'repository'
        elif data_root is not None and path.is_relative_to(data_root):
            label = 'data'
            uses_external_data = True
        else:
            continue
        selected_payload_bytes[label] += size_bytes
        if not path.exists():
            missing_payload_bytes[label] += size_bytes

    model_root = configured["models"].expanduser().resolve()
    model_manifest = load_yaml(artifact_root() / "manifests" / "models.yaml")
    for model in model_manifest.get("models", []):
        identifier = str(model.get("id", ""))
        if required is not None and identifier not in required['models']:
            continue
        revision = str(model.get("revision", ""))
        snapshot = (
            model_root
            / ("models--" + identifier.replace("/", "--"))
            / "snapshots"
            / revision
        )
        size_bytes = int(model.get('integrity', {}).get('size_bytes', 0))
        selected_payload_bytes['models'] += size_bytes
        if not snapshot.is_dir():
            missing_payload_bytes["models"] += size_bytes

    for label, configured_path in configured.items():
        if configured_path is None:
            continue
        if required is not None and ((label == 'data' and not uses_external_data)
                                     or (label == 'models' and not required['models'])):
            continue
        requirement = roots.get(label, {})
        if not isinstance(requirement, dict):
            yield Check(f"disk:{label}", "warning", "storage requirement is not frozen")
            continue
        minimum_capacity = requirement.get("minimum_capacity_bytes")
        runtime_free = requirement.get("minimum_runtime_free_bytes")
        if not isinstance(minimum_capacity, int) or not isinstance(runtime_free, int):
            yield Check(f"disk:{label}", "warning", "storage requirement is not frozen")
            continue
        if required is not None and label in {'models', 'data'}:
            # Full-inventory capacity estimates do not apply to a filtered run.
            minimum_capacity = runtime_free + math.ceil(
                selected_payload_bytes[label] * (1 + overhead_ratio))
        task_estimate = 0
        if label == "results":
            task_estimate = len(resolved.tasks) * int(
                requirement.get("estimated_bytes_per_task", 0)
            )
        bootstrap_bytes = math.ceil(missing_payload_bytes[label] * (1 + overhead_ratio))
        required_free = max(runtime_free + bootstrap_bytes, task_estimate)
        path = configured_path.resolve()
        existing = path
        while not existing.exists() and existing != existing.parent:
            existing = existing.parent
        try:
            usage = shutil.disk_usage(existing)
        except OSError as exc:
            yield Check(f"disk:{label}", "error", str(exc))
            continue
        yield Check(
            f"disk:{label}",
            "ok"
            if usage.total >= minimum_capacity and usage.free >= required_free
            else "error",
            (
                f"path={path}; free_bytes={usage.free}; required_free_bytes={required_free}; "
                f"total_bytes={usage.total}; required_capacity_bytes={minimum_capacity}; "
                f"missing_payload_bytes={missing_payload_bytes[label]}; "
                f"profile_task_estimate_bytes={task_estimate}"
            ),
        )


def _dataset_checks(verify_checksums: bool, identifiers: set[str] | None = None) -> Iterable[Check]:
    manifest = expand_environment(load_yaml(artifact_root() / "manifests" / "datasets.yaml"))
    if identifiers is not None:
        for identifier in sorted(identifiers - {row.get('id') for row in manifest.get('datasets', [])}):
            yield Check(f'dataset:{identifier}', 'error', 'required dataset is absent from manifest')
    for dataset in manifest.get("datasets", []):
        identifier = dataset.get("id", "unknown")
        if identifiers is not None and identifier not in identifiers:
            continue
        raw_path = str(dataset.get("path", ""))
        if has_unresolved_environment(raw_path):
            yield Check(f"dataset:{identifier}", "error", f"unresolved path {raw_path}")
            continue
        path = Path(raw_path)
        if not path.exists():
            yield Check(f"dataset:{identifier}", "error", f"missing {path}")
            continue
        expected = dataset.get("checksum_sha256")
        kind = str(dataset.get("checksum_kind", "file"))
        if not expected:
            yield Check(f"dataset:{identifier}", "warning", f"exists at {path}; checksum not frozen")
            continue
        if not verify_checksums:
            yield Check(f"dataset:{identifier}", "warning", f"exists at {path}; checksum skipped")
            continue
        actual, file_count, size_bytes = hash_manifest_path(path, kind)
        count_matches = dataset.get("file_count") in (None, file_count)
        size_matches = dataset.get("size_bytes") in (None, size_bytes)
        matches = actual == expected and count_matches and size_matches
        detail = f"{kind} sha256={actual}; bytes={size_bytes}"
        if file_count is not None:
            detail += f"; files={file_count}"
        yield Check(f"dataset:{identifier}", "ok" if matches else "error", detail)


def _figure_source_checks(verify_checksums: bool) -> Iterable[Check]:
    config = load_yaml(artifact_root() / "configs" / "figures.yaml")
    checked: set[str] = set()
    for figure in config.get("figures", {}).values():
        source_value = figure.get("source")
        if not isinstance(source_value, str) or source_value in checked:
            continue
        checked.add(source_value)
        path = repo_root() / source_value
        name = f"figure_source:{source_value}"
        if not path.is_file():
            yield Check(name, "error", f"missing {path}")
            continue
        expected = figure.get("source_checksum_sha256")
        if not expected:
            yield Check(name, "warning", "source exists; checksum not frozen")
            continue
        if not verify_checksums:
            yield Check(name, "warning", "source exists; checksum skipped")
            continue
        actual, _, size_bytes = hash_manifest_path(path, "file")
        yield Check(
            name,
            "ok" if actual == expected else "error",
            f"sha256={actual}; bytes={size_bytes}",
        )


def _model_checks(identifiers: set[str] | None = None) -> Iterable[Check]:
    manifest = load_yaml(artifact_root() / "manifests" / "models.yaml")
    if identifiers is not None:
        for identifier in sorted(identifiers - {row.get('id') for row in manifest.get('models', [])}):
            yield Check(f'model:{identifier}', 'error', 'required model is absent from manifest')
    for model in manifest.get("models", []):
        identifier = model.get("id", "unknown")
        if identifiers is not None and identifier not in identifiers:
            continue
        try:
            record = snapshot_records([identifier])[identifier]
            yield Check(f'model:{identifier}', 'ok', json.dumps(record, sort_keys=True))
        except (OSError, ValueError, KeyError, TypeError) as error:
            yield Check(f'model:{identifier}', 'error', str(error))


def _cost_checks(verify_checksums: bool) -> Iterable[Check]:
    manifest = expand_environment(load_yaml(artifact_root() / "manifests" / "cost_profiles.yaml"))
    for profile_id, profile in manifest.get("profiles", {}).items():
        source_groups = (
            ("runtime", profile.get("sources", []), True),
            ("generation", profile.get("generation_sources", []), True),
            ("optional", profile.get("optional_sources", []), False),
        )
        for group, sources, required_by_default in source_groups:
            for index, source in enumerate(sources):
                path = Path(source["path"])
                source_id = source.get("id", index)
                name = f"cost_profile:{profile_id}:{group}:{source_id}"
                required = bool(source.get("required", required_by_default))
                if not path.exists():
                    yield Check(
                        name,
                        "error" if required else "warning",
                        f"missing {'required' if required else 'optional'} source: {path}",
                    )
                    continue
                expected = source.get("checksum_sha256")
                if not expected:
                    yield Check(name, "warning", f"exists at {path}; checksum not frozen")
                    continue
                if not verify_checksums:
                    yield Check(name, "warning", f"exists at {path}; checksum skipped")
                    continue
                actual, file_count, size_bytes = hash_manifest_path(
                    path, str(source.get("checksum_kind", "tree-v1"))
                )
                matches = (
                    actual == expected
                    and source.get("file_count") in (None, file_count)
                    and source.get("size_bytes") in (None, size_bytes)
                )
                mismatch_status = "error" if required else "warning"
                count_detail = f"; files={file_count}" if file_count is not None else ""
                yield Check(
                    name,
                    "ok" if matches else mismatch_status,
                    f"sha256={actual}; bytes={size_bytes}{count_detail}; "
                    f"role={source.get('role', 'unspecified')}",
                )


def run_checks(
    profile: str,
    resource_kind: str = "all",
    verify_checksums: bool = True,
    allow_profile_mismatch: bool = False,
    workload_filter: Sequence[str] = (),
    method_filter: Sequence[str] = (),
    parameter_filter: dict[str, Any] | None = None,
) -> list[Check]:
    resolved = expand_experiment(profile, workload_filter, method_filter, parameter_filter)
    required = _selected_resources(resolved)
    checks: list[Check] = []
    if resource_kind in {"all", "environment"}:
        checks.extend(_environment_checks(resolved, allow_profile_mismatch))
        checks.extend(_cost_checks(verify_checksums))
        checks.extend(_disk_checks(resolved, required))
    if resource_kind in {"all", "data"}:
        checks.extend(_dataset_checks(verify_checksums, required['datasets']))
        checks.extend(_figure_source_checks(verify_checksums))
    if resource_kind in {"all", "models"}:
        checks.extend(_model_checks(required['models']))
    return checks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="smoke")
    parser.add_argument('--suite', action='append', default=[])
    parser.add_argument('--method', action='append', default=[])
    parser.add_argument('--parameter', action='append', default=[], metavar='NAME=VALUE')
    parser.add_argument(
        "--resource-kind",
        choices=["all", "environment", "data", "models"],
        default="all",
    )
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--skip-checksums", action="store_true")
    parser.add_argument(
        "--allow-profile-mismatch",
        action="store_true",
        help="downgrade GPU count/model/memory mismatches and mark results non-comparable",
    )
    parser.add_argument("--no-strict", action="store_true", help="report errors but exit successfully")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        checks = run_checks(
            args.profile,
            args.resource_kind,
            not args.skip_checksums,
            args.allow_profile_mismatch,
            args.suite,
            args.method,
            parse_parameter_filters(args.parameter),
        )
    except (ArtifactConfigError, OSError, ValueError) as exc:
        print(f"doctor error: {exc}", file=sys.stderr)
        return 2
    counts = {status: sum(check.status == status for check in checks) for status in ("ok", "warning", "error")}
    if args.json:
        snapshots = {check.name[len('model:'):]: json.loads(check.detail)
                     for check in checks if check.name.startswith('model:') and check.status == 'ok'}
        print(json.dumps({"summary": counts, "checks": [asdict(check) for check in checks],
                          'model_snapshots': snapshots}, indent=2))
    else:
        for check in checks:
            print(f"[{check.status.upper():7}] {check.name}: {check.detail}")
        print(f"summary: ok={counts['ok']} warning={counts['warning']} error={counts['error']}")
    return 0 if args.no_strict or not counts["error"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
