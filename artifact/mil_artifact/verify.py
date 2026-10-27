"""Validate structured results and compare them with frozen expectations."""

from __future__ import annotations

import argparse
from collections import Counter
import math
import re
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from .config import artifact_root, load_yaml
from .hashing import sha256_file
from .schema import (
    METRIC_FIELDS,
    SCHEMA_VERSION,
    atomic_write_json,
    read_json,
    read_jsonl,
    validate_metric_record,
)


def _matches(record: Mapping[str, Any], selector: Mapping[str, Any]) -> bool:
    for key, expected in selector.items():
        if key.startswith("parameters."):
            actual = record.get("parameters", {}).get(key.split(".", 1)[1])
        else:
            actual = record.get(key)
        if actual != expected:
            return False
    return True


def _compare(actual: float, expected: float, absolute: float, relative: float) -> bool:
    tolerance = max(absolute, abs(expected) * relative)
    return math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance)


def _source_path(run_directory: Path, relative_path: str) -> Path | None:
    try:
        candidate = (run_directory / relative_path).resolve()
        candidate.relative_to(run_directory.resolve())
        return candidate
    except (OSError, ValueError):
        return None


def _provenance_issues(run_directory: Path, record: Mapping[str, Any]) -> list[str]:
    if not record.get("provenance_required"):
        return []
    issues: list[str] = []
    commit = record.get('git_commit')
    snapshot_hash = record.get('source_snapshot_sha256')
    if not (isinstance(commit, str) and commit) and not snapshot_hash:
        issues.append('missing provenance field git_commit or source_snapshot_sha256')
    if snapshot_hash is not None:
        snapshot_path = run_directory / 'inputs/artifact/source_snapshot.json'
        if not isinstance(snapshot_hash, str) or not re.fullmatch(r'[0-9a-f]{64}', snapshot_hash):
            issues.append('invalid source_snapshot_sha256')
        elif not snapshot_path.is_file():
            issues.append('missing source snapshot evidence')
        elif sha256_file(snapshot_path) != snapshot_hash:
            issues.append('source snapshot checksum mismatch')
        else:
            try:
                snapshot = read_json(snapshot_path)
                valid = (isinstance(snapshot, dict) and snapshot.get('schema_version') == 1
                         and snapshot.get('kind') == 'source_snapshot'
                         and isinstance(snapshot.get('files_sha256'), dict)
                         and bool(snapshot['files_sha256']))
            except (OSError, ValueError):
                valid = False
            if not valid:
                issues.append('invalid source snapshot evidence')
    for field in (
        "artifact_input_digest_sha256",
        "task_metadata_checksum_sha256",
        "resolved_config_checksum_sha256",
        "source_checksum_sha256",
    ):
        value = record.get(field)
        if not isinstance(value, str) or not value:
            issues.append(f"missing provenance field {field}")

    source_log = record.get("source_log")
    if not isinstance(source_log, str) or not source_log:
        issues.append("missing provenance field source_log")
        return issues
    source_path = _source_path(run_directory, source_log)
    if source_path is None:
        issues.append(f"source_log escapes run directory: {source_log}")
    elif not source_path.is_file():
        issues.append(f"source_log is missing: {source_log}")
    else:
        if record.get("source_checksum_sha256") != sha256_file(source_path):
            issues.append(f"source_log checksum mismatch: {source_log}")
        if record.get("source_size_bytes") != source_path.stat().st_size:
            issues.append(f"source_log size mismatch: {source_log}")

    task_id = record.get("task_id")
    if isinstance(task_id, str) and task_id:
        provenance_files = (
            (
                "task_metadata_checksum_sha256",
                Path("tasks") / task_id / "metadata.json",
            ),
            (
                "resolved_config_checksum_sha256",
                Path("tasks") / task_id / "resolved_config.yaml",
            ),
        )
        for field, relative_path in provenance_files:
            path = _source_path(run_directory, relative_path.as_posix())
            if path is None:
                issues.append(f"provenance path escapes run directory: {relative_path}")
                continue
            if not path.is_file():
                issues.append(f"provenance input is missing: {relative_path}")
            elif record.get(field) != sha256_file(path):
                issues.append(f"{field} mismatch: {relative_path}")
    return issues


def _task_coverage_issues(run_directory: Path, records: Sequence[Mapping[str, Any]]) -> list[str]:
    issues = []
    if not records:
        issues.append('no task results were collected')
    counts = Counter(record['task_id'] for record in records
                     if isinstance(record.get('task_id'), str))
    duplicates = sorted(task for task, count in counts.items() if count > 1)
    if duplicates:
        issues.append('duplicate task results: ' + ', '.join(duplicates))
    run_path = run_directory / 'run.json'
    if not run_path.is_file():
        return issues
    metadata = read_json(run_path)
    if not isinstance(metadata, dict):
        return issues + ['run.json must be an object']
    # Published historical exports can have no execution task manifest.
    if 'tasks' not in metadata:
        return issues
    planned = metadata['tasks']
    if (not isinstance(planned, list) or not planned
            or any(not isinstance(task, str) or not task for task in planned)):
        return issues + ['run.json.tasks must be a non-empty list of task IDs']
    if len(set(planned)) != len(planned):
        issues.append('duplicate planned task IDs in run.json')
    if metadata.get('status') != 'succeeded':
        issues.append(f"run status is {metadata.get('status')!r}, expected 'succeeded'")
    missing = sorted(set(planned) - set(counts))
    unexpected = sorted(set(counts) - set(planned))
    if missing:
        issues.append('missing planned task results: ' + ', '.join(missing))
    if unexpected:
        issues.append('unexpected task results: ' + ', '.join(unexpected))
    return issues


def verify_run(
    run_directory: Path,
    expected_path: Path,
    require_baseline: bool = False,
) -> dict[str, Any]:
    records = read_jsonl(run_directory / "metrics.jsonl")
    expected_manifest = load_yaml(expected_path)
    issues: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []
    issues.extend({'task_id': '*', 'message': error}
                  for error in _task_coverage_issues(run_directory, records))

    for record in records:
        for error in validate_metric_record(record):
            issues.append({"task_id": str(record.get("task_id")), "message": error})
        for error in _provenance_issues(run_directory, record):
            issues.append({"task_id": str(record.get("task_id")), "message": error})
        if record.get("status") != "succeeded":
            issues.append(
                {
                    "task_id": str(record.get("task_id")),
                    "message": f"task status is {record.get('status')!r}, expected 'succeeded'",
                }
            )
        if record.get("status") == "succeeded":
            metrics = record.get("metrics", {})
            for field in (
                "real_running_time_seconds",
                "search_time_seconds",
                "preparation_time_seconds",
                "total_time_seconds",
            ):
                if metrics.get(field) is None:
                    issues.append(
                        {"task_id": str(record.get("task_id")), "message": f"missing {field}"}
                    )
            for field in METRIC_FIELDS:
                value = metrics.get(field)
                if isinstance(value, (int, float)) and value < 0:
                    issues.append(
                        {
                            "task_id": str(record.get("task_id")),
                            "message": f"negative {field}",
                        }
                    )
            timing_parts = [
                metrics.get("real_running_time_seconds"),
                metrics.get("search_time_seconds"),
            ]
            total = metrics.get("total_time_seconds")
            if total is not None and all(part is not None for part in timing_parts):
                reconstructed = sum(float(part) for part in timing_parts)
                if not math.isclose(float(total), reconstructed, rel_tol=0.0, abs_tol=1e-6):
                    issues.append(
                        {
                            "task_id": str(record.get("task_id")),
                            "message": (
                                f"total_time_seconds={total} does not equal execution + "
                                f"inclusive search ({reconstructed}); preparation is a subset"
                            ),
                        }
                    )
            completed_requests = metrics.get("completed_request_count")
            completion_events = metrics.get("completion_event_count")
            if (
                completed_requests is not None
                and completion_events is not None
                and completed_requests > completion_events
            ):
                issues.append(
                    {
                        "task_id": str(record.get("task_id")),
                        "message": "completed_request_count exceeds completion_event_count",
                    }
                )
            idle_time = metrics.get("gpu_idle_time_seconds")
            capacity_time = metrics.get("gpu_capacity_time_seconds")
            idle_ratio = metrics.get("gpu_idle_ratio")
            if idle_ratio is not None and not 0 <= idle_ratio <= 1:
                issues.append(
                    {
                        "task_id": str(record.get("task_id")),
                        "message": "gpu_idle_ratio must be between zero and one",
                    }
                )
            if (
                idle_time is not None
                and capacity_time is not None
                and idle_time > capacity_time + 1e-6
            ):
                issues.append(
                    {
                        "task_id": str(record.get("task_id")),
                        "message": "gpu_idle_time_seconds exceeds GPU capacity time",
                    }
                )
            exit_code = record.get("exit_code")
            if exit_code not in (None, 0):
                issues.append(
                    {
                        "task_id": str(record.get("task_id")),
                        "message": f"succeeded task has nonzero exit_code {exit_code}",
                    }
                )
        if record.get("errors"):
            issues.append(
                {
                    "task_id": str(record.get("task_id")),
                    "message": "log diagnostics: " + ", ".join(record["errors"]),
                }
            )

    expectations = expected_manifest.get("results", [])
    comparisons: list[dict[str, Any]] = []
    if not expectations:
        item = {"task_id": "*", "message": "expected-results manifest is empty"}
        (issues if require_baseline else warnings).append(item)
    record_profiles = {
        record.get("profile")
        for record in records
        if isinstance(record.get("profile"), str)
    }
    applicable_expectations = [
        expectation
        for expectation in expectations
        if expectation.get("selector", {}).get("profile") is None
        or expectation.get("selector", {}).get("profile") in record_profiles
    ]
    if expectations and not applicable_expectations:
        item = {
            "task_id": "*",
            "message": (
                "no frozen expectation applies to run profile(s): "
                + ", ".join(sorted(record_profiles))
            ),
        }
        (issues if require_baseline else warnings).append(item)
    defaults = expected_manifest.get("default_tolerances", {})
    default_absolute = float(defaults.get("deterministic_absolute") or 0)
    default_relative_value = defaults.get("runtime_relative")

    for expectation in applicable_expectations:
        selector = expectation.get("selector", {})
        matches = [record for record in records if _matches(record, selector)]
        if not matches:
            issues.append({"task_id": "*", "message": f"no record matches {selector}"})
            continue
        successful_matches = [record for record in matches if record.get("status") == "succeeded"]
        minimum_samples = expectation.get("minimum_samples")
        if isinstance(minimum_samples, int) and len(successful_matches) < minimum_samples:
            issues.append(
                {
                    "task_id": "*",
                    "message": (
                        f"selector {selector} has {len(successful_matches)} successful sample(s); "
                        f"requires at least {minimum_samples}"
                    ),
                }
            )

        aggregation = expectation.get("aggregation")
        if aggregation not in (None, "median"):
            issues.append(
                {"task_id": "*", "message": f"unsupported aggregation {aggregation!r}"}
            )
            continue

        for metric, expected_value in expectation.get("metrics", {}).items():
            if metric not in METRIC_FIELDS:
                issues.append({"task_id": "*", "message": f"unknown expected metric {metric}"})
                continue
            if aggregation == "median":
                metric_values = [
                    float(record["metrics"][metric])
                    for record in successful_matches
                    if record.get("metrics", {}).get(metric) is not None
                ]
                if len(metric_values) != len(successful_matches):
                    issues.append(
                        {
                            "task_id": "*",
                            "message": (
                                f"selector {selector} has {len(successful_matches) - len(metric_values)} "
                                f"successful sample(s) missing {metric}"
                            ),
                        }
                    )
                actual_values = [statistics.median(metric_values)] if metric_values else [None]
                task_ids = [f"aggregate:{expectation.get('reference_id', selector)}"]
            else:
                actual_values = [
                    record.get("metrics", {}).get(metric) for record in successful_matches
                ]
                task_ids = [str(record["task_id"]) for record in successful_matches]

            for task_id, actual in zip(task_ids, actual_values):
                tolerance = expectation.get("tolerances", {}).get(metric, {})
                absolute = float(tolerance.get("absolute", default_absolute))
                relative_value = tolerance.get("relative", default_relative_value)
                if relative_value is None:
                    issues.append(
                        {
                            "task_id": task_id,
                            "message": f"relative tolerance for {metric} is not frozen",
                        }
                    )
                    continue
                passed = actual is not None and _compare(
                    float(actual), float(expected_value), absolute, float(relative_value)
                )
                allowed_error = max(absolute, abs(float(expected_value)) * float(relative_value))
                absolute_error = (
                    abs(float(actual) - float(expected_value)) if actual is not None else None
                )
                relative_error = (
                    absolute_error / abs(float(expected_value))
                    if absolute_error is not None and float(expected_value) != 0
                    else None
                )
                comparison = {
                    "task_id": task_id,
                    "metric": metric,
                    "aggregation": aggregation,
                    "sample_count": len(successful_matches),
                    "actual": actual,
                    "expected": expected_value,
                    "absolute_error": absolute_error,
                    "relative_error": relative_error,
                    "allowed_absolute_error": allowed_error,
                    "configured_tolerance": {
                        "absolute": absolute,
                        "relative": float(relative_value),
                    },
                    "passed": passed,
                }
                comparisons.append(comparison)
                if not passed:
                    issues.append(
                        {"task_id": task_id, "message": f"{metric} outside tolerance"}
                    )

    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "failed" if issues else "passed_with_warnings" if warnings else "passed",
        "record_count": len(records),
        "issue_count": len(issues),
        "warning_count": len(warnings),
        "issues": issues,
        "warnings": warnings,
        "comparisons": comparisons,
        "expected_manifest": str(expected_path),
    }
    atomic_write_json(run_directory / "verification.json", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument(
        "--expected",
        type=Path,
        default=artifact_root() / "manifests" / "expected_results.yaml",
    )
    parser.add_argument("--require-baseline", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = verify_run(
            args.input.expanduser().resolve(),
            args.expected.expanduser().resolve(),
            args.require_baseline,
        )
    except (OSError, ValueError) as exc:
        print(f"verification error: {exc}", file=sys.stderr)
        return 2
    print(
        f"verification={report['status']} issues={report['issue_count']} "
        f"warnings={report['warning_count']}"
    )
    return 1 if report["issue_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
