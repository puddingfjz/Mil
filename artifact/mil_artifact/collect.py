"""Collect legacy task logs into deterministic JSON and JSONL records."""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

from .hashing import sha256_file
from .schema import METRIC_FIELDS, SCHEMA_VERSION, atomic_write_json, read_json, write_jsonl


_FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_PATTERNS = {
    "real_running_time_seconds": re.compile(rf"total running time:\s*({_FLOAT})s?"),
    "search_time_seconds": re.compile(rf"Total search time:\s*({_FLOAT})"),
    "preparation_time_seconds": re.compile(
        rf"Total time for preparation before search:\s*({_FLOAT})"
    ),
    "fuse_start_seconds": re.compile(rf"begin fusing similar models: --abs:\s*({_FLOAT})"),
    "fuse_end_seconds": re.compile(rf"finish fusing similar models: --abs:\s*({_FLOAT})"),
    "begin_search_seconds": re.compile(rf"begin search: --abs:\s*({_FLOAT})"),
    "hybridflow_predicted_seconds": re.compile(rf"other hybridflow res infor:\s*\(\s*({_FLOAT})"),
}
_COMPLETION_PATTERN = re.compile(r"finish one req:.*output\.request_id:\s*(\S+)")
REQUIRED_TIMING_FIELDS = (
    "real_running_time_seconds",
    "search_time_seconds",
    "preparation_time_seconds",
    "total_time_seconds",
)


def success_diagnostics(
    metrics: dict[str, float | int | None],
    parser_errors: Sequence[str],
    exit_code: int | None,
) -> list[str]:
    """Return reasons a nominally successful process is not a valid sample."""

    diagnostics = list(parser_errors)
    for field in REQUIRED_TIMING_FIELDS:
        if metrics.get(field) is None:
            diagnostics.append(f"missing_{field}")
    if exit_code not in (None, 0):
        diagnostics.append(f"nonzero_exit_code:{exit_code}")
    return list(dict.fromkeys(diagnostics))


def _idle_time_seconds(value: Any) -> tuple[float, int]:
    if not isinstance(value, dict):
        raise ValueError("non_computation_ranges must be a mapping")
    total = 0.0
    for entries in value.values():
        if not isinstance(entries, list):
            raise ValueError("GPU non-computation entries must be a list")
        for entry in entries:
            if not isinstance(entry, list) or len(entry) != 2:
                raise ValueError("invalid non-computation entry")
            ranges = entry[1]
            if not isinstance(ranges, list) or len(ranges) != 2:
                raise ValueError("invalid non-computation ranges")
            starts, ends = ranges
            if len(starts) != len(ends):
                raise ValueError("non-computation start/end counts differ")
            for start, end in zip(starts, ends):
                width = float(end) - float(start)
                if width < 0:
                    raise ValueError("negative non-computation interval")
                total += width
    return total, len(value)


def parse_legacy_log(path: Path, light_rule: bool = False) -> tuple[dict[str, float | None], list[str]]:
    metrics: dict[str, float | None] = {field: None for field in METRIC_FIELDS}
    errors: list[str] = []
    aux: dict[str, float] = {}
    predicted = 0.0 if light_rule else None
    completion_ids: set[str] = set()
    completion_events = 0
    gpu_idle_time: float | None = None
    observed_gpu_count: int | None = None

    try:
        with path.open("r", encoding="utf-8", errors="ignore") as stream:
            for line in stream:
                if "Best group seq: " in line and not light_rule:
                    fields = line.split()
                    if len(fields) >= 2:
                        try:
                            predicted = float(fields[-2])
                        except ValueError:
                            pass
                if "update_workload_dict min_cost:" in line and predicted is not None:
                    try:
                        predicted += float(line.split("update_workload_dict min_cost:", 1)[1])
                    except ValueError:
                        errors.append("could_not_parse_incremental_predicted_time")
                for name, pattern in _PATTERNS.items():
                    match = pattern.search(line)
                    if not match:
                        continue
                    value = float(match.group(1))
                    if name == "hybridflow_predicted_seconds":
                        predicted = value
                    elif name in metrics:
                        metrics[name] = value
                    else:
                        aux[name] = value
                lowered = line.lower()
                completion = _COMPLETION_PATTERN.search(line)
                if completion:
                    completion_events += 1
                    completion_ids.add(completion.group(1))
                if line.startswith("non_computation_ranges="):
                    try:
                        gpu_idle_time, observed_gpu_count = _idle_time_seconds(
                            ast.literal_eval(line.split("=", 1)[1].strip())
                        )
                    except (SyntaxError, ValueError, TypeError):
                        errors.append("could_not_parse_non_computation_ranges")
                if "traceback (most recent call last)" in lowered and "traceback" not in errors:
                    errors.append("traceback")
                if "cuda out of memory" in lowered and "cuda_out_of_memory" not in errors:
                    errors.append("cuda_out_of_memory")
                if "assertionerror" in lowered and "assertion_error" not in errors:
                    errors.append("assertion_error")
    except OSError as exc:
        return metrics, [f"log_read_error:{exc}"]

    metrics["predicted_running_time_seconds"] = predicted
    metrics["completed_request_count"] = (
        len(completion_ids) if completion_events else None
    )
    metrics["completion_event_count"] = completion_events or None
    metrics["failed_request_count"] = None
    metrics["gpu_idle_time_seconds"] = gpu_idle_time
    if (
        gpu_idle_time is not None
        and observed_gpu_count
        and metrics["real_running_time_seconds"] is not None
    ):
        capacity = float(metrics["real_running_time_seconds"]) * observed_gpu_count
        metrics["gpu_capacity_time_seconds"] = capacity
        metrics["gpu_idle_ratio"] = gpu_idle_time / capacity if capacity else None
    if metrics["preparation_time_seconds"] is None:
        begin = aux.get("begin_search_seconds")
        finish = aux.get("fuse_end_seconds")
        if begin is not None and finish is not None:
            metrics["preparation_time_seconds"] = finish - begin
    total_parts = (
        metrics["real_running_time_seconds"],
        metrics["search_time_seconds"],
    )
    # search_exec_plans measures search from time1, before model construction
    # and fusion. Preparation is a subset of that interval, not another cost.
    if all(part is not None for part in total_parts):
        metrics["total_time_seconds"] = sum(
            float(part) for part in total_parts if part is not None
        )
    return metrics, errors


def _task_directories(run_directory: Path) -> Iterable[Path]:
    tasks = run_directory / "tasks"
    if not tasks.is_dir():
        return []
    return sorted(path for path in tasks.iterdir() if path.is_dir())


def collect_run(run_directory: Path) -> list[dict[str, Any]]:
    run_metadata = read_json(run_directory / "run.json")
    run_id = run_metadata.get("run_id", run_directory.name)
    records: list[dict[str, Any]] = []
    for task_directory in _task_directories(run_directory):
        metadata_path = task_directory / "metadata.json"
        if not metadata_path.is_file():
            continue
        metadata = read_json(metadata_path)
        log_name = metadata.get("stdout_log")
        if not log_name:
            attempts = metadata.get("attempts", [])
            log_name = attempts[-1].get("log") if attempts else None
        log_path = task_directory / log_name if log_name else None
        resolved_config_path = task_directory / "resolved_config.yaml"
        metrics = {field: None for field in METRIC_FIELDS}
        errors: list[str] = []
        source_checksum = None
        source_size_bytes = None
        if log_path and log_path.is_file():
            metrics, errors = parse_legacy_log(
                log_path,
                light_rule="LightRule" in metadata.get("legacy", {}).get(
                    "search_method_baseline", ""
                ),
            )
            source_checksum = sha256_file(log_path)
            source_size_bytes = log_path.stat().st_size
        else:
            errors.append("missing_stdout_log")

        status = metadata.get("status", "failed")
        if status == "succeeded":
            errors = success_diagnostics(metrics, errors, metadata.get("exit_code"))
            if errors:
                status = "invalid"

        record = {
            "schema_version": SCHEMA_VERSION,
            "parser_schema_version": SCHEMA_VERSION,
            "provenance_required": True,
            "run_id": run_id,
            "experiment_id": metadata.get("task_id", task_directory.name),
            "task_id": metadata.get("task_id", task_directory.name),
            "profile": metadata.get("profile", run_metadata.get("profile")),
            "execution_protocol": metadata.get(
                "execution_protocol", run_metadata.get("execution_protocol")
            ),
            "workload_id": metadata.get("workload_id"),
            "method_id": metadata.get("method_id"),
            "paper_label": metadata.get("paper_label"),
            "status": status,
            "comparison_to_control": metadata.get("comparison_to_control"),
            "parameters": metadata.get("parameters", {}),
            "seed_provenance": metadata.get("seed_provenance"),
            "metrics": metrics,
            "errors": errors,
            "exit_code": metadata.get("exit_code"),
            "timed_out": bool(metadata.get("timed_out", False)),
            "task_timeout_seconds": metadata.get("task_timeout_seconds"),
            "source_log": str(log_path.relative_to(run_directory)) if log_path else None,
            "source_checksum_sha256": source_checksum,
            "source_size_bytes": source_size_bytes,
            "git_commit": run_metadata.get("git", {}).get("commit"),
            "source_snapshot_sha256": run_metadata.get("git", {}).get("source_snapshot_sha256"),
            "artifact_input_digest_sha256": run_metadata.get(
                "artifact_input_digest_sha256"
            ),
            "task_metadata_checksum_sha256": sha256_file(metadata_path),
            "resolved_config_checksum_sha256": (
                sha256_file(resolved_config_path)
                if resolved_config_path.is_file()
                else None
            ),
        }
        atomic_write_json(task_directory / "metrics.json", record)
        records.append(record)
    records.sort(key=lambda record: record["task_id"])
    metrics_path = run_directory / "metrics.jsonl"
    write_jsonl(metrics_path, records)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "git_commit": run_metadata.get("git", {}).get("commit"),
        "source_snapshot_sha256": run_metadata.get("git", {}).get("source_snapshot_sha256"),
        "artifact_input_digest_sha256": run_metadata.get(
            "artifact_input_digest_sha256"
        ),
        "record_count": len(records),
        "succeeded": sum(record["status"] == "succeeded" for record in records),
        "failed": sum(record["status"] == "failed" for record in records),
        "invalid": sum(record["status"] == "invalid" for record in records),
        "records_with_errors": sum(bool(record["errors"]) for record in records),
        "metrics_checksum_sha256": sha256_file(metrics_path),
        "inputs": [
            {
                "path": record["source_log"],
                "checksum_sha256": record["source_checksum_sha256"],
                "size_bytes": record["source_size_bytes"],
            }
            for record in records
            if record["source_log"] is not None
        ],
    }
    atomic_write_json(run_directory / "summary.json", summary)
    return records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="run directory")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        records = collect_run(args.input.expanduser().resolve())
    except (OSError, ValueError) as exc:
        print(f"collector error: {exc}", file=sys.stderr)
        return 2
    print(f"collected {len(records)} task record(s) into {args.input / 'metrics.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
