"""Stable schemas and atomic serialization helpers for artifact results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = 1
METRIC_FIELDS = (
    "predicted_running_time_seconds",
    "real_running_time_seconds",
    "search_time_seconds",
    "preparation_time_seconds",
    "total_time_seconds",
    "completed_request_count",
    "completion_event_count",
    "failed_request_count",
    "gpu_idle_time_seconds",
    "gpu_capacity_time_seconds",
    "gpu_idle_ratio",
)
TASK_STATUSES = {
    "pending",
    "running",
    "succeeded",
    "failed",
    "invalid",
    "blocked",
    "skipped",
}
COUNTER_FIELDS = (
    "completed_request_count",
    "completion_event_count",
    "failed_request_count",
)


class ResultSchemaError(ValueError):
    """Raised when a result record is malformed."""


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False)
        stream.write("\n")
    temporary.replace(path)


def write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(dict(record), sort_keys=True, ensure_ascii=False))
            stream.write("\n")
    temporary.replace(path)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ResultSchemaError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ResultSchemaError(f"{path}:{line_number}: record must be an object")
            records.append(value)
    return records


def validate_metric_record(record: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if record.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION}")
    for field in ("run_id", "task_id", "profile", "workload_id", "method_id"):
        if not isinstance(record.get(field), str) or not record[field]:
            errors.append(f"{field} must be a non-empty string")
    status = record.get("status")
    if status not in TASK_STATUSES:
        errors.append(f"invalid status: {status!r}")
    parameters = record.get("parameters")
    if not isinstance(parameters, dict):
        errors.append("parameters must be an object")
    metrics = record.get("metrics")
    if not isinstance(metrics, dict):
        errors.append("metrics must be an object")
    else:
        for field in METRIC_FIELDS:
            value = metrics.get(field)
            if value is not None and not isinstance(value, (int, float)):
                errors.append(f"metrics.{field} must be a number or null")
        for field in COUNTER_FIELDS:
            value = metrics.get(field)
            if value is not None and (not isinstance(value, int) or isinstance(value, bool)):
                errors.append(f"metrics.{field} must be an integer or null")
    diagnostics = record.get("errors")
    if not isinstance(diagnostics, list):
        errors.append("errors must be an array")
    return errors
