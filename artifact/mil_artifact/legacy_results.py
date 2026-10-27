"""Audit checksum-backed legacy result references and recompute aggregates."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from .collect import parse_legacy_log
from .config import (
    artifact_root,
    expand_environment,
    has_unresolved_environment,
    load_yaml,
    repo_root,
)
from .hashing import sha256_file
from .schema import METRIC_FIELDS, SCHEMA_VERSION, atomic_write_json, write_jsonl


def _statistics(values: Sequence[float]) -> dict[str, float]:
    return {
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def audit_reference(
    reference: Mapping[str, Any], verify_checksums: bool = True
) -> dict[str, Any]:
    reference_id = str(reference.get("id", "unknown"))
    light_rule = "LightRule" in str(reference.get("legacy_method", ""))
    source_reports: list[dict[str, Any]] = []
    parsed_metrics: list[dict[str, float | None]] = []
    issues: list[str] = []

    for index, source in enumerate(reference.get("sources", []), 1):
        path_value = source.get("path")
        source_report: dict[str, Any] = {"index": index, "path": path_value}
        if not isinstance(path_value, str) or has_unresolved_environment(path_value):
            source_report.update(status="error", detail="unresolved path")
            issues.append(f"source {index}: unresolved path {path_value}")
            source_reports.append(source_report)
            continue
        path = Path(path_value)
        if not path.is_file():
            source_report.update(status="error", detail="missing file")
            issues.append(f"source {index}: missing {path}")
            source_reports.append(source_report)
            continue

        expected_hash = source.get("checksum_sha256")
        actual_hash = sha256_file(path) if verify_checksums else None
        checksum_matches = not verify_checksums or actual_hash == expected_hash
        metrics, parser_errors = parse_legacy_log(path, light_rule=light_rule)
        if parser_errors:
            issues.append(f"source {index}: parser diagnostics {parser_errors}")
        if not checksum_matches:
            issues.append(f"source {index}: checksum mismatch")
        source_report.update(
            status="error" if parser_errors or not checksum_matches else "ok",
            checksum_status="verified" if verify_checksums else "skipped",
            checksum_sha256=actual_hash,
            size_bytes=path.stat().st_size,
            metrics=metrics,
            parser_errors=parser_errors,
        )
        source_reports.append(source_report)
        if not parser_errors:
            parsed_metrics.append(metrics)

    expected_count = reference.get("sample_count")
    if expected_count != len(parsed_metrics):
        issues.append(
            f"sample_count is {expected_count!r}, parsed {len(parsed_metrics)} successful sources"
        )

    observed: dict[str, dict[str, float]] = {}
    for metric, expected_statistics in reference.get("metrics", {}).items():
        if metric not in METRIC_FIELDS:
            issues.append(f"unknown metric {metric}")
            continue
        values = [
            float(metrics[metric])
            for metrics in parsed_metrics
            if metrics.get(metric) is not None
        ]
        if len(values) != len(parsed_metrics) or not values:
            issues.append(f"metric {metric} is missing from one or more sources")
            continue
        actual_statistics = _statistics(values)
        observed[metric] = actual_statistics
        for statistic_name in ("median", "min", "max"):
            expected = expected_statistics.get(statistic_name)
            actual = actual_statistics[statistic_name]
            if not isinstance(expected, (int, float)) or not math.isclose(
                actual, float(expected), rel_tol=0.0, abs_tol=1e-9
            ):
                issues.append(
                    f"{metric}.{statistic_name}: observed {actual}, expected {expected}"
                )

    return {
        "id": reference_id,
        "status": "failed" if issues else "passed",
        "issues": issues,
        "sources": source_reports,
        "observed_metrics": observed,
    }


def audit_manifest(path: Path, verify_checksums: bool = True) -> dict[str, Any]:
    manifest = expand_environment(load_yaml(path))
    references = [
        audit_reference(reference, verify_checksums)
        for reference in manifest.get("references", [])
    ]
    issue_count = sum(len(reference["issues"]) for reference in references)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "failed" if issue_count else "passed",
        "issue_count": issue_count,
        "reference_count": len(references),
        "references": references,
    }


def export_reference_run(manifest_path: Path, output_directory: Path) -> list[dict[str, Any]]:
    """Regenerate a small, GPU-free structured result package from audited logs."""

    raw_manifest = load_yaml(manifest_path)
    expanded_manifest = expand_environment(raw_manifest)
    report = audit_manifest(manifest_path, verify_checksums=True)
    if report["issue_count"]:
        raise ValueError("cannot export a legacy reference whose audit failed")

    audited_by_id = {reference["id"]: reference for reference in report["references"]}
    records: list[dict[str, Any]] = []
    for raw_reference, expanded_reference in zip(
        raw_manifest.get("references", []), expanded_manifest.get("references", [])
    ):
        reference_id = str(raw_reference["id"])
        audited = audited_by_id[reference_id]
        selector = raw_reference.get("selector", {})
        parameters = {
            key.split(".", 1)[1]: value
            for key, value in selector.items()
            if key.startswith("parameters.")
        }
        for index, (raw_source, expanded_source, source_report) in enumerate(
            zip(
                raw_reference.get("sources", []),
                expanded_reference.get("sources", []),
                audited["sources"],
            ),
            1,
        ):
            source_path = Path(expanded_source["path"])
            try:
                source_display = (
                    source_path.resolve().relative_to(repo_root().resolve()).as_posix()
                )
            except ValueError:
                source_display = str(raw_source["path"])
            records.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "run_id": "user-log-reference-v2",
                    "task_id": f"{reference_id}.sample-{index}",
                    "profile": selector.get("profile", "reference"),
                    "workload_id": selector.get("workload_id"),
                    "method_id": selector.get("method_id"),
                    "paper_label": raw_reference.get("paper_label"),
                    "status": "succeeded",
                    "parameters": {**parameters, "repeat": index},
                    "metrics": source_report["metrics"],
                    "errors": [],
                    "exit_code": 0,
                    "source_log": source_display,
                    "source_checksum_sha256": source_report["checksum_sha256"],
                    "source_size_bytes": source_report["size_bytes"],
                    "historical_reference": True,
                }
            )

    output_directory.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_directory / "metrics.jsonl", records)
    atomic_write_json(
        output_directory / "run.json",
        {
            "schema_version": SCHEMA_VERSION,
            "run_id": "user-log-reference-v2",
            "profile": "reference",
            "status": "published_historical_reference",
            "record_count": len(records),
            "source_manifest": str(manifest_path),
            "source_manifest_checksum_sha256": sha256_file(manifest_path),
        },
    )
    return records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
    )
    parser.add_argument("--skip-checksums", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--export-run",
        type=Path,
        default=None,
        help="write an audited structured reference run to this directory",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest_path = args.manifest.expanduser().resolve()
        report = audit_manifest(manifest_path, not args.skip_checksums)
        if args.export_run:
            if args.skip_checksums:
                raise ValueError("--export-run requires checksum verification")
            export_reference_run(manifest_path, args.export_run.expanduser().resolve())
    except (OSError, ValueError) as exc:
        print(f"legacy result audit error: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        for reference in report["references"]:
            print(
                f"[{reference['status'].upper():6}] {reference['id']}: "
                f"sources={len(reference['sources'])} issues={len(reference['issues'])}"
            )
            for issue in reference["issues"]:
                print(f"  {issue}")
    return 1 if report["issue_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
