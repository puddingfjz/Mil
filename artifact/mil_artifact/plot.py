"""Render configured paper figures from structured Artifact metrics."""

from __future__ import annotations

import argparse
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from .config import artifact_root, load_yaml
from .schema import read_jsonl


def _matches_parameters(record: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    parameters = record.get("parameters", {})
    return all(parameters.get(name) == value for name, value in expected.items())


def _figure_coordinates(specification: Mapping[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    """Normalize a regular x-axis or heterogeneous paper settings to coordinates."""

    configured = specification.get("coordinates")
    if configured is not None:
        if not isinstance(configured, list) or not configured:
            raise ValueError("configured coordinates must be a non-empty list")
        coordinates: list[dict[str, Any]] = []
        for index, raw_coordinate in enumerate(configured):
            if not isinstance(raw_coordinate, dict):
                raise ValueError("each configured coordinate must be a mapping")
            workload_id = raw_coordinate.get("workload_id")
            if not isinstance(workload_id, str) or not workload_id:
                raise ValueError("each configured coordinate requires workload_id")
            label = raw_coordinate.get("label", raw_coordinate.get("value", index))
            coordinates.append(
                {
                    "workload_id": workload_id,
                    "fixed": dict(raw_coordinate.get("fixed", {})),
                    "value": raw_coordinate.get("value", label),
                    "label": label,
                }
            )
        return str(specification.get("x_parameter", "paper_setting")), coordinates

    workload_id = specification["workload_id"]
    fixed = specification.get("fixed", {})
    x_parameter = specification["x"]["parameter"]
    x_values = specification["x"]["values"]
    x_labels = specification["x"].get("labels", x_values)
    if len(x_labels) != len(x_values):
        raise ValueError("configured x.labels must have the same length as x.values")
    return x_parameter, [
        {
            "workload_id": workload_id,
            "fixed": {**fixed, x_parameter: value},
            "value": value,
            "label": label,
        }
        for value, label in zip(x_values, x_labels)
    ]


def _aggregation_for_method(specification: Mapping[str, Any], method_id: str) -> str:
    configured = specification.get("aggregation", "median")
    if isinstance(configured, str):
        name = configured
    elif isinstance(configured, dict):
        overrides = configured.get("methods", {})
        name = overrides.get(method_id, configured.get("default", "median"))
    else:
        raise ValueError("figure aggregation must be a string or mapping")
    if name not in {"mean", "median"}:
        raise ValueError(f"unsupported figure aggregation: {name!r}")
    return name


def _aggregate(values: Sequence[float], name: str) -> float | None:
    if not values:
        return None
    if name == "mean":
        return statistics.fmean(values)
    return statistics.median(values)


def build_figure_table(
    records: Sequence[Mapping[str, Any]],
    specification: Mapping[str, Any],
    default_method_order: Sequence[str],
    lower_metric: str,
    total_metric: str,
) -> dict[str, Any]:
    """Select and aggregate records for one configured figure.

    Repeated records at the exact same method/x-coordinate use the configured
    median/mean policy. Records from other parameter settings are never pooled.
    """

    x_parameter, coordinates = _figure_coordinates(specification)
    method_order = specification.get("method_order", default_method_order)
    labels: dict[str, str] = {}
    values: dict[str, list[dict[str, float | int | None]]] = {}
    for method_id in method_order:
        aggregation = _aggregation_for_method(specification, method_id)
        method_rows: list[dict[str, float | int | None]] = []
        for coordinate in coordinates:
            matches = [
                record
                for record in records
                if record.get("status") == "succeeded"
                and record.get("workload_id") == coordinate["workload_id"]
                and record.get("method_id") == method_id
                and _matches_parameters(record, coordinate["fixed"])
            ]
            lower_values = [
                float(record["metrics"][lower_metric])
                for record in matches
                if record.get("metrics", {}).get(lower_metric) is not None
            ]
            total_values = [
                float(record["metrics"][total_metric])
                for record in matches
                if record.get("metrics", {}).get(total_metric) is not None
            ]
            method_rows.append(
                {
                    "x": coordinate["value"],
                    "sample_count": min(len(lower_values), len(total_values)),
                    "aggregation": aggregation,
                    "lower": _aggregate(lower_values, aggregation),
                    "lower_min": min(lower_values) if lower_values else None,
                    "lower_max": max(lower_values) if lower_values else None,
                    "total": _aggregate(total_values, aggregation),
                    "total_min": min(total_values) if total_values else None,
                    "total_max": max(total_values) if total_values else None,
                }
            )
            if matches:
                labels[method_id] = str(matches[0].get("paper_label") or method_id)
        if any(row["sample_count"] for row in method_rows):
            values[method_id] = method_rows

    return {
        "workload_ids": list(dict.fromkeys(row["workload_id"] for row in coordinates)),
        "x_parameter": x_parameter,
        "x_values": [row["value"] for row in coordinates],
        "x_labels": [row["label"] for row in coordinates],
        "method_order": [method for method in method_order if method in values],
        "labels": labels,
        "values": values,
    }


def plot_run(
    run_directory: Path,
    output_directory: Path,
    figure_ids: Sequence[str] = (),
    config_path: Path | None = None,
) -> list[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for plotting") from exc

    config = load_yaml(config_path or artifact_root() / "configs" / "figures.yaml")
    records = read_jsonl(run_directory / "metrics.jsonl")
    figures = config.get("figures", {})
    unknown = sorted(set(figure_ids) - set(figures))
    if unknown:
        raise ValueError("unknown figure id(s): " + ", ".join(unknown))
    selected_ids = list(figure_ids) if figure_ids else list(figures)
    lower_metric = str(config["metrics"]["lower_bar"])
    total_metric = str(config["metrics"]["stacked_total"])
    default_method_order = config.get("method_order", [])

    output_directory.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for figure_id in selected_ids:
        specification = figures[figure_id]
        single_metric = specification.get("single_metric")
        figure_lower_metric = str(single_metric or lower_metric)
        figure_total_metric = str(single_metric or total_metric)
        table = build_figure_table(
            records,
            specification,
            default_method_order,
            figure_lower_metric,
            figure_total_metric,
        )
        methods = table["method_order"]
        if not methods:
            continue

        x_values = table["x_values"]
        x_positions = list(range(len(x_values)))
        width = min(0.8 / len(methods), 0.14)
        offset_origin = -width * (len(methods) - 1) / 2
        figure, axis = plt.subplots(figsize=(max(8.0, len(x_values) * 2.2), 5.0))
        for method_index, method_id in enumerate(methods):
            rows = table["values"][method_id]
            lower = [
                float(row["lower"]) if row["lower"] is not None else math.nan for row in rows
            ]
            overhead = [
                max(0.0, float(row["total"]) - float(row["lower"]))
                if row["total"] is not None and row["lower"] is not None
                else math.nan
                for row in rows
            ]
            positions = [
                position + offset_origin + method_index * width for position in x_positions
            ]
            bars = axis.bar(
                positions,
                lower,
                width,
                label=table["labels"].get(method_id, method_id),
            )
            color = bars.patches[0].get_facecolor() if bars.patches else None
            axis.bar(
                positions,
                overhead,
                width,
                bottom=lower,
                color=color,
                alpha=0.5,
                hatch="///",
            )
            error_bars = specification.get("error_bars", {})
            if method_id in error_bars.get("methods", []):
                metric_prefix = str(error_bars.get("metric", "total"))
                centers: list[float] = []
                selected_positions: list[float] = []
                lower_errors: list[float] = []
                upper_errors: list[float] = []
                for position, row in zip(positions, rows):
                    center = row.get(metric_prefix)
                    minimum = row.get(f"{metric_prefix}_min")
                    maximum = row.get(f"{metric_prefix}_max")
                    if center is None or minimum is None or maximum is None:
                        continue
                    selected_positions.append(position)
                    centers.append(float(center))
                    lower_errors.append(float(center) - float(minimum))
                    upper_errors.append(float(maximum) - float(center))
                if centers:
                    axis.errorbar(
                        selected_positions,
                        centers,
                        yerr=[lower_errors, upper_errors],
                        fmt="none",
                        ecolor=str(error_bars.get("color", "crimson")),
                        capsize=3,
                    )

        axis.set_xticks(x_positions, [str(value) for value in table["x_labels"]])
        axis.set_xlabel(table["x_parameter"].replace("_", " "))
        axis.set_ylabel(str(specification.get("y_label", "Running time (s)")))
        axis.set_title(figure_id.replace("_", " "))
        axis.grid(axis="y", alpha=0.25)
        axis.legend(ncol=min(3, len(methods)))
        figure.tight_layout()
        output = output_directory / specification["output"]
        figure.savefig(output, dpi=200)
        plt.close(figure)
        outputs.append(output)

    if not outputs:
        raise ValueError("no configured figure has matching successful records")
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--figure", action="append", default=[], help="figure id; repeatable")
    parser.add_argument("--config", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_directory = args.input.expanduser().resolve()
    output = args.output.expanduser().resolve() if args.output else run_directory / "figures"
    config_path = args.config.expanduser().resolve() if args.config else None
    try:
        outputs = plot_run(run_directory, output, args.figure, config_path)
    except (OSError, KeyError, ValueError, RuntimeError) as exc:
        print(f"plot error: {exc}", file=sys.stderr)
        return 2
    for path in outputs:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
