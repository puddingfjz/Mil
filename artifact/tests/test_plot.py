from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from artifact.mil_artifact.plot import build_figure_table, plot_run
from artifact.mil_artifact.schema import write_jsonl


def _record(method: str, replicate: int, runtime: float, total: float, status: str = "succeeded") -> dict:
    return {
        "status": status,
        "workload_id": "router",
        "method_id": method,
        "paper_label": method.upper(),
        "parameters": {
            "reqnum": 10000,
            "max_token_num": 4096,
            "specify_outlen": False,
            "router_replicate_num": replicate,
        },
        "metrics": {
            "real_running_time_seconds": runtime,
            "total_time_seconds": total,
        },
    }


class FigureSelectionTest(unittest.TestCase):
    def test_filters_fixed_parameters_and_aggregates_exact_coordinate(self) -> None:
        records = [
            _record("mil", 1, 10.0, 12.0),
            _record("mil", 1, 14.0, 16.0),
            _record("mil", 2, 20.0, 23.0),
            _record("max", 1, 30.0, 34.0),
            _record("max", 2, 99.0, 100.0, status="failed"),
            {
                **_record("mil", 1, 1000.0, 1001.0),
                "parameters": {
                    "reqnum": 10000,
                    "max_token_num": 4096,
                    "specify_outlen": True,
                    "router_replicate_num": 1,
                },
            },
        ]
        specification = {
            "workload_id": "router",
            "fixed": {"reqnum": 10000, "max_token_num": 4096, "specify_outlen": False},
            "x": {
                "parameter": "router_replicate_num",
                "values": [1, 2],
                "labels": ["two models", "four models"],
            },
        }
        table = build_figure_table(
            records,
            specification,
            ["mil", "max"],
            "real_running_time_seconds",
            "total_time_seconds",
        )

        self.assertEqual(table["method_order"], ["mil", "max"])
        self.assertEqual(table["x_labels"], ["two models", "four models"])
        self.assertEqual(table["values"]["mil"][0]["lower"], 12.0)
        self.assertEqual(table["values"]["mil"][0]["total"], 14.0)
        self.assertEqual(table["values"]["mil"][1]["sample_count"], 1)
        self.assertEqual(table["values"]["max"][1]["sample_count"], 0)

    def test_renders_a_configured_figure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "run"
            run.mkdir()
            write_jsonl(
                run / "metrics.jsonl",
                [_record("mil", 1, 10.0, 12.0), _record("mil", 2, 20.0, 23.0)],
            )
            config = root / "figures.yaml"
            config.write_text(
                yaml.safe_dump(
                    {
                        "schema_version": 1,
                        "method_order": ["mil"],
                        "metrics": {
                            "lower_bar": "real_running_time_seconds",
                            "stacked_total": "total_time_seconds",
                        },
                        "figures": {
                            "router_test": {
                                "workload_id": "router",
                                "fixed": {
                                    "reqnum": 10000,
                                    "max_token_num": 4096,
                                    "specify_outlen": False,
                                },
                                "x": {
                                    "parameter": "router_replicate_num",
                                    "values": [1, 2],
                                },
                                "output": "router.png",
                            }
                        },
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
            outputs = plot_run(run, root / "figures", config_path=config)

            self.assertEqual(outputs, [root / "figures" / "router.png"])
            self.assertGreater(outputs[0].stat().st_size, 0)

    def test_renders_a_single_metric_figure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "run"
            run.mkdir()
            record = _record("mil", 1, 10.0, 12.0)
            record["metrics"]["gpu_idle_ratio"] = 0.125
            write_jsonl(run / "metrics.jsonl", [record])
            config = root / "figures.yaml"
            config.write_text(
                yaml.safe_dump(
                    {
                        "schema_version": 1,
                        "method_order": ["mil"],
                        "metrics": {
                            "lower_bar": "real_running_time_seconds",
                            "stacked_total": "total_time_seconds",
                        },
                        "figures": {
                            "idle_test": {
                                "workload_id": "router",
                                "fixed": {
                                    "reqnum": 10000,
                                    "max_token_num": 4096,
                                    "specify_outlen": False,
                                },
                                "x": {
                                    "parameter": "router_replicate_num",
                                    "values": [1],
                                },
                                "single_metric": "gpu_idle_ratio",
                                "y_label": "Fraction of GPU capacity idle",
                                "output": "idle.png",
                            }
                        },
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )

            outputs = plot_run(run, root / "figures", config_path=config)

            self.assertEqual(outputs, [root / "figures" / "idle.png"])
            self.assertGreater(outputs[0].stat().st_size, 0)

    def test_rejects_mismatched_x_labels(self) -> None:
        with self.assertRaisesRegex(ValueError, "x.labels"):
            build_figure_table(
                [_record("mil", 1, 10.0, 12.0)],
                {
                    "workload_id": "router",
                    "x": {
                        "parameter": "router_replicate_num",
                        "values": [1, 2],
                        "labels": ["two models"],
                    },
                },
                ["mil"],
                "real_running_time_seconds",
                "total_time_seconds",
            )

    def test_combines_heterogeneous_paper_coordinates(self) -> None:
        router = _record("mil", 4, 10.0, 12.0)
        mixed = {
            **_record("mil", 1, 20.0, 24.0),
            "workload_id": "mixed_paper_ablation",
            "parameters": {"reqnum": 500, "model_replicate_num": 1},
        }
        router["workload_id"] = "router_paper_ablation"
        specification = {
            "coordinates": [
                {
                    "label": "LLM Routing",
                    "workload_id": "router_paper_ablation",
                    "fixed": {"reqnum": 10000, "router_replicate_num": 4},
                },
                {
                    "label": "Mixed",
                    "workload_id": "mixed_paper_ablation",
                    "fixed": {"reqnum": 500, "model_replicate_num": 1},
                },
            ]
        }
        table = build_figure_table(
            [router, mixed],
            specification,
            ["mil"],
            "real_running_time_seconds",
            "total_time_seconds",
        )
        self.assertEqual(table["x_labels"], ["LLM Routing", "Mixed"])
        self.assertEqual([row["lower"] for row in table["values"]["mil"]], [10.0, 20.0])

    def test_per_method_mean_and_range(self) -> None:
        records = [
            _record("min_r", 1, 10.0, 20.0),
            _record("min_r", 1, 14.0, 30.0),
        ]
        table = build_figure_table(
            records,
            {
                "workload_id": "router",
                "x": {"parameter": "router_replicate_num", "values": [1]},
                "aggregation": {"default": "median", "methods": {"min_r": "mean"}},
            },
            ["min_r"],
            "real_running_time_seconds",
            "total_time_seconds",
        )
        row = table["values"]["min_r"][0]
        self.assertEqual(row["aggregation"], "mean")
        self.assertEqual(row["total"], 25.0)
        self.assertEqual((row["total_min"], row["total_max"]), (20.0, 30.0))


if __name__ == "__main__":
    unittest.main()
