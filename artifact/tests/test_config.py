from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from artifact.mil_artifact.config import (
    ArtifactConfigError,
    expand_experiment,
    load_profile,
    parse_parameter_filters,
    repo_root,
)


class ConfigExpansionTest(unittest.TestCase):
    def test_release_profiles_allow_explicit_execution_without_claiming_gpu_measurements(self):
        for name in ('smoke', 'main', 'reference', 'ablation', 'scalability', 'full', 'paper'):
            with self.subTest(profile=name):
                resolved = expand_experiment(name)
                self.assertTrue(resolved.profile['execution_ready'])
                self.assertTrue(resolved.profile['validation_pending'])
                self.assertTrue(resolved.provisional)

    def test_smoke_expands_one_task_per_workload(self) -> None:
        resolved = expand_experiment("smoke")
        self.assertEqual(len(resolved.tasks), 4)
        self.assertEqual(
            {task.workload_id for task in resolved.tasks},
            {"router", "llm_blender", "chain_summary", "mixed"},
        )
        self.assertTrue(resolved.provisional)
        chain = next(task for task in resolved.tasks if task.workload_id == "chain_summary")
        self.assertEqual(chain.method_id, "mil_chain_summary")
        self.assertEqual(chain.legacy["search_method_baseline"], "greedy_saturn")

    def test_main_matrix_size_and_hybridflow_dispatch(self) -> None:
        resolved = expand_experiment("main")
        self.assertEqual(len(resolved.tasks), 1359)
        blocked = [task for task in resolved.tasks if not task.executable]
        self.assertEqual(blocked, [])
        hybridflow_dispatch = {
            task.method_id: task.legacy["search_method_baseline"]
            for task in resolved.tasks
            if task.workload_id == "router"
            and task.parameters["router_replicate_num"] == 1
            and task.parameters["specify_outlen"] is False
            and task.method_id in {"hybridflow", "hybridflow_star"}
        }
        self.assertEqual(
            hybridflow_dispatch,
            {
                "hybridflow": "hybridflow_model_parallel_old",
                "hybridflow_star": "hybridflow_model_parallel",
            },
        )

        chain_mil = [
            task
            for task in resolved.tasks
            if task.workload_id == "chain_summary" and task.paper_label == "Mil"
        ]
        self.assertEqual({task.method_id for task in chain_mil}, {"mil_chain_summary"})
        self.assertEqual(
            {task.legacy["search_method_baseline"] for task in chain_mil},
            {"greedy_saturn"},
        )

    def test_method_filter_builds_legacy_command(self) -> None:
        resolved = expand_experiment("smoke", workload_filter=["router"], method_filter=["mil"])
        task = resolved.tasks[0]
        command = task.command(python_executable="python3")
        self.assertIn("--gen-execplans-baseline", command)
        self.assertEqual(command[command.index("--gen-execplans-baseline") + 1], "ours")
        self.assertEqual(
            command[command.index("--search_method_baseline") + 1], "greedy_saturn_LPT"
        )
        self.assertEqual(command[command.index("--test-case") + 1], "router")
        self.assertEqual(command[command.index("--reqnum") + 1], "5")
        serialized = task.to_dict(python_executable="/opt/artifact/python")
        self.assertEqual(serialized["command"][0], "/opt/artifact/python")
        self.assertIn("/opt/artifact/python", serialized["display_command"])

    def test_min_r_sample_size_is_explicit_environment_provenance(self) -> None:
        resolved = expand_experiment(
            "main", workload_filter=["router"], method_filter=["min_r"]
        )
        task = resolved.tasks[0]
        self.assertEqual(task.parameters["sample_size"], 8)
        self.assertEqual(task.environment()["SAMPLE_SIZE"], "8")
        self.assertIn("SAMPLE_SIZE=8", task.display_command())
        self.assertNotIn("--sample_size", task.command(python_executable="python3"))

    def test_unified_seed_is_explicit_environment_provenance(self) -> None:
        task = expand_experiment(
            "smoke", workload_filter=["router"], method_filter=["mil"]
        ).tasks[0]
        self.assertEqual(task.parameters["seed"], 0)
        self.assertEqual(task.environment()["MIL_EXPERIMENT_SEED"], "0")
        self.assertEqual(task.environment()["PYTHONHASHSEED"], "0")
        self.assertIn("MIL_EXPERIMENT_SEED=0", task.display_command())
        self.assertNotIn("--seed", task.command(python_executable="python3"))

    def test_profile_is_bound_to_versioned_execution_protocol(self) -> None:
        resolved = expand_experiment("smoke", workload_filter=["router"])
        self.assertEqual(
            resolved.profile["execution_protocol"],
            resolved.execution_protocol["protocol_id"],
        )
        self.assertEqual(
            resolved.tasks[0].execution_protocol,
            "camera_ready_reconstruction_v1",
        )
        self.assertIsNone(resolved.profile["task_timeout_seconds"])

    def test_boolean_false_is_not_emitted(self) -> None:
        resolved = expand_experiment(
            "ablation", workload_filter=["router_runtime_adjustment"], method_filter=["mil"]
        )
        commands = [task.command(python_executable="python3") for task in resolved.tasks]
        self.assertEqual(sum("--no_runtime_adjustment" in command for command in commands), 1)

    def test_hybridflow_old_compatibility_dispatch_is_present(self) -> None:
        source = (repo_root() / "benchmarks" / "search_exec_plans.py").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            "search_method_baseline == 'hybridflow_model_parallel_old'",
            source,
        )
        self.assertIn("self.get_candidate_plan_groups_HybridFlow_old(", source)

    def test_final_ablation_method_mapping(self) -> None:
        resolved = expand_experiment("ablation")
        self.assertEqual(len(resolved.tasks), 18)
        router = [task for task in resolved.tasks if task.workload_id == "router_paper_ablation"]
        mixed = [task for task in resolved.tasks if task.workload_id == "mixed_paper_ablation"]
        self.assertEqual(len(router), 8)
        self.assertEqual(len(mixed), 8)
        by_id = {task.method_id: task for task in router}
        self.assertEqual(
            by_id["mil_no_preemption"].legacy["search_method_baseline"],
            "greedy_saturn_LPT_no_preemption",
        )
        self.assertIn(
            "--outlen_known",
            by_id["mil_known_output_length"].command(python_executable="python3"),
        )
        comparison = by_id["mil_known_output_length"].comparison_to_control
        self.assertEqual(comparison["control_method_id"], "mil")
        self.assertEqual(
            comparison["parameter_diff"]["outlen_known"],
            {"control": None, "variant": True},
        )
        self.assertEqual(comparison["legacy_diff"], {})

        runtime = [
            task
            for task in expand_experiment("ablation").tasks
            if task.workload_id == "router_runtime_adjustment"
        ]
        control = next(task for task in runtime if not task.parameters["no_runtime_adjustment"])
        variant = next(task for task in runtime if task.parameters["no_runtime_adjustment"])
        self.assertIsNone(control.comparison_to_control)
        self.assertEqual(variant.comparison_to_control["control_task_id"], control.task_id)
        self.assertEqual(
            variant.comparison_to_control["parameter_diff"]["no_runtime_adjustment"],
            {"control": False, "variant": True},
        )

    def test_scalability_matches_final_method_grid(self) -> None:
        resolved = expand_experiment("scalability")
        self.assertEqual(len(resolved.tasks), 27)
        self.assertEqual(
            {task.parameters["model_replicate_num"] for task in resolved.tasks},
            {1, 2, 4},
        )
        blocked = [task for task in resolved.tasks if not task.executable]
        self.assertEqual(blocked, [])

    def test_full_profile_is_union_of_paper_matrices(self) -> None:
        resolved = expand_experiment("full")
        self.assertEqual(len(resolved.tasks), 1404)
        blocked = [task for task in resolved.tasks if not task.executable]
        self.assertEqual(blocked, [])
        task_ids = {task.task_id for task in resolved.tasks}
        comparison_ids = {
            task.comparison_to_control["control_task_id"]
            for task in resolved.tasks
            if task.comparison_to_control is not None
        }
        self.assertTrue(comparison_ids)
        self.assertLessEqual(comparison_ids, task_ids)

    def test_reference_profile_repeats_without_changing_legacy_cli(self) -> None:
        resolved = expand_experiment("reference")
        self.assertEqual(len(resolved.tasks), 5)
        self.assertEqual({task.parameters["repeat"] for task in resolved.tasks}, {1, 2, 3, 4, 5})
        self.assertEqual(len({task.task_id for task in resolved.tasks}), 5)
        self.assertTrue(
            all("--repeat" not in task.command(python_executable="python3") for task in resolved.tasks)
        )

    def test_parameter_filter_selects_scale_and_seed(self) -> None:
        resolved = expand_experiment(
            "main",
            workload_filter=["router"],
            method_filter=["mil"],
            parameter_filter={"router_replicate_num": 2, "ratio_seed": 0},
        )
        self.assertEqual(len(resolved.tasks), 2)
        self.assertEqual(
            {task.parameters["router_replicate_num"] for task in resolved.tasks},
            {2},
        )
        self.assertEqual(
            {task.parameters["specify_outlen"] for task in resolved.tasks},
            {False, True},
        )

    def test_parameter_filter_parser_preserves_scalar_types(self) -> None:
        self.assertEqual(
            parse_parameter_filters(
                ["ratio_seed=3", "specify_outlen=true", "similar_threshold=0.2"]
            ),
            {"ratio_seed": 3, "specify_outlen": True, "similar_threshold": 0.2},
        )
        with self.assertRaisesRegex(ArtifactConfigError, "duplicate"):
            parse_parameter_filters(["ratio_seed=0", "ratio_seed=1"])

    def test_profile_name_must_be_path_safe(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            profile = Path(temporary) / "unsafe.yaml"
            profile.write_text(
                yaml.safe_dump(
                    {
                        "schema_version": 1,
                        "name": "../../escape",
                        "hardware": "a100_80gb_8gpu",
                        "workloads": [{}],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ArtifactConfigError, "path-safe identifier"):
                load_profile(profile)

    def test_profile_task_timeout_must_be_null_or_positive(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            profile = Path(temporary) / "invalid-timeout.yaml"
            profile.write_text(
                yaml.safe_dump(
                    {
                        "schema_version": 1,
                        "name": "invalid-timeout",
                        "hardware": "a100_80gb_8gpu",
                        "execution_protocol": "camera_ready_reconstruction_v1",
                        "task_timeout_seconds": 0,
                        "workloads": [{}],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ArtifactConfigError, "positive number"):
                load_profile(profile)


if __name__ == "__main__":
    unittest.main()
