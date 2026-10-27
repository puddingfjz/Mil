from __future__ import annotations

import hashlib
import ast
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from artifact.mil_artifact.doctor import (
    Check,
    _custom_vllm_checks,
    _cost_checks,
    _disk_checks,
    _environment_checks,
    _hardware_match_check,
)


class HardwareMismatchTest(unittest.TestCase):
    def test_mismatch_is_error_by_default(self) -> None:
        check = _hardware_match_check("gpu_model", False, "different GPU", False)
        self.assertEqual(check.status, "error")

    def test_explicit_override_is_warning_and_non_comparable(self) -> None:
        check = _hardware_match_check("gpu_model", False, "different GPU", True)
        self.assertEqual(check.status, "warning")
        self.assertIn("non-comparable", check.detail)


class SelectedResourcesTest(unittest.TestCase):
    def test_resource_selection_uses_test_case_and_parameters_not_display_id(self):
        from artifact.mil_artifact import doctor
        from artifact.mil_artifact.config import artifact_root, load_yaml
        catalog = load_yaml(artifact_root() / 'configs/paper_experiments.yaml')
        def select(case, **parameters):
            return doctor._selected_resources(SimpleNamespace(tasks=[SimpleNamespace(
                workload_id='figure_6_setting_a1b2c3', test_case=case, parameters=parameters)]))
        router = select('router', router_question_version='multiple_choice_question')
        self.assertEqual(router['datasets'], {'router_multiple_choice', 'legacy_output_lengths'})
        self.assertEqual(router['models'], set(catalog['model_sets']['routing']))
        oracle = select('general', oracle_model_num=2)
        self.assertEqual(oracle['models'], set(catalog['model_sets']['ensembling'][3:5]))
        chain = select('chain-summary', summarize_model='custom/summary', evaluator_model='custom/judge')
        self.assertEqual(chain['models'], {'custom/summary', 'custom/judge'})
        self.assertEqual(chain['datasets'], {'legacy_output_lengths', 'booookscore_all_books', 'booksum'})
        mixed = select('mixed', summarize_model='custom/summary', evaluator_model='custom/judge', oracle_model_num=2)
        self.assertEqual(mixed['models'], chain['models'] | oracle['models'])
        self.assertEqual(mixed['datasets'], chain['datasets'] | oracle['datasets'])
        with self.assertRaisesRegex(ValueError, 'no published resource mapping'):
            select('map-reduce')

    def test_model_catalog_matches_actual_entrypoint_lists(self):
        from artifact.mil_artifact.config import artifact_root, load_yaml, repo_root
        catalog = load_yaml(artifact_root() / 'configs/paper_experiments.yaml')['model_sets']
        tree = ast.parse((repo_root() / 'benchmarks/schedule_multi_model.py').read_text())
        for function_name, group in (('_get_router_bench_data', 'routing'), ('_get_llm_blender_data', 'ensembling')):
            function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == function_name)
            assignment = next(n for n in ast.walk(function) if isinstance(n, ast.Assign)
                              and isinstance(n.value, ast.List)
                              and any(isinstance(t, ast.Name) and t.id == 'model_paths' for t in n.targets))
            self.assertEqual(ast.literal_eval(assignment.value), catalog[group])

    def test_selected_missing_resources_still_fail(self):
        from artifact.mil_artifact import doctor
        with patch.object(doctor, 'load_yaml', return_value={'datasets': [], 'models': []}):
            checks = list(doctor._dataset_checks(False, {'missing-data'}))
            checks += list(doctor._model_checks({'missing/model'}))
        self.assertEqual([(c.name, c.status) for c in checks],
                         [('dataset:missing-data', 'error'), ('model:missing/model', 'error')])
        with tempfile.TemporaryDirectory() as temporary:
            manifest = {'datasets':[{'id':'selected', 'path':str(Path(temporary)/'absent')}]}
            with patch.object(doctor, 'load_yaml', return_value=manifest):
                check, = list(doctor._dataset_checks(False, {'selected'}))
            self.assertEqual(check.status, 'error')
            self.assertIn('missing', check.detail)

    def test_router_only_does_not_require_chain_data_or_ensemble_models(self):
        from artifact.mil_artifact import doctor
        from artifact.mil_artifact.config import artifact_root, load_yaml
        catalog = load_yaml(artifact_root() / 'configs/paper_experiments.yaml')
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            available = root / 'router.json'
            available.write_text('{}')
            datasets = {'schema_version': 1, 'datasets': [
                {'id':'router_not_multiple_choice', 'path':str(available), 'required_for':['router']},
                {'id':'legacy_output_lengths', 'path':str(available), 'required_for':['router']},
                {'id':'booookscore_all_books', 'path':str(root/'missing-books'), 'required_for':['chain_summary']}]}
            models = {'schema_version':1, 'models':[{'id':m} for m in catalog['model_sets']['routing'] + catalog['model_sets']['ensembling']]}
            def manifest(path):
                return {'datasets.yaml':datasets, 'models.yaml':models, 'paper_experiments.yaml':catalog}[path.name]
            def snapshots(ids):
                if set(ids) - set(catalog['model_sets']['routing']):
                    raise ValueError('unavailable unselected model')
                return {m:{'revision':'a'*40} for m in ids}
            with patch.object(doctor, 'load_yaml', side_effect=manifest), \
                    patch.object(doctor, 'snapshot_records', side_effect=snapshots), \
                    patch.object(doctor, '_figure_source_checks', return_value=[]):
                checks = doctor.run_checks('smoke', resource_kind='data', verify_checksums=False,
                                           workload_filter=['router'], method_filter=['mil'])
                checks += doctor.run_checks('smoke', resource_kind='models',
                                            workload_filter=['router'], method_filter=['mil'])
            self.assertFalse([c for c in checks if c.status == 'error'], checks)
            self.assertNotIn('dataset:booookscore_all_books', {c.name for c in checks})
            self.assertEqual({c.name[6:] for c in checks if c.name.startswith('model:')},
                             set(catalog['model_sets']['routing']))


class SolverRequirementTest(unittest.TestCase):
    @patch('artifact.mil_artifact.doctor._custom_vllm_checks',
           return_value=[Check('custom_vllm_identity', 'ok', 'test fixture')])
    @patch("artifact.mil_artifact.doctor._command")
    @patch("artifact.mil_artifact.doctor.load_yaml")
    @patch("artifact.mil_artifact.doctor.importlib.metadata.version")
    def test_gurobi_license_check_is_profile_gated(
        self, version, load_yaml, command, _identity
    ) -> None:
        version.return_value = "1"
        load_yaml.return_value = {
            "packages": {},
            "torch_cuda": "12.1",
        }
        command.return_value = (0, "12.1")
        hardware = {
            "software": {"minimum_python": "3.9"},
            "gpu": {
                "count": 1,
                "name_match_regex": "12\\.1",
                "memory_bytes": 1,
            },
        }

        without_saturn = SimpleNamespace(
            hardware=hardware,
            tasks=[SimpleNamespace(legacy={"search_method_baseline": "greedy_saturn_LPT"})],
        )
        list(_environment_checks(without_saturn))
        self.assertFalse(
            any("gurobipy as gp" in " ".join(call.args[0]) for call in command.call_args_list)
        )

        command.reset_mock()
        with_saturn = SimpleNamespace(
            hardware=hardware,
            tasks=[SimpleNamespace(legacy={"search_method_baseline": "saturnLP"})],
        )
        list(_environment_checks(with_saturn))
        self.assertTrue(
            any("gurobipy as gp" in " ".join(call.args[0]) for call in command.call_args_list)
        )


class CustomRuntimeTest(unittest.TestCase):
    @patch('artifact.mil_artifact.runtime_identity.verify', side_effect=ValueError('foreign runtime'))
    def test_identity_failure_is_error_not_a_version_warning(self, _verify):
        check, = list(_custom_vllm_checks())
        self.assertEqual(check.status, 'error')
        self.assertIn('official vLLM wheel', check.detail)

    @patch('artifact.mil_artifact.doctor._command')
    @patch('artifact.mil_artifact.doctor.load_yaml', return_value={'packages': {}})
    @patch('artifact.mil_artifact.doctor._custom_vllm_checks',
           return_value=[Check('custom_vllm_identity', 'error', 'missing build')])
    def test_unverified_runtime_stops_before_legacy_or_gpu_probe(self, _identity, _yaml, command):
        resolved = SimpleNamespace(hardware={'software': {}}, tasks=[])
        checks = list(_environment_checks(resolved))
        self.assertEqual(checks[-1].name, 'custom_vllm_identity')
        self.assertEqual(checks[-1].status, 'error')
        command.assert_not_called()


class CostProfileSourceTest(unittest.TestCase):
    @patch("artifact.mil_artifact.doctor.load_yaml")
    def test_required_and_optional_missing_sources_have_distinct_status(
        self, load_yaml
    ) -> None:
        load_yaml.return_value = {
            "profiles": {
                "test": {
                    "sources": [{"id": "runtime", "path": "/missing/runtime"}],
                    "optional_sources": [
                        {"id": "raw", "path": "/missing/raw"}
                    ],
                }
            }
        }
        checks = list(_cost_checks(True))
        self.assertEqual([check.status for check in checks], ["error", "warning"])
        self.assertIn("missing required", checks[0].detail)
        self.assertIn("missing optional", checks[1].detail)

    @patch("artifact.mil_artifact.doctor.load_yaml")
    def test_required_runtime_file_is_checksum_verified(self, load_yaml) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "table.py"
            source.write_bytes(b"frozen cost table\n")
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            load_yaml.return_value = {
                "profiles": {
                    "test": {
                        "sources": [
                            {
                                "id": "runtime",
                                "role": "selected_runtime_cost_table",
                                "path": str(source),
                                "checksum_kind": "file",
                                "size_bytes": source.stat().st_size,
                                "checksum_sha256": digest,
                            }
                        ]
                    }
                }
            }
            checks = list(_cost_checks(True))
        self.assertEqual(len(checks), 1)
        self.assertEqual(checks[0].status, "ok")
        self.assertIn("selected_runtime_cost_table", checks[0].detail)


class DiskRequirementTest(unittest.TestCase):
    def test_filtered_capacity_excludes_unselected_payloads_and_external_data(self):
        from artifact.mil_artifact import doctor
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            requirements = {name: {'minimum_capacity_bytes': 10000, 'minimum_runtime_free_bytes': 10}
                            for name in ('repository', 'models', 'data', 'results')}
            resolved = SimpleNamespace(tasks=[object()], hardware={'storage': {
                'bootstrap_overhead_ratio': 0.05, 'roots': requirements}})
            datasets = {'datasets': [
                {'id': 'selected', 'path': str(root/'router.json'), 'size_bytes': 100},
                {'id': 'unselected', 'path': str(root/'external/books'), 'size_bytes': 10000}]}
            models = {'models': [
                {'id':'org/selected', 'revision':'a'*40, 'integrity':{'size_bytes':200}},
                {'id':'org/unselected', 'revision':'b'*40, 'integrity':{'size_bytes':10000}}]}
            with patch.object(doctor, 'load_yaml', side_effect=[datasets, models]), \
                    patch.object(doctor, 'repo_root', return_value=root), \
                    patch.object(doctor.shutil, 'disk_usage', return_value=SimpleNamespace(total=1000, free=500)), \
                    patch.dict(os.environ, {'MIL_MODEL_ROOT':str(root/'cache'),
                                            'MIL_DATA_ROOT':str(root/'external')}):
                checks = {c.name:c for c in _disk_checks(resolved, {'models':{'org/selected'}, 'datasets':{'selected'}})}
            self.assertNotIn('disk:data', checks)
            self.assertEqual(checks['disk:models'].status, 'ok')
            self.assertIn('required_capacity_bytes=220', checks['disk:models'].detail)
            self.assertIn('missing_payload_bytes=200', checks['disk:models'].detail)
            self.assertIn('missing_payload_bytes=100', checks['disk:repository'].detail)

    @patch("artifact.mil_artifact.doctor.shutil.disk_usage")
    @patch("artifact.mil_artifact.doctor.load_yaml")
    def test_accounts_for_missing_payloads_and_profile_task_count(
        self, load_yaml, disk_usage
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repository = root / "repo"
            data = root / "data"
            models = root / "models"
            results = root / "results"
            for path in (repository, data, models, results):
                path.mkdir()
            load_yaml.side_effect = [
                {
                    "datasets": [
                        {"path": str(repository / "missing.bin"), "size_bytes": 100},
                        {"path": str(data / "missing.bin"), "size_bytes": 50},
                    ]
                },
                {
                    "models": [
                        {
                            "id": "org/model",
                            "revision": "rev",
                            "integrity": {"size_bytes": 200},
                        }
                    ]
                },
            ]
            disk_usage.return_value = SimpleNamespace(total=1000, free=200)
            requirements = {
                label: {
                    "minimum_capacity_bytes": 500,
                    "minimum_runtime_free_bytes": 10,
                    **({"estimated_bytes_per_task": 20} if label == "results" else {}),
                }
                for label in ("repository", "data", "models", "results")
            }
            resolved = SimpleNamespace(
                hardware={
                    "storage": {
                        "bootstrap_overhead_ratio": 0.05,
                        "roots": requirements,
                    }
                },
                tasks=[object(), object(), object()],
            )
            environment = {
                "MIL_DATA_ROOT": str(data),
                "MIL_MODEL_ROOT": str(models),
                "MIL_RESULTS_ROOT": str(results),
            }
            with patch("artifact.mil_artifact.doctor.repo_root", return_value=repository):
                with patch("artifact.mil_artifact.doctor.artifact_root", return_value=root):
                    with patch.dict("os.environ", environment, clear=False):
                        checks = list(_disk_checks(resolved))

        by_name = {check.name: check for check in checks}
        self.assertEqual(by_name["disk:repository"].status, "ok")
        self.assertEqual(by_name["disk:data"].status, "ok")
        self.assertEqual(by_name["disk:models"].status, "error")
        self.assertIn("required_free_bytes=220", by_name["disk:models"].detail)
        self.assertEqual(by_name["disk:results"].status, "ok")
        self.assertIn("profile_task_estimate_bytes=60", by_name["disk:results"].detail)


if __name__ == "__main__":
    unittest.main()
