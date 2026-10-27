"""CPU result identity and validation using artificial schedules."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from artifact.mil_artifact import validate_cpu
from artifact.mil_artifact.config import repo_root
from artifact.mil_artifact.hashing import sha256_file
from artifact.tests.synthetic import cpu_reference


class RepresentativeCpuTest(unittest.TestCase):
    def result(self, method='mil'):
        reference = cpu_reference(method)
        case = reference['cases'][0]
        root = repo_root()
        result = {'status': 'completed_cpu_scheduling',
                  'method': method, 'request_set_sha256': case['request_set_sha256'],
                  'parameters': case['parameters'], 'actual': case['actual'],
                  'resource_hashes': {k: v['sha256'] for k, v in reference['resources'].items()},
                  'source_hashes': {n: sha256_file(root / 'benchmarks' / n) for n in validate_cpu.SOURCE_FILES},
                  'imported_modules': {'vllm': str(root / 'vllm/__init__.py'),
                                       'search_exec_plans': str(root / 'benchmarks/search_exec_plans.py'),
                                       'fake_scheduling': str(root / 'benchmarks/fake_scheduling.py')},
                  'cuda_initialized': False, 'compiled_gpu_extension_loaded': False,
                  'gpu_operator_module': '<CPU-only forbidden GPU operators>'}
        return root / case['request_path'], result, reference

    def test_no_reference_means_structure_not_historical_equality(self):
        requests, result, _ = self.result()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'result.json'
            path.write_text(json.dumps(result))
            report = validate_cpu.verify_result(requests, path, None, repo_root(), 'mil')
            self.assertTrue(report['passed'])
            self.assertIsNone(report['regression'])
            self.assertFalse(report['comparison_to_prior_results'])

    def test_identity_inputs_sources_and_structure_fail_independently(self):
        requests, good, _ = self.result()
        mutations = [
            lambda r: r.update(method='max'),
            lambda r: r.update(status='failed'),
            lambda r: r.update(cuda_initialized=True),
            lambda r: r.update(source_hashes={}),
            lambda r: r.update(imported_modules={}),
            lambda r: r.update(resource_hashes={}),
            lambda r: r.update(request_set_sha256='0' * 64),
            lambda r: r['parameters'].update(seed=-1),
            lambda r: r['actual'].update(plans=[]),
            lambda r: r['actual'].update(predicted_seconds=float('nan')),
            lambda r: r['actual'].update(stage_times=[1.0]),
            lambda r: r['imported_modules'].update(vllm='/outside/vllm.py'),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'result.json'
            for mutate in mutations:
                result = copy.deepcopy(good)
                mutate(result)
                path.write_text(json.dumps(result))
                with self.subTest(mutation=mutate):
                    self.assertFalse(validate_cpu.verify_result(requests, path, None, repo_root(), 'mil')['passed'])

    def test_explicit_reference_and_history_are_enforced(self):
        requests, result, reference = self.result()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, ref = root / 'result.json', root / 'reference.json'
            path.write_text(json.dumps(result))
            ref.write_text(json.dumps(reference))
            self.assertTrue(validate_cpu.verify_result(requests, path, ref, repo_root(), 'mil')['passed'])
            reference['cases'][0]['actual']['plans'].reverse()
            ref.write_text(json.dumps(reference))
            self.assertFalse(validate_cpu.verify_result(requests, path, ref, repo_root(), 'mil')['passed'])
            history = {'available': True, 'candidates': [
                {'eligible_for_initial_comparison': True,
                 'expected': {**result['actual'], 'predicted_seconds': 7.0}}]}
            self.assertFalse(validate_cpu.verify_result(requests, path, None, repo_root(), 'mil', history)['passed'])
