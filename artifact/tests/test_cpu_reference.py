from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from artifact.mil_artifact.compare_schedules import compare_frozen
from artifact.mil_artifact.config import repo_root
from artifact.mil_artifact.cpu_reference import load_reference, validate_references, validate_request
from artifact.mil_artifact.source_contract import fingerprint, source_path, verify_source_contract
from artifact.tests.synthetic import cpu_reference


class SourceContractTests(unittest.TestCase):
    def test_protected_paths_rejected_before_filesystem_access(self):
        for path in ('../source.py', '/source.py', '.git/config', '.git_ori/file',
                     'backup/file', 'no_robot_copy/file', 'vllm-latest-camera-ready/file'):
            with self.subTest(path=path), patch.object(Path, 'is_symlink', side_effect=AssertionError('no access')):
                with self.assertRaises(ValueError):
                    source_path('/repository', path)

    def test_symlink_and_byte_changes_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'linked').symlink_to('/outside')
            with self.assertRaises(ValueError):
                source_path(root, 'linked/anything.py')
            source = root / 'code.py'
            source.write_text('value = 1\n')
            contract = {'schema_version': 1, 'kind': 'source_semantics', 'files': [
                {'path': 'code.py', 'comparison': 'python_ast_and_directives',
                 'fingerprint': fingerprint(source.read_bytes())}]}
            self.assertTrue(verify_source_contract(root, contract)['passed'])
            source.write_text('value = 2\n')
            self.assertFalse(verify_source_contract(root, contract)['passed'])
            contract['files'][0] = {'path': 'code.py', 'comparison': 'bytes', 'sha256': '0' * 64}
            self.assertFalse(verify_source_contract(root, contract)['passed'])


class CpuReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.reference_path = Path(cls.temporary.name) / 'synthetic-reference.json'
        cls.reference_path.write_text(json.dumps(cpu_reference()))
        cls.reference, cls.reference_path = load_reference(cls.reference_path)

    def test_synthetic_reference_and_request_validation(self):
        self.assertEqual(len(self.reference['cases']), 1)
        self.assertEqual(len({c['request_path'] for c in self.reference['cases']}), 1)
        for case in self.reference['cases']:
            validate_request(json.loads((repo_root() / case['request_path']).read_text()))

    def test_invalid_reference_is_not_accepted(self):
        for fault in ('duplicate', 'nonfinite', 'stages', 'resource_identity'):
            reference = deepcopy(self.reference)
            case = reference['cases'][0]
            if fault == 'duplicate':
                reference['cases'].append(deepcopy(case))
            elif fault == 'nonfinite':
                case['actual']['predicted_seconds'] = float('nan')
            elif fault == 'stages':
                case['actual']['stage_times'] = [0]
            else:
                case['recorded_resource_hashes']['cost_table'] = '0' * 64
            with self.subTest(fault=fault), self.assertRaises(ValueError):
                validate_references(reference)

    def test_request_graph_checks(self):
        request = json.loads((repo_root() / self.reference['cases'][0]['request_path']).read_text())
        for fault in ('missing_node', 'unknown_edge', 'cycle', 'length_mismatch'):
            changed = deepcopy(request)
            if fault == 'missing_node':
                changed['input_lengths'].pop('0')
            elif fault == 'unknown_edge':
                changed['input_edges']['0'] = [len(changed['model_paths'])]
            elif fault == 'cycle':
                changed['input_edges']['0'] = [0]
            else:
                changed['input_lengths']['0'].append(1)
            with self.subTest(fault=fault), self.assertRaises(ValueError):
                validate_request(changed)

    def test_portable_comparison_checks_all_inputs_and_values(self):
        case = next(c for c in self.reference['cases'] if c['case_id'] == 'ensemble/mil')
        good = {'method': case['method'], 'request_set_sha256': case['request_set_sha256'],
                'parameters': case['parameters'], 'actual': case['actual'],
                'resource_hashes': {role: spec['sha256'] for role, spec in self.reference['resources'].items()},
                'resources': {'/relocated/host/costs.py': 'irrelevant legacy path'}}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'result.json'
            requests = Path(directory) / 'relocated_requests.json'
            requests.write_bytes((repo_root() / case['request_path']).read_bytes())
            for fault in (None, 'plan_order', 'cost', 'stage', 'parameters', 'missing_role', 'swapped_roles'):
                changed = deepcopy(good)
                if fault == 'plan_order':
                    changed['actual']['plans'].reverse()
                elif fault == 'cost':
                    changed['actual']['predicted_seconds'] += 1
                elif fault == 'stage':
                    changed['actual']['stage_times'][0] += 1
                elif fault == 'parameters':
                    changed['parameters']['sample_size'] = -1
                elif fault == 'missing_role':
                    changed['resource_hashes'].pop('model_configurations')
                elif fault == 'swapped_roles':
                    roles = changed['resource_hashes']
                    roles['cost_table'], roles['output_lengths'] = roles['output_lengths'], roles['cost_table']
                output.write_text(json.dumps(changed))
                with self.subTest(fault=fault):
                    self.assertEqual(compare_frozen(requests, output, self.reference_path)['passed'], fault is None)
            requests.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'different request set'):
                compare_frozen(requests, output, self.reference_path)


if __name__ == '__main__':
    unittest.main()
