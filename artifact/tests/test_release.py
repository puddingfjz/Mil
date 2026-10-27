"""Source-release integrity is separate from historical performance evidence."""
import unittest
from unittest.mock import patch
from artifact.mil_artifact import release, verify_preparation


class ReleaseAuditTest(unittest.TestCase):
    def test_source_only_audit_does_not_certify_gpu_performance(self):
        with patch.object(release, 'verify', return_value={'checks': [], 'issues': []}):
            checks = release.run_release_checks()
        self.assertEqual(checks[0].status, 'ok')
        self.assertEqual(checks[-1].name, 'gpu_performance')
        self.assertEqual(checks[-1].status, 'warning')

    def test_integrity_failures_are_reported(self):
        report = {'checks': [{'name': 'source', 'passed': False, 'value': 'changed', 'expected': 'original'}],
                  'issues': ['source changed']}
        with patch.object(release, 'verify', return_value=report):
            checks = release.run_release_checks()
        self.assertTrue(any(c.status == 'error' for c in checks))

    def test_source_contract_tampering_is_not_accepted(self):
        with patch.object(verify_preparation, 'verify_source_contract', side_effect=ValueError('tampered')):
            report = verify_preparation.verify()
        self.assertFalse(report['passed'])
        self.assertIn('source_semantics', report['issues'])
