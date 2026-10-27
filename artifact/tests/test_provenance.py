import math
import unittest
from artifact.mil_artifact.provenance import attribute, statistics_match


def record(identifier, runtime=None, extra=None, **other):
    metrics = {}
    if runtime is not None:
        metrics['real_running_time_seconds'] = runtime
    if extra is not None:
        metrics['search_time_seconds'] = extra
    if runtime is not None and extra is not None:
        metrics['total_time_seconds'] = runtime + extra
    return {'id': identifier, 'metrics': metrics, **other}


class AttributionTest(unittest.TestCase):
    def test_unique_and_ambiguous_not_chosen_by_mtime(self):
        first = record('one', 10, 2, mtime=0)
        later = record('two', 10, 2, mtime=999)
        self.assertEqual(attribute(10, 12, [first])['status'], 'exact_unique_same_run_candidate')
        result = attribute(10, 12, [first, later])
        self.assertEqual(result['status'], 'exact_ambiguous_same_run_candidates')
        self.assertEqual(result['same_run_candidates'], ['one', 'two'])

    def test_cross_run_and_cpu_only_metric(self):
        result = attribute(10, 12, [record('gpu', 10, 3), record('cpu_only', extra=2)])
        self.assertEqual(result['status'], 'separate_component_candidates')
        self.assertEqual(result['same_run_candidates'], [])
        self.assertEqual(result['components']['extra']['exact_candidates'], ['cpu_only'])

    def test_numeric_provenance_does_not_drop_failed_record(self):
        result = attribute(10, 12, [record('failed', 10, 2, status='failed')])
        self.assertEqual(result['same_run_candidates'], ['failed'])

    def test_missing_zero_is_not_fastest_measurement(self):
        result = attribute(0, 0, [record('zero', 0, 0)])
        self.assertEqual(result['status'], 'missing_placeholder')
        self.assertEqual(result['same_run_candidates'], [])

    def test_integer_compatibility_is_never_exact_identity(self):
        result = attribute(10, 12, [record('rounded', 10.4, 2.2)], rounded=True)
        self.assertEqual(result['status'], 'rounding_compatible_only')
        self.assertEqual(result['same_run_candidates'], [])
        self.assertEqual(result['components']['runtime']['rounding_compatible_candidates'], ['rounded'])
        self.assertEqual(attribute(10, 12, [record('rounded', 10.4, 2.2)])['status'], 'unresolved')

    def test_no_relative_tolerance_for_large_seconds(self):
        self.assertEqual(attribute(10000, 10001, [record('near', 10000.0001, 1)])['status'], 'partial_exact_components')
        self.assertEqual(attribute(1, 2, [record('nan', math.nan, math.nan)])['status'], 'unresolved')

    def test_rounded_four_run_statistics(self):
        samples = [{'runtime': x, 'total': x + 2} for x in [1, 2, 3, 5]]
        stats = {'run_time': 2.75, 'tot_time': 4.75, 'run_time_min': 1, 'run_time_max': 5,
                 'tot_time_min': 3, 'tot_time_max': 7}
        self.assertTrue(statistics_match(samples, stats))
        self.assertFalse(statistics_match(samples[:3], stats))
