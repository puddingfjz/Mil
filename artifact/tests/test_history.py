from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from artifact.mil_artifact import history
from artifact.mil_artifact import compare_schedules
from artifact.mil_artifact.hashing import sha256_file


class HistoryTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.bundle = {'parameters': {'test_case': 'general'}, 'model_paths': ['model'],
                       'input_lengths': {'0': [1, 2]}}

    def record(self, name='test.log', prefix='', suffix='', plan=None, complete=False):
        text = (prefix + 'len(inp_lens) of model 0: [1, 2]\nTotal search time: 1.0\n' +
                (plan if plan is not None else 'Best group seq: [[1]] [3.0] 3.0 sec\n') + suffix)
        path = self.root / name
        path.write_text(text)
        return {'path': name, 'mtime_ns': 1, 'sha256': sha256_file(path),
                'model_paths': ['model'], 'parameters': {'search_method_baseline': 'greedy_saturn_LPT'},
                'status': 'complete_timing' if complete else 'incomplete', 'complete_timing': complete,
                'metrics': {'search_time_seconds': 1.0, 'preparation_time_seconds': 0.2}}

    def test_cpu_only_valid_without_gpu_time(self):
        row = history.inspect_log(self.record(), self.root)
        self.assertTrue(row['cpu_plan_usable'])
        self.assertTrue(row['cpu_timing_available'])
        self.assertFalse(row['gpu_timing_complete'])
        self.assertEqual(row['version']['status'], 'unverified')

    def test_searchonly_name_does_not_certify_completion(self):
        row = history.inspect_log(self.record('searchonly.log', plan='still searching\n'), self.root)
        self.assertFalse(row['cpu_plan_usable'])

    def test_errors_before_completion_reject_cpu_stage(self):
        row = history.inspect_log(self.record(prefix='Traceback (most recent call last)\n'), self.root)
        self.assertFalse(row['cpu_plan_usable'])
        self.assertEqual(row['diagnostics_before_cpu_completion'], ['traceback'])

    def test_later_errors_preserve_cpu_not_gpu(self):
        row = history.inspect_log(self.record(suffix='CUDA out of memory\n', complete=True), self.root)
        self.assertTrue(row['cpu_plan_usable'])
        self.assertFalse(row['gpu_timing_complete'])
        self.assertEqual(row['cpu_status'], 'complete_with_later_diagnostics')

    def test_hash_changes_fail_without_fallback(self):
        record = self.record()
        (self.root / record['path']).write_text('changed')
        with self.assertRaisesRegex(ValueError, 'changed since indexing'):
            history.inspect_log(record, self.root)

    def test_truncated_plan_is_not_cpu_complete(self):
        row = history.inspect_log(self.record(plan='Best group seq: [[1]\n'), self.root)
        self.assertFalse(row['cpu_plan_usable'])
        self.assertIsNotNone(row['parse_error'])

    def test_nonfinite_cost_is_invalid(self):
        row = history.inspect_log(self.record(plan='Best group seq: [[1]] [3.0] nan sec\n'), self.root)
        self.assertFalse(row['cpu_plan_usable'])

    def test_hybridflow_and_lightrule_formats(self):
        hybrid = self.record(plan='best_group_seq: [[1]]\nother hybridflow res infor: (3.0, [])\n')
        hybrid['parameters']['search_method_baseline'] = 'hybridflow_model_parallel'
        self.assertTrue(history.inspect_log(hybrid, self.root)['cpu_plan_usable'])
        light = self.record(plan='Best group seq: [[1]]\n')
        light['parameters']['search_method_baseline'] = 'LightRule'
        self.assertTrue(history.inspect_log(light, self.root)['cpu_plan_usable'])

    def test_no_mtime_ranking_and_no_successful_result_selection(self):
        older = self.record('a_old.log')
        newer = self.record('z_new.log', plan='Best group seq: [[1]] [4.0] 4.0 sec\n')
        newer['mtime_ns'] = 999
        with mock.patch.object(history, 'matches', return_value=True):
            snapshot = history.snapshot_history(self.bundle, {}, [newer, older], self.root)
        self.assertEqual([r['source'] for r in snapshot['candidates']], ['a_old.log', 'z_new.log'])
        result = history.compare_history(snapshot, {'plans': [[1]], 'stage_times': [3.0], 'predicted_seconds': 3.0})
        self.assertFalse(result['passed'])
        self.assertEqual((result['compared'], result['matched']), (2, 1))

    def test_inputs_and_model_order_checked_separately(self):
        record = self.record()
        record['model_paths'] = ['different']
        bundle = {**self.bundle, 'input_lengths': {'0': [9]}}
        with mock.patch.object(history, 'matches', return_value=True):
            result = history.snapshot_history(bundle, {}, [record], self.root)
        row = result['candidates'][0]
        self.assertFalse(row['input_lengths_equal'])
        self.assertFalse(row['model_order_equal'])
        self.assertFalse(result['available'])

    def test_no_candidates_is_not_passed(self):
        self.assertFalse(history.compare_history({'candidates': []}, {})['passed'])

    def test_backup_parent_symlink_and_traversal_refused(self):
        for relative in ('../test.log', '/tmp/test.log', 'backup/test.log', 'vllm-latest-camera-ready/test.log'):
            with self.assertRaises(ValueError):
                history.safe_log_path(self.root, relative)
        (self.root/'linked').symlink_to(self.root, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'symlinks'):
            history.safe_log_path(self.root, 'linked/test.log')

    def test_compact_evidence_removes_arrays_not_differences(self):
        result = history.compact_snapshot({'candidates': [
            {'expected': {'plans': [[1]], 'predicted_seconds': 3}, 'comparison': {'passed': False}}]})
        row = result['candidates'][0]
        self.assertNotIn('expected', row)
        self.assertEqual(row['initial_predicted_seconds'], 3)
        self.assertFalse(row['comparison']['passed'])

    def test_sources_entrypoint_uses_the_same_stage_policy(self):
        record = self.record('cpu_only.log')
        request, result = self.root/'requests.json', self.root/'result.json'
        bundle = {**self.bundle, 'application': 'llm_blender'}
        request.write_text(json.dumps(bundle))
        result.write_text(json.dumps({'method': 'mil', 'request_set_sha256': sha256_file(request),
            'actual': {'plans': [[1]], 'stage_times': [3.0], 'predicted_seconds': 3.0}}))
        with mock.patch.object(compare_schedules, 'repo_root', return_value=self.root), \
             mock.patch.object(history, 'matches', return_value=True):
            report = compare_schedules.compare_result(request, result, [record])
        self.assertTrue(report['all_historical_comparisons_match'])
        self.assertTrue(report['sources'][0]['cpu_plan_usable'])
        self.assertFalse(report['sources'][0]['gpu_timing_complete'])


if __name__ == '__main__':
    unittest.main()
