import json
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from artifact.mil_artifact import archive_enrich
from artifact.mil_artifact.archive import bounded_metadata, full_metadata, read_execution_events, final_models_and_counters
from artifact.mil_artifact.paper_results import matches
from artifact.mil_artifact.supporting_analyses import distributions, js_divergence


class HistoricalResultsTests(unittest.TestCase):
    def test_initial_models_stop_at_plan_when_search_timer_is_absent(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'run.log'
            path.write_text("real model_paths: ['component']\nreal model_paths: ['composed']\n"
                            "Best group seq: []\nreal model_paths: ['later runtime']\n")
            self.assertEqual(final_models_and_counters(path)['model_paths'], ['composed'])

    def test_streamed_enrichment_grep_fallback_and_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folder = root / 'benchmarks/test_end2end_schedule'
            folder.mkdir(parents=True)
            (folder / 'a.log').write_text('already scanned\n')
            old = {'path': 'benchmarks/test_end2end_schedule/a.log',
                   'metadata_scan': 'all_file_numeric_markers', 'metrics': {}, 'parameters': {}}
            (folder / 'b.log').write_text('Best group seq: [] [] 10 seconds\n'
                + 'private request text\n' * 120000
                + 'Total search time: 9\nTotal time for preparation before search: 3\n'
                + 'update_workload_dict min_cost: 2\nupdate_workload_dict min_cost: 3\n'
                + 'total running time: 20s\n')
            stdout = io.StringIO()
            with patch.object(archive_enrich, 'repo_root', return_value=root), \
                    patch.object(archive_enrich.shutil, 'which', return_value=None), \
                    patch('sys.argv', ['archive_enrich', '--output', '-', '--reuse', '-']), \
                    patch('sys.stdin', io.StringIO(json.dumps(old) + '\n')), \
                    contextlib.redirect_stdout(stdout):
                self.assertEqual(archive_enrich.main(), 0)
            result = [json.loads(line) for line in stdout.getvalue().splitlines()]
        self.assertEqual(result[0], old)
        self.assertEqual(result[1]['metrics']['incremental_predicted_seconds'], 15)
        self.assertEqual(result[1]['metrics']['total_time_seconds'], 29)
        self.assertNotIn('private request text', stdout.getvalue())

    def test_inclusive_search_is_not_double_counted(self):
        result = bounded_metadata(b'Total search time: 9\nTotal time for preparation before search: 3\n',
                                  b'total running time: 20s\n')
        self.assertEqual(result['metrics']['total_time_seconds'], 29)
        self.assertEqual(result['metrics']['extra_time_seconds'], 9)
        self.assertEqual(result['metrics']['search_after_preparation_seconds'], 6)

    def test_middle_of_large_log_and_search_counters(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'run.log'
            with path.open('wb') as stream:
                stream.write(b'x' * (3 * 1024**2) + b'\n')
                stream.write(b'Total search time: 9\nTotal time for preparation before search: 3\n')
                stream.write(b'Cost_Query_Time, Cost_Query_num, Plan_Group_Enu_Time, Plan_Group_Enu_num, Plan_Group_num_After_Pruning, Select_Best_Time: (2, 10, 1, 4, 2, 0.5)\n')
                stream.write(b'x' * (5 * 1024**2) + b'\ntotal running time: 20s\n')
            result = full_metadata(path)
        self.assertTrue(result['complete_timing'])
        self.assertEqual(result['metrics']['Cost_Query_num'], 10)
        self.assertEqual(result['metrics']['total_time_seconds'], 29)

    def test_timeline_requires_observed_events(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'run.log'
            path.write_text('Best group seq: [] [] 100 seconds\n')
            self.assertFalse(read_execution_events(path)['complete'])
            path.write_text('all stage times: time lengths: []\n')
            self.assertFalse(read_execution_events(path)['complete'])
            path.write_text("MAIN PROCESS: ['model_id: 0, comp_gpus: [0]']\nall stage times: time lengths: [4.5]\n")
            event = read_execution_events(path)
            self.assertTrue(event['complete'])
            self.assertEqual(event['durations_seconds'], [4.5])

    def test_config_matching_never_pools_runtime_or_model_variants(self):
        method = {'legacy': {'gen_execplans_baseline': 'ours', 'search_method_baseline': 'greedy_saturn_LPT'}}
        params = dict(test_case='general', tot_gpu_num=8, machine_name='zxcpu', reqnum=1000,
                      gpu_name='A100-80G', byte_per_gpu=80 * 1024**3, fully_connected_gpu_unit=2,
                      max_token_num=256, **method['legacy'])
        record = {'path': 'run.log', 'parameters': params, 'model_paths': ['a']}
        case = {'test_case': 'general', 'parameters': {'reqnum': 1000, 'max_token_num': 256,
                                                    'no_runtime_adjustment': False}, 'model_paths': ['a']}
        self.assertTrue(matches(record, case, method, {}))
        record['parameters']['no_runtime_adjustment'] = True
        self.assertFalse(matches(record, case, method, {}))
        record['parameters']['no_runtime_adjustment'] = False
        record['parameters']['byte_per_gpu'] = 40 * 1024**3
        self.assertFalse(matches(record, case, method, {}))
        record['parameters']['byte_per_gpu'] = 80 * 1024**3
        record['model_paths'] = ['b']
        self.assertFalse(matches(record, case, method, {}))

    def test_jsd_and_categories_preserve_observed_requests(self):
        self.assertEqual(js_divergence([1, 0], [0, 1]), 1)
        self.assertEqual(js_divergence([1, 2], [1, 2]), 0)
        rows = [{'model': 'vicuna-13b-v1.5', 'dataset': 'no_robot', 'input_length': 120,
                 'output_length': 8, 'category': 'coding'},
                {'model': 'vicuna-13b-v1.5', 'dataset': 'no_robot', 'input_length': 240,
                 'output_length': 2}]
        result = distributions(rows)
        self.assertEqual(result['histograms'][0]['requests'], 2)
        self.assertEqual(set(result['categories']), {'coding'})
        self.assertEqual(set(result['input_length_bins']), {'100', '200'})
