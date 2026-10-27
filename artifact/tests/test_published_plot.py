"""Paper-style rendering with generated, artificial test values only."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from artifact.mil_artifact.plot_entry import chart
from artifact.mil_artifact.published_plot import validate_chart
from artifact.tests.synthetic import chart_data
from artifact.mil_artifact.config import repo_root


class ChartTests(unittest.TestCase):
    def test_invalid_values_or_methods_fail_before_rendering(self):
        mutations = [
            lambda c: c['bars'][0]['runtime'].__setitem__(0, float('nan')),
            lambda c: c['bars'][0]['total'].__setitem__(0, -1),
            lambda c: c['bars'][1]['runtime'].pop(),
            lambda c: c['bars'][0].__setitem__('method_key', ['unknown', 1]),
            lambda c: c['min_r'][0]['statistics'].pop('tot_time_max'),
            lambda c: c['min_r'][0]['samples'].pop(),
        ]
        for mutate in mutations:
            data = chart_data()
            mutate(data)
            with self.subTest(mutation=mutate), self.assertRaises(ValueError):
                validate_chart(data)

    def test_external_chart_renders_without_gpu_imports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / 'chart.json'
            input_path.write_text(json.dumps({'schema_version': 1, 'figures': {'test': chart_data()}}))
            code = """import builtins, sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'ray', 'vllm', 'transformers', 'gurobipy'}:
        raise AssertionError('GPU dependency imported: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from artifact.mil_artifact.plot_entry import main
raise SystemExit(main(sys.argv[1:]))
"""
            output = root / 'charts'
            result = subprocess.run([sys.executable, '-c', code, 'chart', '--input', str(input_path),
                                     '--output', str(output)], cwd=repo_root(), capture_output=True, text=True,
                                    env={**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'PYTHONDONTWRITEBYTECODE': '1'})
            self.assertEqual(result.returncode, 0, result.stderr)
            drawn = json.loads((output / 'plot_data.json').read_text())['test']
            self.assertEqual(drawn['bars'], chart_data()['bars'])
            self.assertEqual(drawn['min_r'], chart_data()['min_r'])
            self.assertTrue((output / 'synthetic.png').is_file())
            self.assertFalse(json.loads((output / 'plot_manifest.json').read_text())['gpu_execution'])

    def test_bad_filename_or_selection_does_not_create_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = chart_data()
            data['png'] = '../escape.png'
            input_path = root / 'chart.json'
            input_path.write_text(json.dumps({'schema_version': 1, 'figures': {'test': data}}))
            with self.assertRaises(ValueError):
                chart(input_path, root / 'out')
            self.assertFalse((root / 'out').exists())
