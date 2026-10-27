import copy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from artifact.mil_artifact.config import artifact_root, load_yaml, repo_root
from artifact.mil_artifact.hashing import sha256_file
from artifact.mil_artifact.plot_entry import (
    historical, main, new_output_path, plotting_environment, chart, results, safe_path,
)


class OutputSafetyTest(unittest.TestCase):
    def test_backup_is_rejected_before_filesystem_inspection(self):
        with patch.object(Path, 'is_symlink', side_effect=AssertionError('must not inspect backup')):
            for path in ['/tmp/backup-test/new', '/tmp/vllm-latest-camera-ready/new',
                         '/tmp/repo/.git_ori/new', '/tmp/dataset_copy/new',
                         '/tmp/dataset_v2_copy/new', '/tmp/repo/.git/config',
                         '/tmp/DATASET_COPY/new']:
                with self.subTest(path=path), self.assertRaises(ValueError):
                    safe_path(path)

    def test_os_tmp_alias_cannot_hide_a_protected_target(self):
        with patch.object(Path, 'is_symlink', return_value=True) as inspect, \
                patch('os.readlink', return_value='/data/dataset_copy'):
            with self.assertRaises(ValueError):
                safe_path('/tmp/new')
        self.assertEqual(inspect.call_count, 1)  # Only the OS alias, never its target.

    def test_parent_traversal_is_rejected(self):
        with self.assertRaises(ValueError):
            safe_path('/tmp/a/../b')

    def test_source_and_evidence_directories_are_not_outputs(self):
        for path in ['benchmarks/figures/new', 'artifact/expected/new', 'artifact/evidence/new', '.git/new']:
            with self.subTest(path=path), self.assertRaises(ValueError):
                new_output_path(repo_root() / path)
        self.assertEqual(new_output_path(repo_root() / 'artifact/results/plot-entry-test-new'),
                         repo_root() / 'artifact/results/plot-entry-test-new')

    def test_existing_directory_and_file_remain_untouched(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            sentinel = output / 'old.txt'
            sentinel.write_bytes(b'keep')
            with patch('artifact.mil_artifact.plot_entry.json.loads') as inputs:
                for target in [output, sentinel]:
                    with self.assertRaises(FileExistsError):
                        chart(output / 'unused-input.json', target)
                inputs.assert_not_called()
            self.assertEqual(sentinel.read_bytes(), b'keep')

    def test_symlink_and_symlink_parent_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'link').symlink_to(root / 'missing', target_is_directory=True)
            for target in [root / 'link', root / 'link/new']:
                with self.assertRaises(ValueError):
                    new_output_path(target)

    def test_symlink_parent_rejected_before_inspecting_child(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            link = root / 'alias'
            link.symlink_to(root / 'missing', target_is_directory=True)
            original = Path.is_symlink
            def inspect(path):
                if link in path.parents:
                    raise AssertionError('must not inspect through a symlinked parent')
                return original(path)
            with patch.object(Path, 'is_symlink', inspect), self.assertRaises(ValueError):
                safe_path(link / 'nested/child')

    def test_cache_environment_is_restored(self):
        with patch.dict(os.environ, {'MPLCONFIGDIR': '/tmp/original-cache'}):
            with plotting_environment():
                temporary = Path(os.environ['MPLCONFIGDIR'])
                self.assertTrue(temporary.is_dir())
                self.assertNotEqual(str(temporary), '/tmp/original-cache')
            self.assertFalse(temporary.exists())
            self.assertEqual(os.environ['MPLCONFIGDIR'], '/tmp/original-cache')


class ExplicitInputsTest(unittest.TestCase):
    def test_all_modes_require_an_input(self):
        for mode in ('results', 'chart', 'candidates', 'supporting'):
            with self.subTest(mode=mode), self.assertRaises(SystemExit) as error:
                main([mode, '--output', '/tmp/not-created'])
            self.assertNotEqual(error.exception.code, 0)

    def test_result_mode_uses_supplied_metrics(self):
        from artifact.tests.test_plot import _record
        from artifact.mil_artifact.schema import write_jsonl
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / 'run'
            run.mkdir()
            write_jsonl(run / 'metrics.jsonl', [_record('mil', 1, 10., 12.)])
            report = results(run, root / 'plots', ['router_unknown_output_length'])
            self.assertTrue(report['passed'])
            self.assertEqual(len(report['files']), 1)
            self.assertEqual(report['input_sha256'], sha256_file(run / 'metrics.jsonl'))

    def test_missing_input_has_no_cached_result_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(FileNotFoundError):
                results(root / 'missing', root / 'out')
            self.assertFalse((root / 'out').exists())

    def test_analysis_bundle_is_explicit_and_preserves_its_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = root / 'snapshot'
            snapshot.mkdir()
            (snapshot / 'results.json').write_text('{}')
            def fake_plot(source, destination):
                destination.mkdir()
                (destination / 'plot_manifest.json').write_text('{"synthetic": true}')
            with patch('artifact.mil_artifact.paper_results.plot', side_effect=fake_plot):
                report = historical('candidates', snapshot, root / 'out')
            self.assertEqual(report['mode'], 'candidates')
            self.assertTrue(json.loads((root / 'out/plot_manifest.json').read_text())['synthetic'])


if __name__ == '__main__':
    unittest.main()
