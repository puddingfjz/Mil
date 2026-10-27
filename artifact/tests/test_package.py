import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from artifact.mil_artifact.package import SNAPSHOT_RECORD, assemble, verify_snapshot
from artifact.mil_artifact.config import artifact_root, load_yaml


class PackageTests(unittest.TestCase):
    def test_distribution_covers_current_runtime_and_generators(self):
        root = artifact_root()
        files = set(load_yaml(root / 'manifests/source_distribution.yaml')['files'])
        contract = json.loads((root / 'expected/cpu/source_contract.json').read_text())
        self.assertTrue({r['path'] for r in contract['files']} <= files)
        profiles = load_yaml(root / 'manifests/cost_profiles.yaml')['profiles']
        for profile in profiles.values():
            for row in profile.get('sources', []) + profile.get('generation_sources', []):
                relative = row['path'].replace('${MIL_REPO_ROOT}/', '')
                self.assertIn(relative, files)

    def test_explicit_files_only_and_no_directory_walk(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / 'repo'
            root.mkdir()
            (root / 'final.py').write_text('value = 1\n')
            (root / 'untracked.txt').write_text('not distributed')
            manifest = {'schema_version': 1, 'kind': 'source_distribution', 'files': ['final.py']}
            with patch.object(Path, 'rglob', side_effect=AssertionError('no tree walk')):
                report = assemble(root, manifest, Path(temporary) / 'out')
            self.assertEqual(list(report['files_sha256']), ['final.py'])
            with tarfile.open(Path(temporary) / 'out/mil-source.tar.gz') as stream:
                self.assertEqual(stream.getnames(), ['mil/final.py', 'mil/' + SNAPSHOT_RECORD])
                self.assertEqual(stream.extractfile('mil/final.py').read(), b'value = 1\n')
            self.assertTrue((root / 'untracked.txt').exists())
            with self.assertRaises(FileExistsError):
                assemble(root, manifest, Path(temporary) / 'out')

    def test_unpacked_source_passes_preflight_and_runner_without_git(self):
        import yaml
        from artifact.mil_artifact import doctor, run
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / 'repo'
            manifest_path = root / 'artifact/manifests/source_distribution.yaml'
            manifest_path.parent.mkdir(parents=True)
            (root / 'final.py').write_text('value = 1\n')
            manifest = {'schema_version': 1, 'kind': 'source_distribution',
                        'files': ['final.py', 'artifact/manifests/source_distribution.yaml']}
            manifest_path.write_text(yaml.safe_dump(manifest))
            assemble(root, manifest, Path(temporary) / 'out')
            source = Path(temporary) / 'out/source'
            self.assertFalse((source / '.git').exists())
            self.assertEqual(verify_snapshot(source)['files'], 2)
            with patch.object(doctor, 'repo_root', return_value=source), \
                    patch.object(run, 'repo_root', return_value=source), \
                    patch.object(doctor, '_command', side_effect=AssertionError('no Git needed')), \
                    patch.object(run, '_git_output', side_effect=AssertionError('no Git needed')):
                check, = list(doctor._source_checkout_checks())
                self.assertEqual(check.status, 'ok')
                self.assertFalse(run.git_metadata()['dirty'])
                (source / 'final.py').write_text('value = 2\n')
                self.assertEqual(list(doctor._source_checkout_checks())[0].status, 'error')
                self.assertTrue(run.git_metadata()['dirty'])

    def test_protected_missing_duplicate_and_symlink_inputs_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'link.py').symlink_to('/outside/source.py')
            for files in [[], ['.git_ori/file'], ['no_robot_copy/file'], ['../file'],
                          ['missing.py'], ['link.py'], ['x', 'x']]:
                with self.subTest(files=files), self.assertRaises((ValueError, FileNotFoundError)):
                    assemble(root, {'schema_version': 1, 'kind': 'source_distribution', 'files': files}, root / 'out')
                self.assertFalse((root / 'out').exists())

    def test_unpacked_source_success_survives_collection_and_verification(self):
        import yaml
        from artifact.mil_artifact import run
        from artifact.mil_artifact.config import expand_experiment
        from artifact.mil_artifact.schema import read_json
        from artifact.mil_artifact.verify import verify_run
        resolved = expand_experiment('smoke', workload_filter=['router'])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / 'repo'
            manifest_path = root / 'artifact/manifests/source_distribution.yaml'
            manifest_path.parent.mkdir(parents=True)
            manifest = {'schema_version': 1, 'kind': 'source_distribution',
                        'files': ['artifact/manifests/source_distribution.yaml']}
            manifest_path.write_text(yaml.safe_dump(manifest))
            assemble(root, manifest, Path(temporary) / 'package')
            source = Path(temporary) / 'package/source'
            results = Path(temporary) / 'results'
            with patch.object(run, 'repo_root', return_value=source):
                self.assertIn(source / SNAPSHOT_RECORD, run._artifact_input_sources(resolved))
            def success(command, cwd, environment, log, timeout):
                log.write('Total search time: 2\nTotal time for preparation before search: 1\n'
                          'Best group seq: [] 3 0\ntotal running time: 4s\n')
                return 0, None, False
            with patch.object(run, 'repo_root', return_value=source), \
                    patch.object(run, '_artifact_input_sources', return_value=[source / SNAPSHOT_RECORD]), \
                    patch.object(run, '_host_metadata', return_value={}), \
                    patch.object(run, '_stage_shared_resources', return_value=source), \
                    patch.object(run, '_stage_task_workdir', return_value=source), \
                    patch.object(run, '_run_scoped_process', side_effect=success):
                code = run.execute(resolved, results, None, 'python3', {'summary': {'error': 0}})
            self.assertEqual(code, 0)
            result = next(p for p in results.iterdir() if p.name != '.gpu-locks')
            self.assertEqual(read_json(result / 'run.json')['status'], 'succeeded')
            metric = read_json(result / 'tasks' / resolved.tasks[0].task_id / 'metrics.json')
            self.assertIsNone(metric['git_commit'])
            self.assertEqual(metric['source_snapshot_sha256'], verify_snapshot(source)['sha256'])
            evidence = result / 'inputs' / SNAPSHOT_RECORD
            self.assertTrue(evidence.is_file())
            evidence.write_text('{}')
            report = verify_run(result, artifact_root() / 'manifests/expected_results.yaml')
            self.assertGreater(report['issue_count'], 0)


if __name__ == '__main__':
    unittest.main()
