from __future__ import annotations

import copy
import importlib.machinery
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import yaml

from artifact.mil_artifact import runtime_identity as identity
from artifact.mil_artifact.config import repo_root
from artifact.mil_artifact.hashing import sha256_file
from artifact.mil_artifact.source_contract import fingerprint


class RuntimeIdentityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve() / 'source'
        for name in ('vllm', 'csrc', 'artifact/env', 'artifact/manifests', 'artifact/expected/cpu'):
            (self.root / name).mkdir(parents=True, exist_ok=True)
        (self.root / 'vllm/__init__.py').write_text('__version__ = "0.3.0"\n')
        (self.root / 'csrc/kernel.cu').write_text('int custom_kernel = 1;\n')
        runtime = {'schema_version': 1, 'python': '3.9.20', 'torch_cuda': '12.1', 'packages': {'torch': '2.1.2'},
                   'custom_vllm': {'implementation': 'mil_modified_source', 'installation': 'local_source_build',
                                   'package_version': '0.3.0', 'build_record': identity.BUILD_RECORD,
                                   'source_contract_manifest': identity.PREPARATION},
                   'container': {'build_parameters': {'torch_cuda_arch_list': '8.0'}}}
        (self.root / 'artifact/env/runtime.yaml').write_text(yaml.safe_dump(runtime))
        contract = {'schema_version': 1, 'kind': 'source_semantics', 'files': [
            {'path': 'vllm/__init__.py', 'comparison': 'python_ast_and_directives',
             'fingerprint': fingerprint((self.root / 'vllm/__init__.py').read_text())},
            {'path': 'csrc/kernel.cu', 'comparison': 'bytes',
             'sha256': sha256_file(self.root / 'csrc/kernel.cu')}]}
        contract_path = self.root / 'artifact/expected/cpu/source_contract.json'
        contract_path.write_text(json.dumps(contract))
        (self.root / identity.PREPARATION).write_text(yaml.safe_dump({
            'schema_version': 1,
            'source_contract': 'artifact/expected/cpu/source_contract.json',
            'source_contract_sha256': sha256_file(contract_path), 'source_files': 2}))
        self.extension = self.root / 'vllm' / ('_C' + importlib.machinery.EXTENSION_SUFFIXES[0])
        self.observed = {'origins': {'vllm': str(self.root / 'vllm/__init__.py'),
                                     'vllm._C': str(self.extension)},
                         'python': '3.9.20', 'python_abi': 'cpython-39', 'torch': '2.1.2',
                         'torch_build': '2.1.2+cu121', 'torch_cuda': '12.1', 'torch_cxx11_abi': False,
                         'vllm_distribution': '0.3.0', 'vllm_label': '0.3.0', 'cuda_initialized': False}
        self.compiler = {'cuda': '12.1', 'nvcc_version': 'fixture nvcc 12.1',
                         'cxx_version': 'fixture c++', 'cuda_arch_list': '8.0'}
        self.env = patch.dict(os.environ, {'VLLM_USE_PRECOMPILED': '', 'MIL_CPU_SCHEDULING': ''})
        self.env.start()
        self.addCleanup(self.env.stop)

    def build_fixture(self, action=None):
        def compile_fixture(*args, **kwargs):
            self.extension.write_bytes(b'fixture binary; NEVER imported')
            if action:
                action()
            return subprocess.CompletedProcess(args[0], 0)
        with patch.object(identity, 'toolchain', return_value=self.compiler), \
                patch.object(identity, 'probe', return_value=self.observed), \
                patch.object(identity.subprocess, 'run', side_effect=compile_fixture) as run:
            record = identity.build(self.root)
        self.assertIn('--no-index', record['command'])
        self.assertEqual(record['command'][-2:], ['--editable', '.'])
        self.assertEqual(run.call_args.kwargs['cwd'], self.root)
        self.assertEqual(run.call_args.kwargs['env']['CUDA_VISIBLE_DEVICES'], '')
        return record

    def test_local_build_record_passes_without_claiming_gpu_validation(self):
        record = self.build_fixture()
        with patch.object(identity, 'probe', return_value=self.observed):
            report = identity.verify(self.root)
        self.assertTrue(report['passed'])
        self.assertFalse(report['gpu_inference_validated'])
        self.assertFalse(record['gpu_inference_validated'])
        self.assertEqual(report['extensions'], ['vllm._C'])
        self.assertEqual(record['source']['files_sha256']['csrc/kernel.cu'],
                         sha256_file(self.root / 'csrc/kernel.cu'))

    def test_missing_record_cannot_pass_based_on_version(self):
        with patch.object(identity, 'probe') as probe, self.assertRaises(FileNotFoundError):
            identity.verify(self.root)
        probe.assert_not_called()

    def test_changed_binary_is_rejected_before_import(self):
        self.build_fixture()
        self.extension.write_bytes(b'replacement official binary')
        with patch.object(identity, 'probe') as probe, self.assertRaisesRegex(ValueError, 'checksum mismatch'):
            identity.verify(self.root)
        probe.assert_not_called()

    def test_changed_source_is_rejected_before_import(self):
        self.build_fixture()
        (self.root / 'csrc/kernel.cu').write_text('int custom_kernel = 2;\n')
        with patch.object(identity, 'probe') as probe, self.assertRaisesRegex(ValueError, 'source contract failed'):
            identity.verify(self.root)
        probe.assert_not_called()

    def test_new_optional_binary_is_rejected_before_import(self):
        self.build_fixture()
        extra = self.root / 'vllm' / ('_punica_C' + importlib.machinery.EXTENSION_SUFFIXES[0])
        extra.write_bytes(b'unrecorded native binary; NEVER imported')
        with patch.object(identity, 'probe') as probe, self.assertRaisesRegex(ValueError, 'unrecorded'):
            identity.verify(self.root)
        probe.assert_not_called()

    def test_comment_changes_also_require_a_new_exact_build_record(self):
        self.build_fixture()
        with (self.root / 'vllm/__init__.py').open('a') as stream:
            stream.write('# ordinary comment\n')
        with patch.object(identity, 'probe') as probe, self.assertRaisesRegex(ValueError, 'modified sources'):
            identity.verify(self.root)
        probe.assert_not_called()

    def test_foreign_origins_and_abi_changes_fail_with_same_version(self):
        self.build_fixture()
        for field in ('python_abi', 'torch_cxx11_abi', 'python_source', 'native_source', 'missing_native', 'cuda'):
            changed = copy.deepcopy(self.observed)
            if field == 'python_abi':
                changed[field] = 'cpython-310'
            elif field == 'torch_cxx11_abi':
                changed[field] = True
            elif field == 'python_source':
                changed['origins']['vllm'] = '/site-packages/vllm/__init__.py'
            elif field == 'native_source':
                changed['origins']['vllm._C'] = '/site-packages/vllm/_C.so'
            elif field == 'missing_native':
                changed['origins'].pop('vllm._C')
            else:
                changed['cuda_initialized'] = True
            with self.subTest(field=field), patch.object(identity, 'probe', return_value=changed), self.assertRaises(ValueError):
                identity.verify(self.root)

    def test_bad_records_fail_without_loading_extensions(self):
        record = self.build_fixture()
        target = self.root / identity.BUILD_RECORD
        for field in ('status', 'source_root', 'toolchain', 'extension_path'):
            changed = copy.deepcopy(record)
            if field == 'status':
                changed[field] = 'failed'
            elif field == 'source_root':
                changed[field] = '/another/source'
            elif field == 'toolchain':
                changed[field] = {}
            else:
                changed['runtime']['origins']['vllm._C'] = str(self.root / 'no_robot_copy/_C.so')
            target.write_text(json.dumps(changed))
            with self.subTest(field=field), patch.object(identity, 'probe') as probe, self.assertRaises(ValueError):
                identity.verify(self.root)
            probe.assert_not_called()

    def test_dirty_build_tree_is_not_overwritten(self):
        for relative in ('build', 'dist', 'vllm.egg-info', 'vllm/_C.so'):
            with self.subTest(relative=relative):
                target = self.root / relative
                target.write_bytes(b'old artifact')
                with patch.object(identity, 'toolchain') as compiler, self.assertRaisesRegex(ValueError, 'fresh source'):
                    identity.build(self.root)
                compiler.assert_not_called()
                self.assertEqual(target.read_bytes(), b'old artifact')
                target.unlink()

    def test_failed_install_or_changed_build_input_never_creates_success_record(self):
        with patch.object(identity, 'toolchain', return_value=self.compiler), \
                patch.object(identity.subprocess, 'run', side_effect=subprocess.CalledProcessError(1, ['fixture'])):
            with self.assertRaises(subprocess.CalledProcessError):
                identity.build(self.root)
        self.assertFalse((self.root / identity.BUILD_RECORD).exists())
        self.assertTrue((self.root / Path(identity.BUILD_RECORD).parent / 'failure.json').is_file())

    def test_changed_source_during_build_never_gets_attested(self):
        with self.assertRaisesRegex(ValueError, 'source contract failed'):
            self.build_fixture(lambda: (self.root / 'csrc/kernel.cu').write_text('changed'))
        self.assertFalse((self.root / identity.BUILD_RECORD).exists())

    def test_precompiled_and_cpu_shims_are_rejected(self):
        for name in ('VLLM_USE_PRECOMPILED', 'MIL_CPU_SCHEDULING'):
            with self.subTest(name=name), patch.dict(os.environ, {name: '1'}), self.assertRaises(ValueError):
                identity.build(self.root)

    def test_toolchain_requires_matching_cuda_and_explicit_architecture(self):
        runtime = identity.policy(self.root)
        variables = {'CUDA_HOME': str(self.root / 'toolkit'), 'TORCH_CUDA_ARCH_LIST': '8.0'}
        with patch.dict(os.environ, variables), patch.object(identity.shutil, 'which', return_value='/usr/bin/c++'), \
                patch.object(identity.subprocess, 'check_output', side_effect=['Cuda compilation tools, release 12.1', 'c++ fixture']):
            result = identity.toolchain(runtime)
        self.assertEqual(result['cuda'], '12.1')
        with patch.dict(os.environ, variables), \
                patch.object(identity.subprocess, 'check_output', return_value='Cuda compilation tools, release 12.4'), \
                self.assertRaisesRegex(ValueError, 'CUDA toolkit'):
            identity.toolchain(runtime)
        with patch.dict(os.environ, {**variables, 'TORCH_CUDA_ARCH_LIST': ''}), \
                patch.object(identity.shutil, 'which', return_value='/usr/bin/c++'), \
                patch.object(identity.subprocess, 'check_output', side_effect=['release 12.1', 'c++ fixture']), \
                self.assertRaisesRegex(ValueError, 'TORCH_CUDA_ARCH_LIST'):
            identity.toolchain(runtime)

    def test_native_origin_rejects_non_native_and_external_files_before_reading(self):
        for path in ('/site-packages/vllm/_C.so', str(self.root / 'vllm/_C.py'),
                     str(self.root / 'no_robot_copy/_C.so')):
            observed = copy.deepcopy(self.observed)
            observed['origins']['vllm._C'] = path
            with self.subTest(path=path), patch.object(identity, 'sha256_file') as digest, self.assertRaises(ValueError):
                identity.extension_identity(self.root, observed)
            digest.assert_not_called()

    def test_failed_post_build_probe_does_not_create_success_record(self):
        def fake_install(*args, **kwargs):
            self.extension.write_bytes(b'fixture only')
        with patch.object(identity, 'toolchain', return_value=self.compiler), \
                patch.object(identity.subprocess, 'run', side_effect=fake_install), \
                patch.object(identity, 'probe', side_effect=ValueError('foreign installed runtime')), \
                self.assertRaisesRegex(ValueError, 'foreign installed runtime'):
            identity.build(self.root)
        self.assertFalse((self.root / identity.BUILD_RECORD).exists())

    def test_child_environment_drops_shadow_paths(self):
        with patch.dict(os.environ, {'PYTHONPATH': '/foreign', 'PYTHONHOME': '/foreign',
                                    'MIL_CPU_SCHEDULING': '1', 'VLLM_USE_PRECOMPILED': '1'}):
            env = identity.child_environment()
        for name in ('PYTHONPATH', 'PYTHONHOME', 'MIL_CPU_SCHEDULING', 'VLLM_USE_PRECOMPILED'):
            self.assertNotIn(name, env)
        self.assertEqual(env['PYTHONNOUSERSITE'], '1')
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '')

    def test_nvcc_override_is_rejected_before_compiler_invocation(self):
        with patch.dict(os.environ, {'PYTORCH_NVCC': '/another/nvcc'}), \
                patch.object(identity.subprocess, 'check_output') as compiler, \
                self.assertRaisesRegex(ValueError, 'unset PYTORCH_NVCC'):
            identity.toolchain(identity.policy(self.root))
        compiler.assert_not_called()

    def test_probe_and_script_share_source_with_foreign_package_on_parent_path(self):
        # All fixture packages are inert Python; neither process imports torch/CUDA.
        foreign = self.root.parent / 'foreign/vllm'
        foreign.mkdir(parents=True)
        (foreign / '__init__.py').write_text('origin = "foreign"\n')
        benchmarks = self.root / 'benchmarks'
        benchmarks.mkdir()
        script = benchmarks / 'entry.py'
        script.write_text('import vllm; print(vllm.__file__)\n')
        module = self.root / 'artifact/mil_artifact'
        module.mkdir()
        (self.root / 'artifact/__init__.py').write_text('')
        (module / '__init__.py').write_text('')
        (module / 'runtime_identity.py').write_text(
            'def main(args):\n import vllm, json\n'
            ' print("MIL_RUNTIME_IDENTITY=" + json.dumps({"origin": vllm.__file__}))\n return 0\n')
        workdir = self.root.parent / 'work'
        workdir.mkdir()
        with patch.dict(os.environ, {'PYTHONPATH': str(foreign.parent), 'MIL_CPU_SCHEDULING': '1'}):
            observed = identity.probe(self.root)
            actual = subprocess.run([sys.executable, str(script)], cwd=workdir,
                                    env=identity.child_environment(self.root),
                                    capture_output=True, text=True, check=True)
        self.assertEqual(observed['origin'], str(self.root / 'vllm/__init__.py'))
        self.assertEqual(actual.stdout.strip(), observed['origin'])

    def test_help_has_no_gpu_framework_imports(self):
        code = '''import builtins, sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'vllm', 'ray'}:
        raise AssertionError('GPU framework import: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from artifact.mil_artifact.runtime_identity import main
raise SystemExit(main(['--help']))
'''
        result = subprocess.run([sys.executable, '-B', '-c', code], cwd=repo_root(),
                                capture_output=True, text=True, env=identity.child_environment())
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
