"""Build and verify the Mil-local vLLM, never an upstream replacement wheel.

The build command is explicit and writes only in a fresh source distribution.
Verification checks local provenance, not GPU inference correctness or a signed
third-party build attestation. Neither command runs an inference experiment.
"""
from __future__ import annotations

import argparse
import ast
import importlib.machinery
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

from .config import load_yaml, repo_root
from .hashing import sha256_file
from .plot_entry import new_output_path, safe_path
from .schema import atomic_write_json
from .source_contract import source_path, verify_source_contract


BUILD_RECORD = 'artifact/results/runtime-build/build.json'
PREPARATION = 'artifact/manifests/cpu_preparation.yaml'
REQUIRED_MODULES = {'vllm', 'vllm._C'}
OPTIONAL_MODULES = {'vllm._punica_C'}


def policy(root):
    runtime = load_yaml(source_path(root, 'artifact/env/runtime.yaml'))
    custom = runtime.get('custom_vllm', {})
    if (not isinstance(custom, dict) or custom.get('implementation') != 'mil_modified_source'
            or custom.get('installation') != 'local_source_build'
            or custom.get('build_record') != BUILD_RECORD
            or custom.get('source_contract_manifest') != PREPARATION):
        raise ValueError('missing or invalid custom vLLM source policy')
    tree = ast.parse(source_path(root, 'vllm/__init__.py').read_text())
    label = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == '__version__' for t in n.targets))
    if label != custom.get('package_version'):
        raise ValueError('custom vLLM source label differs from the declared package version')
    return runtime


def source_identity(root):
    """Validate the release contract, then hash the exact build/runtime inputs."""
    root = safe_path(root)
    manifest = load_yaml(source_path(root, PREPARATION))
    contract_path = source_path(root, manifest['source_contract'])
    digest = sha256_file(contract_path)
    if digest != manifest['source_contract_sha256']:
        raise ValueError('source contract checksum mismatch')
    contract = json.loads(contract_path.read_text())
    report = verify_source_contract(root, contract)
    if not report['passed'] or report['checked_files'] != manifest['source_files']:
        raise ValueError('custom vLLM source contract failed: ' + str(report['issues']))
    return {'contract_sha256': digest,
            'files_sha256': {row['path']: sha256_file(source_path(root, row['path']))
                             for row in contract['files']}}


def child_environment(root=None):
    # The runtime build/probe must not inherit the CPU-only import shim or an
    # unrelated Python search path. Compilation targets are explicit, not probed.
    env = dict(os.environ)
    for name in ('PYTHONPATH', 'PYTHONHOME', 'MIL_CPU_SCHEDULING', 'VLLM_USE_PRECOMPILED',
                 'PYTORCH_NVCC', 'MIL_MODEL_SNAPSHOTS'):
        env.pop(name, None)
    env.update(PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='')
    if root is not None:
        env['PYTHONPATH'] = str(safe_path(root))
    return env


def native_origins(root):
    """Locate extensions without importing vLLM or executing a native loader."""
    selected = {}
    for name in sorted((REQUIRED_MODULES | OPTIONAL_MODULES) - {'vllm'}):
        extension = importlib.machinery.PathFinder.find_spec(name, [str(root / 'vllm')])
        if extension is None and name in OPTIONAL_MODULES:
            continue
        if extension is None or not isinstance(extension.loader, importlib.machinery.ExtensionFileLoader):
            raise ValueError('missing locally built native extension: ' + name)
        path = safe_path(extension.origin)
        if path.parent != root / 'vllm':
            raise ValueError('foreign native extension: ' + name)
        selected[name] = str(path)
    return selected


def _probe(root):
    """Child-only import probe. No operators, models, or CUDA initialization."""
    import importlib
    import importlib.metadata
    import importlib.util
    import platform

    spec = importlib.util.find_spec('vllm')
    expected = source_path(root, 'vllm/__init__.py')
    if spec is None or spec.origin != str(expected):
        raise ValueError('vLLM Python source is not this repository')
    selected = {'vllm': str(expected), **native_origins(root)}
    import torch
    origins = {}
    for name, expected_origin in selected.items():
        module = importlib.import_module(name)
        origin = str(safe_path(module.__file__))
        if origin != expected_origin:
            raise ValueError('loaded module origin changed: ' + name)
        origins[name] = origin
    if torch.cuda.is_initialized():
        raise ValueError('import probe unexpectedly initialized CUDA')
    return {'origins': origins, 'python': platform.python_version(),
            'python_abi': sys.implementation.cache_tag,
            'torch': importlib.metadata.version('torch'), 'torch_build': str(torch.__version__),
            'torch_cuda': torch.version.cuda,
            'torch_cxx11_abi': bool(torch._C._GLIBCXX_USE_CXX11_ABI),
            'vllm_distribution': importlib.metadata.version('vllm'),
            'vllm_label': sys.modules['vllm'].__version__, 'cuda_initialized': False}


def probe(root):
    # Match the script entrypoint's sys.path[0], not the repository-root -m path.
    code = ("import sys; sys.path[0] = sys.argv[1]; "
            "from artifact.mil_artifact.runtime_identity import main; "
            "raise SystemExit(main(['_probe']))")
    command = [sys.executable, '-B', '-c', code, str(root / 'benchmarks')]
    result = subprocess.run(command, cwd=root, env=child_environment(root),
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
    # Some legacy imports print diagnostics. Only the explicit final marker is JSON.
    marker = 'MIL_RUNTIME_IDENTITY='
    rows = [line[len(marker):] for line in result.stdout.splitlines() if line.startswith(marker)]
    if len(rows) != 1:
        raise ValueError('runtime probe did not return one identity record')
    return json.loads(rows[0])


def extension_identity(root, observed):
    origins = observed['origins']
    if (not isinstance(origins, dict) or not REQUIRED_MODULES.issubset(origins)
            or set(origins) - REQUIRED_MODULES - OPTIONAL_MODULES):
        raise ValueError('missing or unexpected runtime module origins')
    if origins['vllm'] != str(root / 'vllm/__init__.py'):
        raise ValueError('foreign vLLM Python source')
    extensions = {}
    for name, origin in origins.items():
        path = safe_path(origin)
        if name == 'vllm':
            continue
        base = name.rsplit('.', 1)[1]
        if (path.parent != root / 'vllm'
                or not any(path.name == base + suffix for suffix in importlib.machinery.EXTENSION_SUFFIXES)):
            raise ValueError('foreign or non-native extension: ' + name)
        extensions[name] = {'path': path.relative_to(root).as_posix(), 'sha256': sha256_file(path)}
    return extensions


def validate_observed(runtime, observed):
    if not isinstance(observed, dict):
        raise ValueError('invalid runtime identity record')
    expected = {'python': str(runtime['python']), 'torch': str(runtime['packages']['torch']),
                'torch_cuda': str(runtime['torch_cuda']),
                'vllm_distribution': runtime['custom_vllm']['package_version'],
                'vllm_label': runtime['custom_vllm']['package_version'], 'cuda_initialized': False}
    if any(observed.get(key) != value for key, value in expected.items()):
        raise ValueError('runtime versions or CUDA guard differ from the declared custom build')
    if not observed.get('python_abi') or type(observed.get('torch_cxx11_abi')) is not bool:
        raise ValueError('runtime ABI metadata is missing')
    if observed.get('torch_build', '').split('+', 1)[0] != expected['torch']:
        raise ValueError('loaded PyTorch build differs from its distribution metadata')


def toolchain(runtime):
    if os.environ.get('PYTORCH_NVCC'):
        raise ValueError('unset PYTORCH_NVCC; the build uses CUDA_HOME/bin/nvcc')
    cuda_home = os.environ.get('CUDA_HOME')
    if not cuda_home:
        raise ValueError('set CUDA_HOME to the documented CUDA toolkit before building')
    nvcc = safe_path(Path(cuda_home) / 'bin/nvcc')
    # CUDA_HOME may itself be a conventional symlink; choose its explicit real
    # installation path in the environment rather than following arbitrary links.
    output = subprocess.check_output([str(nvcc), '-V'], text=True)
    match = re.search(r'release\s+([0-9]+\.[0-9]+)', output)
    if match is None or match[1] != str(runtime['torch_cuda']):
        raise ValueError('nvcc must match the documented CUDA toolkit version')
    cxx = shutil.which(os.environ.get('CXX', 'c++'))
    if cxx is None:
        raise ValueError('C++ compiler not found')
    cxx_version = subprocess.check_output([cxx, '--version'], text=True)
    arch = os.environ.get('TORCH_CUDA_ARCH_LIST')
    if arch != str(runtime['container']['build_parameters']['torch_cuda_arch_list']):
        raise ValueError('set the declared TORCH_CUDA_ARCH_LIST before building')
    return {'nvcc': str(nvcc), 'nvcc_version': output.strip(),
            'cuda': match[1], 'cxx': cxx, 'cxx_version': cxx_version.strip(), 'cuda_arch_list': arch}


def assert_clean_build_tree(root):
    for name in ('build', 'dist', 'vllm.egg-info'):
        if source_path(root, name).exists():
            raise ValueError('use a fresh source distribution; old build metadata exists: ' + name)
    for path in source_path(root, 'vllm').iterdir():
        if path.name.endswith(('.so', '.pyd', '.dll')):
            raise ValueError('use a fresh source distribution; existing native binary: ' + path.name)
    if os.environ.get('VLLM_USE_PRECOMPILED') or os.environ.get('MIL_CPU_SCHEDULING'):
        raise ValueError('precompiled binaries and the CPU-only import shim are not build inputs')


def build(root):
    root = safe_path(root)
    runtime = policy(root)
    identity = source_identity(root)
    assert_clean_build_tree(root)
    output = new_output_path(root / Path(BUILD_RECORD).parent)
    compiler = toolchain(runtime)
    command = [sys.executable, '-m', 'pip', 'install', '--no-index', '--no-cache-dir',
               '--no-build-isolation', '--no-deps', '--editable', '.']
    output.mkdir(parents=True, exist_ok=False)
    try:
        with (output / 'build.log').open('x') as stream:
            subprocess.run(command, cwd=root, env=child_environment(), stdout=stream,
                           stderr=subprocess.STDOUT, check=True)
        if source_identity(root) != identity:
            raise ValueError('source changed during build')
        observed = probe(root)
        validate_observed(runtime, observed)
        record = {'schema_version': 1, 'kind': 'mil_local_source_build', 'status': 'built',
                  'source_root': str(root), 'source': identity, 'runtime': observed,
                  'extensions': extension_identity(root, observed), 'toolchain': compiler,
                  'command': command, 'exit_code': 0, 'gpu_inference_validated': False,
                  'scope': 'Local clean-build provenance; not a signed attestation or GPU inference validation.'}
        atomic_write_json(root / BUILD_RECORD, record)
        return record
    except Exception as error:
        atomic_write_json(output / 'failure.json', {'status': 'failed', 'error': str(error),
                                                  'gpu_inference_validated': False})
        raise


def verify(root):
    root = safe_path(root)
    runtime = policy(root)
    current_source = source_identity(root)
    record = json.loads(source_path(root, BUILD_RECORD).read_text())
    if (not isinstance(record, dict) or record.get('schema_version') != 1 or record.get('kind') != 'mil_local_source_build'
            or record.get('status') != 'built' or record.get('exit_code') != 0):
        raise ValueError('missing successful local source build record')
    if record.get('source_root') != str(root) or record.get('source') != current_source:
        raise ValueError('build belongs to different or modified sources; rebuild in a fresh source distribution')
    if (not isinstance(record.get('toolchain'), dict) or not record['toolchain'].get('nvcc_version')
            or not record['toolchain'].get('cxx_version')):
        raise ValueError('missing compiler metadata')
    if (record['toolchain'].get('cuda') != str(runtime['torch_cuda'])
            or record['toolchain'].get('cuda_arch_list') != str(runtime['container']['build_parameters']['torch_cuda_arch_list'])):
        raise ValueError('build used a different CUDA toolkit or architecture')
    # Hash recorded native files before loading them, so replacement binaries are
    # rejected before any import can execute their initialization code.
    old_runtime = record['runtime']
    validate_observed(runtime, old_runtime)
    extensions = extension_identity(root, old_runtime)
    if extensions != record.get('extensions'):
        raise ValueError('native extension checksum mismatch')
    on_disk = {'origins': {'vllm': str(root / 'vllm/__init__.py'), **native_origins(root)}}
    if extension_identity(root, on_disk) != extensions:
        raise ValueError('unrecorded or differently resolved native extension')
    current = probe(root)
    validate_observed(runtime, current)
    if current != old_runtime or extension_identity(root, current) != extensions:
        raise ValueError('loaded runtime origin, ABI, or native extension differs from the recorded build')
    return {'passed': True, 'implementation': 'mil_modified_source',
            'source_files': len(current_source['files_sha256']), 'extensions': sorted(extensions),
            'gpu_inference_validated': False, 'scope': 'Local provenance only; GPU validation remains separate.'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('build', 'verify', '_probe'))
    args = parser.parse_args(argv)
    try:
        root = safe_path(repo_root())
        if args.command == '_probe':
            print('MIL_RUNTIME_IDENTITY=' + json.dumps(_probe(root)))
        elif args.command == 'build':
            build(root)
            print(json.dumps({'status': 'built', 'record': str(root / BUILD_RECORD),
                              'gpu_inference_validated': False}))
        else:
            print(json.dumps(verify(root)))
    except (OSError, ValueError, KeyError, TypeError, StopIteration, subprocess.SubprocessError) as error:
        parser.exit(2, f'custom vLLM identity failed: {error}\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
