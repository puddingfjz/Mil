from __future__ import annotations

import ast
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import tempfile
import subprocess
import sys
import importlib.util
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import yaml

from artifact.mil_artifact import doctor, model_runtime
from artifact.mil_artifact.config import repo_root
from artifact.mil_artifact.hashing import sha256_file


class ModelRuntimeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.revision = 'a' * 40
        self.snapshot = self.root / 'cache/models--org--model/snapshots' / self.revision
        self.snapshot.mkdir(parents=True)
        (self.snapshot / 'config.json').write_text('{"model_type":"llama"}')
        (self.snapshot / 'weights.bin').write_bytes(b'inert test fixture, not model weights')
        artifact = self.root / 'artifact'
        (artifact / 'manifests').mkdir(parents=True)
        self.manifest = {'schema_version': 1, 'models': [{'id': 'org/model', 'revision': self.revision,
                          'integrity': {'config_sha256': sha256_file(self.snapshot / 'config.json')}}]}
        (artifact / 'manifests/models.yaml').write_text(yaml.safe_dump(self.manifest))
        for context in (patch.object(model_runtime, 'artifact_root', return_value=artifact),
                        patch.object(doctor, 'artifact_root', return_value=artifact),
                        patch.dict(os.environ, {'MIL_MODEL_ROOT': str(self.root / 'cache'),
                                                'MIL_MODEL_SNAPSHOTS': ''})):
            context.start()
            self.addCleanup(context.stop)

    def test_preflight_snapshot_reaches_real_inference_entrypoint_without_download(self):
        check, = list(doctor._model_checks())
        self.assertEqual(check.status, 'ok')
        records = {'org/model': json.loads(check.detail)}
        tree = ast.parse((repo_root() / 'benchmarks/schedule_multi_model.py').read_text())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                        and node.name == 'start_a_model_inference_child_process')
        seen = []
        namespace = {'LLM_COMMUNICATOR': object, 'SHARED_CONTECT': SimpleNamespace(),
                     'seed_process': lambda **kwargs: None,
                     'InferenceArgs': lambda path, count: SimpleNamespace(model=path),
                     'benchmark_throughput': SimpleNamespace(main=lambda args: seen.append(args.model))}
        exec(compile(ast.Module(body=[function], type_ignores=[]), '<inference-entry>', 'exec'), namespace)
        with patch.dict(os.environ, {'MIL_MODEL_SNAPSHOTS': json.dumps(records),
                                     'MIL_MODEL_ROOT': str(self.root / 'wrong-cache')}), \
                patch('huggingface_hub.snapshot_download', side_effect=AssertionError('no downloads')), \
                redirect_stdout(io.StringIO()):
            namespace[function.name](None, True, '', 0, 'org/model', req_num=1)
        self.assertEqual(seen, [str(self.snapshot)])

    def test_scheduler_configuration_uses_the_same_pinned_snapshot(self):
        records = model_runtime.snapshot_records()
        tree = ast.parse((repo_root() / 'benchmarks/search_exec_plans.py').read_text())
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'MyModelInfor')
        function = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == 'set_hf_config')
        get_config = Mock(return_value=SimpleNamespace(num_hidden_layers=2))
        namespace = {'_hf_config_DICT': {}, 'get_config': get_config}
        exec(compile(ast.Module(body=[function], type_ignores=[]), '<config-entry>', 'exec'), namespace)
        model = SimpleNamespace(model_path='org/model', trust_remote_code=True, revision=None)
        with patch.dict(os.environ, {'MIL_MODEL_SNAPSHOTS': json.dumps(records)}):
            namespace[function.name](model)
        get_config.assert_called_once_with(str(self.snapshot), True, None)

    def test_missing_cache_is_error_not_warning(self):
        with patch.dict(os.environ, {'MIL_MODEL_ROOT': str(self.root / 'absent')}):
            checks = list(doctor._model_checks())
        self.assertEqual([check.status for check in checks], ['error'])

    def test_direct_resolved_path_must_belong_to_pinned_cache(self):
        self.assertEqual(model_runtime.resolve_model_snapshot(str(self.snapshot)), str(self.snapshot))
        with self.assertRaisesRegex(ValueError, 'not verified'):
            model_runtime.resolve_model_snapshot(str(self.root / 'unverified-model'))

    def test_changed_config_and_unselected_model_fail_without_fallback(self):
        records = model_runtime.snapshot_records()
        with patch.dict(os.environ, {'MIL_MODEL_SNAPSHOTS': json.dumps(records)}):
            with self.assertRaisesRegex(ValueError, 'not verified'):
                model_runtime.resolve_model_snapshot('other/model')
            (self.snapshot / 'config.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'changed'):
                model_runtime.resolve_model_snapshot('org/model')

    @unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('transformers'),
                         'requires optional CPU scheduling dependencies')
    def test_real_engine_configuration_uses_pinned_snapshot_without_cpu_config_replacement(self):
        config = {'model_type':'llama', 'architectures':['LlamaForCausalLM'],
                  'hidden_size':32, 'intermediate_size':64, 'num_hidden_layers':2,
                  'num_attention_heads':4, 'num_key_value_heads':4,
                  'max_position_embeddings':128, 'vocab_size':64, 'torch_dtype':'float16'}
        (self.snapshot / 'config.json').write_text(json.dumps(config))
        self.manifest['models'][0]['integrity']['config_sha256'] = sha256_file(self.snapshot / 'config.json')
        (self.root / 'artifact/manifests/models.yaml').write_text(yaml.safe_dump(self.manifest))
        records = model_runtime.snapshot_records()
        code = '''import ast, os, sys
from pathlib import Path
from unittest.mock import patch
root, snapshot = Path(sys.argv[1]), sys.argv[2]
sys.path.insert(0, str(root / 'benchmarks'))
from artifact.mil_artifact.cpu_guard import install
install()
import torch
import vllm.config as engine_config
from vllm.transformers_utils.config import get_config
tree = ast.parse((root / 'benchmarks/search_exec_plans.py').read_text())
function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'get_engin_args')
namespace = {'os':os}
exec(compile(ast.Module(body=[function], type_ignores=[]), '<real-engine-config>', 'exec'), namespace)
assert engine_config.get_config is get_config
model, *_ = namespace['get_engin_args']('org/model', 1)
assert model.model == snapshot, model.model
assert model.tokenizer == snapshot, model.tokenizer
assert model.hf_config.hidden_size == 32
model, *_ = namespace['get_engin_args'](snapshot, 1)
assert model.model == snapshot
assert not torch.cuda.is_initialized()
# The explicit CPU-only route must still work without a model cache.
frozen = get_config(snapshot, True)
with patch.dict(os.environ, {'MIL_CPU_SCHEDULING':'1', 'MIL_MODEL_SNAPSHOTS':'{}'}), \\
        patch('vllm.config.get_config', return_value=frozen) as read_frozen, \\
        patch('artifact.mil_artifact.model_runtime.resolve_model_snapshot', side_effect=AssertionError('CPU must use frozen config')):
    model, *_ = namespace['get_engin_args']('org/model', 1)
    assert model.model == 'org/model'
    read_frozen.assert_called_once_with('org/model', True, None)
assert not torch.cuda.is_initialized()
print('real engine config: pinned runtime and frozen CPU routes passed; CUDA not initialized')
'''
        env = {**os.environ, 'HF_HUB_OFFLINE':'1', 'TRANSFORMERS_OFFLINE':'1',
               'HF_HOME':str(self.root/'empty-cache'), 'HF_HUB_CACHE':str(self.root/'empty-cache/hub'),
               'CUDA_VISIBLE_DEVICES':'', 'PYTHONDONTWRITEBYTECODE':'1', 'PYTHONNOUSERSITE':'1',
               'MIL_MODEL_SNAPSHOTS':json.dumps(records)}
        env.pop('MIL_CPU_SCHEDULING', None)
        result = subprocess.run([sys.executable, '-B', '-c', code, str(repo_root()), str(self.snapshot)],
                                cwd=repo_root(), env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    @unittest.skipUnless(importlib.util.find_spec('transformers'), 'requires optional CPU scheduling dependencies')
    def test_real_tokenizer_and_runtime_templates_use_preflight_snapshot_offline(self):
        (self.snapshot / 'config.json').write_text('{"model_type":"bert"}')
        (self.snapshot / 'tokenizer_config.json').write_text('{"tokenizer_class":"BertTokenizer"}')
        (self.snapshot / 'vocab.txt').write_text('[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nhello\nworld\n')
        self.manifest['models'][0]['integrity']['config_sha256'] = sha256_file(self.snapshot / 'config.json')
        (self.root / 'artifact/manifests/models.yaml').write_text(yaml.safe_dump(self.manifest))
        check, = list(doctor._model_checks())
        self.assertEqual(check.status, 'ok')
        code = '''import ast, json, sys
from pathlib import Path
from types import SimpleNamespace
from typing import List
from transformers import AutoTokenizer
root = Path(sys.argv[1])
tree = ast.parse((root / 'benchmarks/schedule_multi_model.py').read_text())
nodes = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in {'InferenceArgs','get_token_num'}]
namespace = {'List':List, 'experiment_seed':lambda:0}
exec(compile(ast.Module(body=nodes, type_ignores=[]), '<real-tokenizer-path>', 'exec'), namespace)
lengths = namespace['get_token_num']('org/model', ['hello world'])
args = namespace['InferenceArgs']('org/model')
assert args.tokenizer == sys.argv[2], args.tokenizer
assert lengths == [4], lengths
assert namespace['InferenceArgs'](args.tokenizer).tokenizer == args.tokenizer
tree = ast.parse((root / 'vllm/core/multimodel_scheduler.py').read_text())
function = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == '_init_model_tokenizers')
runtime = {'AutoTokenizer':AutoTokenizer}
exec(compile(ast.Module(body=[function], type_ignores=[]), '<real-runtime-tokenizers>', 'exec'), runtime)
communicator = SimpleNamespace(prompt_template_args={0:(args.tokenizer, True)}, base_model_tokenizers={})
runtime[function.name](communicator)
assert len(communicator.base_model_tokenizers[0]('hello world').input_ids) == 4
print('preflight -> request tokenization -> runtime tokenizer: passed')
'''
        env = {**os.environ, 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
               'HF_HOME': str(self.root / 'empty-cache'), 'HF_HUB_CACHE': str(self.root / 'empty-cache/hub'),
               'CUDA_VISIBLE_DEVICES': '', 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONNOUSERSITE': '1',
               'MIL_MODEL_SNAPSHOTS': json.dumps({'org/model': json.loads(check.detail)})}
        result = subprocess.run([sys.executable, '-B', '-c', code, str(repo_root()), str(self.snapshot)],
                                cwd=repo_root(), env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_model_adapters_are_the_only_core_semantic_changes(self):
        from artifact.mil_artifact.source_contract import fingerprint
        records = json.loads((repo_root() / 'artifact/expected/cpu/source_contract.json').read_text())['files']
        replacements = {
            'benchmarks/schedule_multi_model.py': (
                'from artifact.mil_artifact.model_runtime import resolve_model_snapshot\n'
                '        local_model_path = resolve_model_snapshot(model)',
                'from huggingface_hub import snapshot_download\n'
                '        local_model_path = snapshot_download(model)'),
            'benchmarks/search_exec_plans.py': (
                'from artifact.mil_artifact.model_runtime import resolve_model_snapshot\n'
                '            hf_config = get_config(resolve_model_snapshot(self.model_path), self.trust_remote_code, self.revision)',
                'hf_config = get_config(*key)')}
        for name, (adapter, original) in replacements.items():
            source = (repo_root() / name).read_text()
            if name == 'benchmarks/schedule_multi_model.py':
                tokenizer_adapter = ('from artifact.mil_artifact.model_runtime import resolve_model_snapshot\n'
                                     '            self.tokenizer = resolve_model_snapshot(self.model)')
                self.assertEqual(source.count(tokenizer_adapter), 1)
                source = source.replace(tokenizer_adapter, 'self.tokenizer = self.model')
            if name == 'benchmarks/search_exec_plans.py':
                engine_adapter = ("    if os.environ.get('MIL_CPU_SCHEDULING') != '1':\n"
                                  '        from artifact.mil_artifact.model_runtime import resolve_model_snapshot\n'
                                  '        model = resolve_model_snapshot(model_path)\n')
                self.assertEqual(source.count(engine_adapter), 1)
                source = source.replace(engine_adapter, '')
            self.assertEqual(source.count(adapter), 1)
            record = next(row for row in records if row['path'] == name)
            self.assertEqual(fingerprint(source.replace(adapter, original)), record['pre_adapter_fingerprint'])


if __name__ == '__main__':
    unittest.main()
