"""Freeze model configuration metadata for weight-free CPU scheduling."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from .config import artifact_root, load_yaml
from .schema import atomic_write_json


def load_configurations(path):
    from transformers import AutoConfig
    from vllm.transformers_utils.config import _CONFIG_REGISTRY
    entries = json.loads(path.read_text())['models']
    result = {}
    for model, entry in entries.items():
        values = dict(entry['configuration'])
        model_type = values.pop('model_type')
        if model_type in _CONFIG_REGISTRY:
            config = _CONFIG_REGISTRY[model_type](**values)
        else:
            config = AutoConfig.for_model(model_type, **values)
        result[(model, True, None)] = config
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    from .cpu_schedule import cpu_environment
    cpu_environment(0)
    from vllm.transformers_utils.config import get_config
    manifest = load_yaml(artifact_root() / 'manifests/models.yaml')
    models = {}
    for entry in manifest['models']:
        config = get_config(entry['id'], True, entry['revision'])
        models[entry['id']] = {'revision': entry['revision'],
                              'source_config_sha256': entry['integrity']['config_sha256'],
                              'configuration': config.to_dict()}
    path = args.output / 'model_configurations.json'
    atomic_write_json(path, {'schema_version': 1, 'models': models,
                            'scope': 'Configuration metadata only; no weights, tokenizer vocabulary, or remote model code.'})
    restored = load_configurations(path)
    for model, entry in models.items():
        original = dict(entry['configuration'])
        actual = restored[(model, True, None)].to_dict()
        # Transformers records local runtime class/version metadata separately.
        for key in ('transformers_version', '_commit_hash', 'model_type'):
            original.pop(key, None)
            actual.pop(key, None)
        if original != actual:
            raise ValueError(f'configuration roundtrip differs: {model}')
    print(f'froze and roundtrip-verified {len(models)} model configurations')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
