from __future__ import annotations

import ast
import unittest

import yaml

from artifact.mil_artifact.config import artifact_root, load_yaml, repo_root


class ContainerProvenanceTest(unittest.TestCase):
    def test_documented_cuda_build_version_matches_runtime_pin(self) -> None:
        # Evaluate the packaging version function without importing setup.py,
        # probing CUDA, compiling extensions, or consulting installed metadata.
        root = repo_root()
        tree = ast.parse((root / 'setup.py').read_text())
        function = next(node for node in tree.body
                        if isinstance(node, ast.FunctionDef) and node.name == 'get_vllm_version')
        main_cuda = next(ast.literal_eval(node.value) for node in tree.body
                         if isinstance(node, ast.Assign)
                         and any(isinstance(target, ast.Name) and target.id == 'MAIN_CUDA_VERSION'
                                 for target in node.targets))
        module = ast.Module(body=[function], type_ignores=[])
        namespace = {'MAIN_CUDA_VERSION': main_cuda,
                     'find_version': lambda path: next(ast.literal_eval(node.value)
                         for node in ast.parse(path.read_text()).body
                         if isinstance(node, ast.Assign)
                         and any(isinstance(target, ast.Name) and target.id == '__version__'
                                 for target in node.targets)),
                     'get_path': lambda *parts: root.joinpath(*parts),
                     '_is_hip': lambda: False, '_is_neuron': lambda: False}
        exec(compile(module, 'setup.py', 'exec'), namespace)
        runtime = load_yaml(artifact_root() / 'env/runtime.yaml')
        namespace['nvcc_cuda_version'] = runtime['torch_cuda']
        self.assertEqual(main_cuda, runtime['torch_cuda'])
        self.assertEqual(namespace['get_vllm_version'](), runtime['custom_vllm']['package_version'])
        namespace['nvcc_cuda_version'] = '12.4'
        self.assertNotEqual(namespace['get_vllm_version'](), runtime['custom_vllm']['package_version'])

    def test_custom_vllm_is_not_an_upstream_dependency(self) -> None:
        root = artifact_root()
        runtime = load_yaml(root / 'env/runtime.yaml')
        self.assertNotIn('vllm', runtime['packages'])
        self.assertEqual(runtime['custom_vllm']['implementation'], 'mil_modified_source')
        self.assertEqual(runtime['custom_vllm']['installation'], 'local_source_build')
        self.assertEqual(runtime['custom_vllm']['gpu_validation'], 'pending_build_and_inference_validation')
        for name in ('requirements-cpu.txt', 'runtime-freeze.txt', 'requirements-lock.txt'):
            self.assertFalse(any(line.lower().startswith('vllm==') for line in (root / 'env' / name).read_text().splitlines()))
        for name in ('README.md', 'Dockerfile'):
            text = (root / 'env' / name).read_text()
            self.assertIn('artifact.mil_artifact.runtime_identity build', text)

    def test_cpu_requirements_use_cpu_torch_and_captured_constraints(self) -> None:
        root = artifact_root()
        lines = (root / "env/requirements-cpu.txt").read_text().splitlines()
        self.assertIn("-c runtime-freeze.txt", lines)
        self.assertIn("-r requirements-lock.txt", lines)
        direct = dict(line.split("==", 1) for line in lines
                      if line and not line.startswith(("#", "-")))
        captured = dict(line.split("==", 1) for line in
                        (root / "env/runtime-freeze.txt").read_text().splitlines()
                        if line and not line.startswith("#"))
        self.assertEqual(direct["torch"], "2.1.2+cpu")
        cpu = load_yaml(root / "env/runtime.yaml")["cpu_reproduction"]
        self.assertEqual(cpu["torch"], direct["torch"])
        self.assertIsNone(cpu["torch_cuda"])
        self.assertFalse(cpu["gpu_extension_built"])
        for name, version in direct.items():
            self.assertFalse(name.lower().startswith("nvidia-"))
            self.assertNotIn(name.lower(), ("xformers", "triton", "vllm"))
            self.assertEqual(version.removesuffix("+cpu"), captured[name])
        distributed = load_yaml(root / "manifests/source_distribution.yaml")["files"]
        self.assertIn("artifact/env/requirements-cpu.txt", distributed)

    def test_tooling_environment_and_lock_match_runtime_versions(self) -> None:
        root = artifact_root()
        environment = yaml.safe_load(
            (root / "env" / "environment.yml").read_text(encoding="utf-8")
        )
        runtime = load_yaml(root / "env" / "runtime.yaml")
        pip_requirements = {
            requirement.split("==", 1)[0]: requirement.split("==", 1)[1]
            for requirement in environment["dependencies"][-1]["pip"]
        }
        lock_requirements = {
            requirement.split("==", 1)[0]: requirement.split("==", 1)[1]
            for requirement in (root / "env" / "requirements-lock.txt")
            .read_text(encoding="utf-8")
            .splitlines()
            if requirement and not requirement.startswith("#")
        }
        self.assertEqual(pip_requirements, lock_requirements)
        for distribution, version in lock_requirements.items():
            self.assertEqual(runtime["packages"][distribution], version)
        self.assertIn("python=3.9.20", environment["dependencies"])
        self.assertIn("pip=24.2", environment["dependencies"])

    def test_docker_base_matches_pinned_runtime_manifest(self) -> None:
        root = artifact_root()
        runtime = load_yaml(root / "env" / "runtime.yaml")
        container = runtime["container"]
        digest = container["linux_amd64_manifest_digest"]
        self.assertRegex(digest, r"^sha256:[0-9a-f]{64}$")
        dockerfile = (root / "env" / "Dockerfile").read_text(encoding="utf-8")
        expected = f"ARG BASE_IMAGE={container['base_image']}@{digest}"
        self.assertIn(expected, dockerfile)
        for name, version in container["build_parameters"].items():
            if name == "torch_cuda_arch_list":
                self.assertIn(f"TORCH_CUDA_ARCH_LIST={version}", dockerfile)
            else:
                self.assertIn(f"{name}={version}", dockerfile)


if __name__ == "__main__":
    unittest.main()
