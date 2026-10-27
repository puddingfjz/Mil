"""Configuration loading, validation, and legacy command expansion."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import re
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml


SCHEMA_VERSION = 1
ALLOWED_TEST_CASES = {"general", "map-reduce", "chain-summary", "router", "mixed"}

# Keep this order stable: commands are provenance records and should have small,
# reviewable diffs when a profile changes.
PARAM_FLAGS: dict[str, str] = {
    "ratio_seed": "--ratio-seed",
    "ratio_set": "--ratio-set",
    "reqnum": "--reqnum",
    "router_replicate_num": "--router_replicate_num",
    "router_question_version": "--router_question_version",
    "max_token_num": "--max_token_num",
    "specify_outlen": "--specify_outlen",
    "outlen_known": "--outlen_known",
    "noise_level": "--noise_level",
    "oracle_model_num": "--oracle_model_num",
    "no_runtime_adjustment": "--no_runtime_adjustment",
    "gpu_name": "--gpu_name",
    "byte_per_gpu": "--byte_per_gpu",
    "tot_gpu_num": "--tot_gpu_num",
    "max_group_seq_num": "--max_group_seq_num",
    "top_k": "--top_k",
    "similar_threshold": "--similar_threshold",
    "fully_connected_gpu_unit": "--fully_connected_gpu_unit",
    "machine_name": "--machine_name",
    "evaluator_num": "--evaluator_num",
    "summarize_model": "--summarize_model",
    "evaluator_model": "--evaluator_model",
    "reqnum_mixed_blender": "--reqnum_mixed_blender",
    "max_token_num_mixed_blender": "--max_token_num_mixed_blender",
    "model_replicate_num": "--model_replicate_num",
    "test_id": "--test_id",
}

BOOLEAN_PARAMS = {"specify_outlen", "outlen_known", "no_runtime_adjustment"}
ENVIRONMENT_PARAMS = {
    "sample_size": "SAMPLE_SIZE",
    "seed": "MIL_EXPERIMENT_SEED",
}
ARTIFACT_ONLY_PARAMS = {"repeat"} | set(ENVIRONMENT_PARAMS)
REQUIRED_PARAMS: dict[str, set[str]] = {
    "router": {"reqnum", "router_replicate_num", "router_question_version", "max_token_num"},
    "general": {"reqnum", "max_token_num"},
    "chain-summary": {
        "reqnum",
        "max_token_num",
        "evaluator_num",
        "summarize_model",
        "evaluator_model",
    },
    "mixed": {
        "reqnum",
        "max_token_num",
        "evaluator_num",
        "summarize_model",
        "evaluator_model",
        "reqnum_mixed_blender",
        "max_token_num_mixed_blender",
        "model_replicate_num",
    },
    "map-reduce": {"reqnum", "max_token_num"},
}

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_IDENTIFIER_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")


class ArtifactConfigError(ValueError):
    """Raised when an artifact configuration cannot be validated."""


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def artifact_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _require_mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ArtifactConfigError(f"{context} must be a mapping")
    return value


def _require_identifier(value: Any, context: str) -> str:
    if (
        not isinstance(value, str)
        or not _IDENTIFIER_PATTERN.fullmatch(value)
        or value in {".", ".."}
    ):
        raise ArtifactConfigError(
            f"{context} must be a non-empty path-safe identifier using only "
            "letters, digits, '.', '_' or '-'"
        )
    return value


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = yaml.safe_load(stream)
    except FileNotFoundError as exc:
        raise ArtifactConfigError(f"configuration file does not exist: {path}") from exc
    except yaml.YAMLError as exc:
        raise ArtifactConfigError(f"invalid YAML in {path}: {exc}") from exc

    result = _require_mapping(value, str(path))
    version = result.get("schema_version")
    if version != SCHEMA_VERSION:
        raise ArtifactConfigError(
            f"{path}: schema_version must be {SCHEMA_VERSION}, got {version!r}"
        )
    return result


def dump_yaml(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(dict(value), stream, sort_keys=False, allow_unicode=True)
    temporary.replace(path)


def resolve_profile_path(profile: str | os.PathLike[str]) -> Path:
    requested = Path(profile)
    if requested.is_file():
        return requested.resolve()
    if requested.suffix:
        raise ArtifactConfigError(f"experiment profile does not exist: {requested}")
    return artifact_root() / "configs" / "experiments" / f"{profile}.yaml"


def load_profile(profile: str | os.PathLike[str]) -> tuple[Path, dict[str, Any]]:
    path = resolve_profile_path(profile)
    config = load_yaml(path)
    if "paper_catalog" in config:
        from .paper_suite import compile_suite
        catalog_path = (path.parent / config["paper_catalog"]).resolve()
        _, config, _, _ = compile_suite(catalog_path, resolve_tasks=False)
    _require_identifier(config.get("name"), f"{path}: name")
    _require_identifier(config.get("hardware"), f"{path}: hardware")
    _require_identifier(
        config.get("execution_protocol"), f"{path}: execution_protocol"
    )
    timeout = config.get("task_timeout_seconds")
    if timeout is not None and (
        not isinstance(timeout, (int, float))
        or isinstance(timeout, bool)
        or not math.isfinite(float(timeout))
        or timeout <= 0
    ):
        raise ArtifactConfigError(
            f"{path}: task_timeout_seconds must be null or a positive number"
        )
    workloads = config.get("workloads")
    if not isinstance(workloads, list) or not workloads:
        raise ArtifactConfigError(f"{path}: workloads must be a non-empty list")
    return path, config


def load_execution_protocol() -> tuple[Path, dict[str, Any]]:
    path = artifact_root() / "manifests" / "execution_protocol.yaml"
    config = load_yaml(path)
    _require_identifier(config.get("protocol_id"), f"{path}: protocol_id")
    return path, config


def load_hardware(name: str) -> tuple[Path, dict[str, Any]]:
    _require_identifier(name, "hardware profile name")
    path = artifact_root() / "configs" / "hardware" / f"{name}.yaml"
    config = load_yaml(path)
    if config.get("name") != name:
        raise ArtifactConfigError(f"{path}: name does not match requested profile {name!r}")
    gpu = _require_mapping(config.get("gpu"), f"{path}: gpu")
    if not isinstance(gpu.get("count"), int) or gpu["count"] <= 0:
        raise ArtifactConfigError(f"{path}: gpu.count must be a positive integer")
    devices = gpu.get("visible_devices")
    if not isinstance(devices, list) or len(devices) != gpu["count"]:
        raise ArtifactConfigError(
            f"{path}: gpu.visible_devices must contain exactly gpu.count entries"
        )
    return path, config


def load_methods() -> tuple[Path, dict[str, dict[str, Any]]]:
    path = artifact_root() / "configs" / "methods.yaml"
    config = load_yaml(path)
    methods = _require_mapping(config.get("methods"), f"{path}: methods")
    for method_id, value in methods.items():
        _require_identifier(method_id, f"{path}: method id")
        method = _require_mapping(value, f"{path}: methods.{method_id}")
        legacy = _require_mapping(method.get("legacy"), f"{path}: methods.{method_id}.legacy")
        for key in ("gen_execplans_baseline", "search_method_baseline"):
            if not isinstance(legacy.get(key), str) or not legacy[key]:
                raise ArtifactConfigError(f"{path}: methods.{method_id}.legacy.{key} is required")
        if not isinstance(method.get("executable", True), bool):
            raise ArtifactConfigError(f"{path}: methods.{method_id}.executable must be boolean")
    return path, methods


def _matrix_values(value: Any) -> list[Any]:
    return value if isinstance(value, list) else [value]


def _matrix_product(matrix: Mapping[str, Any]) -> Iterable[dict[str, Any]]:
    unknown = sorted(set(matrix) - set(PARAM_FLAGS) - ARTIFACT_ONLY_PARAMS)
    if unknown:
        raise ArtifactConfigError(f"unknown legacy parameters in matrix: {', '.join(unknown)}")
    keys = list(matrix)
    value_lists = [_matrix_values(matrix[key]) for key in keys]
    if any(not values for values in value_lists):
        raise ArtifactConfigError("matrix values cannot be empty lists")
    if not keys:
        yield {}
        return
    for values in itertools.product(*value_lists):
        yield dict(zip(keys, values))


def _stable_hash(value: Mapping[str, Any], length: int = 10) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:length]


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    profile: str
    execution_protocol: str
    workload_id: str
    test_case: str
    method_id: str
    paper_label: str | None
    mapping_status: str
    executable: bool
    blocked_reason: str | None
    comparison_to_control: dict[str, Any] | None
    parameters: dict[str, Any]
    legacy: dict[str, str]
    gpu_devices: tuple[int, ...]

    def command(
        self,
        repository: Path | None = None,
        python_executable: str | os.PathLike[str] = sys.executable,
    ) -> list[str]:
        root = (repository or repo_root()).resolve()
        command = [
            os.fspath(python_executable),
            os.fspath(root / "benchmarks" / "schedule_multi_model.py"),
            "--gen-execplans-baseline",
            self.legacy["gen_execplans_baseline"],
            "--search_method_baseline",
            self.legacy["search_method_baseline"],
            "--test-case",
            self.test_case,
        ]
        for name, flag in PARAM_FLAGS.items():
            if name not in self.parameters or self.parameters[name] is None:
                continue
            value = self.parameters[name]
            if name in BOOLEAN_PARAMS:
                if not isinstance(value, bool):
                    raise ArtifactConfigError(
                        f"{self.task_id}: boolean parameter {name} has value {value!r}"
                    )
                if value:
                    command.append(flag)
            else:
                if isinstance(value, (dict, list)):
                    raise ArtifactConfigError(
                        f"{self.task_id}: scalar parameter {name} has value {value!r}"
                    )
                command.extend([flag, str(value)])
        return command

    def environment(self) -> dict[str, str]:
        environment = {
            "CUDA_VISIBLE_DEVICES": ",".join(str(device) for device in self.gpu_devices)
        }
        for name, variable in ENVIRONMENT_PARAMS.items():
            if name not in self.parameters or self.parameters[name] is None:
                continue
            value = self.parameters[name]
            if isinstance(value, (dict, list)):
                raise ArtifactConfigError(
                    f"{self.task_id}: scalar environment parameter {name} "
                    f"has value {value!r}"
                )
            environment[variable] = str(value)
        if "seed" in self.parameters and self.parameters["seed"] is not None:
            environment["PYTHONHASHSEED"] = str(self.parameters["seed"])
        return environment

    def display_command(
        self,
        repository: Path | None = None,
        python_executable: str | os.PathLike[str] = "python3",
    ) -> str:
        environment = " ".join(
            f"{name}={shlex.quote(value)}" for name, value in self.environment().items()
        )
        command = shlex.join(
            self.command(repository=repository, python_executable=python_executable)
        )
        return f"{environment} {command}"

    def to_dict(
        self,
        repository: Path | None = None,
        python_executable: str | os.PathLike[str] = sys.executable,
    ) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "profile": self.profile,
            "execution_protocol": self.execution_protocol,
            "workload_id": self.workload_id,
            "test_case": self.test_case,
            "method_id": self.method_id,
            "paper_label": self.paper_label,
            "mapping_status": self.mapping_status,
            "executable": self.executable,
            "blocked_reason": self.blocked_reason,
            "comparison_to_control": self.comparison_to_control,
            "parameters": self.parameters,
            "legacy": self.legacy,
            "gpu_devices": list(self.gpu_devices),
            "environment": self.environment(),
            "command": self.command(
                repository=repository,
                python_executable=python_executable,
            ),
            "display_command": self.display_command(
                repository=repository,
                python_executable=python_executable,
            ),
        }


@dataclass(frozen=True)
class ResolvedExperiment:
    profile_path: Path
    hardware_path: Path
    methods_path: Path
    execution_protocol_path: Path
    profile: dict[str, Any]
    hardware: dict[str, Any]
    execution_protocol: dict[str, Any]
    tasks: tuple[TaskSpec, ...]

    @property
    def provisional(self) -> bool:
        return bool(self.profile.get("provisional") or self.hardware.get("provisional"))


def expand_experiment(
    profile_name: str | os.PathLike[str],
    workload_filter: Sequence[str] = (),
    method_filter: Sequence[str] = (),
    parameter_filter: Mapping[str, Any] | None = None,
) -> ResolvedExperiment:
    profile_path, profile = load_profile(profile_name)
    hardware_path, hardware = load_hardware(profile["hardware"])
    methods_path, methods = load_methods()
    execution_protocol_path, execution_protocol = load_execution_protocol()
    if profile["execution_protocol"] != execution_protocol["protocol_id"]:
        raise ArtifactConfigError(
            f"{profile_path}: execution_protocol {profile['execution_protocol']!r} "
            f"does not match {execution_protocol_path}: "
            f"{execution_protocol['protocol_id']!r}"
        )

    defaults = _require_mapping(profile.get("defaults", {}), f"{profile_path}: defaults")
    unknown_defaults = sorted(set(defaults) - set(PARAM_FLAGS) - ARTIFACT_ONLY_PARAMS)
    if unknown_defaults:
        raise ArtifactConfigError(
            f"{profile_path}: unknown default parameters: {', '.join(unknown_defaults)}"
        )

    gpu = hardware["gpu"]
    legacy_hardware = _require_mapping(hardware.get("legacy", {}), f"{hardware_path}: legacy")
    injected = {
        "gpu_name": gpu["expected_name"],
        "byte_per_gpu": gpu["memory_bytes"],
        "tot_gpu_num": gpu["count"],
        "fully_connected_gpu_unit": gpu["fully_connected_gpu_unit"],
        "machine_name": legacy_hardware["machine_name"],
    }
    devices = tuple(int(device) for device in gpu["visible_devices"])

    selected_workloads = set(workload_filter)
    selected_methods = set(method_filter)
    selected_parameters = dict(parameter_filter or {})
    unknown_parameter_filters = sorted(
        set(selected_parameters) - set(PARAM_FLAGS) - ARTIFACT_ONLY_PARAMS
    )
    if unknown_parameter_filters:
        raise ArtifactConfigError(
            "unknown parameter filter(s): " + ", ".join(unknown_parameter_filters)
        )
    seen_workloads: set[str] = set()
    seen_methods: set[str] = set()
    tasks: list[TaskSpec] = []
    task_ids: set[str] = set()

    for raw_workload in profile["workloads"]:
        workload = _require_mapping(raw_workload, f"{profile_path}: workload")
        workload_id = workload.get("id")
        test_case = workload.get("test_case")
        _require_identifier(workload_id, f"{profile_path}: workload id")
        if workload_id in seen_workloads:
            raise ArtifactConfigError(f"{profile_path}: duplicate workload id {workload_id!r}")
        seen_workloads.add(workload_id)
        if selected_workloads and workload_id not in selected_workloads:
            continue
        if test_case not in ALLOWED_TEST_CASES:
            raise ArtifactConfigError(f"{profile_path}: invalid test_case {test_case!r}")

        method_ids = workload.get("methods")
        if not isinstance(method_ids, list) or not method_ids:
            raise ArtifactConfigError(f"{profile_path}: workload {workload_id} requires methods")
        matrix = _require_mapping(workload.get("matrix", {}), f"{profile_path}: {workload_id}.matrix")
        control_parameters = _require_mapping(
            workload.get("control_parameters", {}),
            f"{profile_path}: {workload_id}.control_parameters",
        )
        unknown_control_parameters = sorted(
            set(control_parameters) - set(PARAM_FLAGS) - ARTIFACT_ONLY_PARAMS
        )
        if unknown_control_parameters:
            raise ArtifactConfigError(
                f"{profile_path}: {workload_id} has unknown control parameters: "
                + ", ".join(unknown_control_parameters)
            )

        for method_id in method_ids:
            if not isinstance(method_id, str) or method_id not in methods:
                raise ArtifactConfigError(
                    f"{profile_path}: workload {workload_id} references unknown method {method_id!r}"
                )
            if selected_methods and method_id not in selected_methods:
                continue
            seen_methods.add(method_id)
            method = methods[method_id]
            control_method_id = method.get("control_method")
            if control_method_id is not None:
                if not isinstance(control_method_id, str) or control_method_id not in methods:
                    raise ArtifactConfigError(
                        f"{methods_path}: methods.{method_id}.control_method references "
                        f"unknown method {control_method_id!r}"
                    )
                if control_method_id not in method_ids:
                    raise ArtifactConfigError(
                        f"{profile_path}: workload {workload_id} must include control method "
                        f"{control_method_id!r} for variant {method_id!r}"
                    )
            method_args = _require_mapping(
                method.get("args", {}), f"{methods_path}: methods.{method_id}.args"
            )
            unknown_method_args = sorted(
                set(method_args) - set(PARAM_FLAGS) - ARTIFACT_ONLY_PARAMS
            )
            if unknown_method_args:
                raise ArtifactConfigError(
                    f"{methods_path}: method {method_id} has unknown args: "
                    + ", ".join(unknown_method_args)
                )

            method_matrix = _require_mapping(
                workload.get("method_matrix", {}).get(method_id, {}),
                f"{profile_path}: {workload_id}.method_matrix.{method_id}",
            )
            for matrix_values in _matrix_product({**matrix, **method_matrix}):
                parameters = dict(defaults)
                parameters.update(method_args)
                parameters.update(matrix_values)
                for name, value in injected.items():
                    parameters.setdefault(name, value)

                sample_size = parameters.get("sample_size")
                if sample_size is not None and (
                    not isinstance(sample_size, int)
                    or isinstance(sample_size, bool)
                    or sample_size <= 0
                ):
                    raise ArtifactConfigError(
                        f"{profile_path}: {workload_id}/{method_id} sample_size "
                        "must be a positive integer"
                    )

                seed = parameters.get("seed")
                if seed is not None and (
                    not isinstance(seed, int)
                    or isinstance(seed, bool)
                    or not 0 <= seed <= 2**32 - 1
                ):
                    raise ArtifactConfigError(
                        f"{profile_path}: {workload_id}/{method_id} seed must be "
                        "an integer between 0 and 4294967295"
                    )

                if any(
                    parameters.get(name) != expected
                    for name, expected in selected_parameters.items()
                ):
                    continue

                missing = sorted(
                    name
                    for name in REQUIRED_PARAMS[test_case]
                    if parameters.get(name) is None
                )
                if missing:
                    raise ArtifactConfigError(
                        f"{profile_path}: {workload_id}/{method_id} is missing: {', '.join(missing)}"
                    )

                control_parameters_resolved: dict[str, Any] | None = None
                control_legacy: dict[str, str] | None = None
                resolved_control_method_id: str | None = None
                if control_method_id is not None:
                    control = methods[control_method_id]
                    control_args = _require_mapping(
                        control.get("args", {}),
                        f"{methods_path}: methods.{control_method_id}.args",
                    )
                    control_parameters_resolved = dict(defaults)
                    control_parameters_resolved.update(control_args)
                    control_parameters_resolved.update(matrix_values)
                    for name, value in injected.items():
                        control_parameters_resolved.setdefault(name, value)
                    resolved_control_method_id = control_method_id
                    control_legacy = dict(control["legacy"])
                elif control_parameters and any(
                    parameters.get(name) != value
                    for name, value in control_parameters.items()
                ):
                    control_parameters_resolved = dict(parameters)
                    control_parameters_resolved.update(control_parameters)
                    resolved_control_method_id = method_id
                    control_legacy = dict(method["legacy"])

                comparison_to_control = None
                if (
                    control_parameters_resolved is not None
                    and control_legacy is not None
                    and resolved_control_method_id is not None
                ):
                    control_identity = {
                        "profile": profile["name"],
                        "workload_id": workload_id,
                        "method_id": resolved_control_method_id,
                        "test_case": test_case,
                        "parameters": control_parameters_resolved,
                    }
                    parameter_diff = {
                        name: {
                            "control": control_parameters_resolved.get(name),
                            "variant": parameters.get(name),
                        }
                        for name in sorted(set(control_parameters_resolved) | set(parameters))
                        if control_parameters_resolved.get(name) != parameters.get(name)
                    }
                    legacy_diff = {
                        name: {
                            "control": control_legacy.get(name),
                            "variant": method["legacy"].get(name),
                        }
                        for name in sorted(set(control_legacy) | set(method["legacy"]))
                        if control_legacy.get(name) != method["legacy"].get(name)
                    }
                    comparison_to_control = {
                        "control_task_id": (
                            f"{profile['name']}.{workload_id}.{resolved_control_method_id}."
                            f"{_stable_hash(control_identity)}"
                        ),
                        "control_method_id": resolved_control_method_id,
                        "parameter_diff": parameter_diff,
                        "legacy_diff": legacy_diff,
                    }

                identity = {
                    "profile": profile["name"],
                    "workload_id": workload_id,
                    "method_id": method_id,
                    "test_case": test_case,
                    "parameters": parameters,
                }
                task_id = f"{profile['name']}.{workload_id}.{method_id}.{_stable_hash(identity)}"
                if task_id in task_ids:
                    raise ArtifactConfigError(f"duplicate expanded task id: {task_id}")
                task_ids.add(task_id)
                tasks.append(
                    TaskSpec(
                        task_id=task_id,
                        profile=profile["name"],
                        execution_protocol=profile["execution_protocol"],
                        workload_id=workload_id,
                        test_case=test_case,
                        method_id=method_id,
                        paper_label=method.get("paper_label"),
                        mapping_status=str(method.get("mapping_status", "unknown")),
                        executable=bool(method.get("executable", True)),
                        blocked_reason=method.get("blocked_reason"),
                        comparison_to_control=comparison_to_control,
                        parameters=parameters,
                        legacy=dict(method["legacy"]),
                        gpu_devices=devices,
                    )
                )

    if selected_workloads:
        unknown = sorted(selected_workloads - seen_workloads)
        if unknown:
            raise ArtifactConfigError(f"unknown workload filter(s): {', '.join(unknown)}")
    if selected_methods:
        unknown = sorted(selected_methods - seen_methods)
        if unknown:
            raise ArtifactConfigError(
                "method filter(s) are not present in selected workload(s): " + ", ".join(unknown)
            )
    if not tasks:
        raise ArtifactConfigError("filters selected zero tasks")

    return ResolvedExperiment(
        profile_path=profile_path,
        hardware_path=hardware_path,
        methods_path=methods_path,
        execution_protocol_path=execution_protocol_path,
        profile=profile,
        hardware=hardware,
        execution_protocol=execution_protocol,
        tasks=tuple(tasks),
    )


def parse_parameter_filters(expressions: Sequence[str]) -> dict[str, Any]:
    """Parse repeatable NAME=YAML_VALUE command-line filters."""

    parsed: dict[str, Any] = {}
    for expression in expressions:
        name, separator, raw_value = expression.partition("=")
        if not separator or not name or not raw_value:
            raise ArtifactConfigError(
                f"parameter filter must use NAME=VALUE syntax, got {expression!r}"
            )
        if name in parsed:
            raise ArtifactConfigError(f"duplicate parameter filter: {name}")
        try:
            value = yaml.safe_load(raw_value)
        except yaml.YAMLError as exc:
            raise ArtifactConfigError(
                f"invalid value for parameter filter {name}: {exc}"
            ) from exc
        if isinstance(value, (dict, list)):
            raise ArtifactConfigError(
                f"parameter filter {name} must have a scalar value, got {value!r}"
            )
        parsed[name] = value
    return parsed


def expand_environment(value: Any, extra: Mapping[str, str] | None = None) -> Any:
    """Expand ${NAME} recursively while retaining unresolved placeholders."""

    environment = dict(os.environ)
    environment.setdefault("MIL_REPO_ROOT", os.fspath(repo_root()))
    if extra:
        environment.update(extra)

    if isinstance(value, str):
        return _ENV_PATTERN.sub(lambda match: environment.get(match.group(1), match.group(0)), value)
    if isinstance(value, list):
        return [expand_environment(item, environment) for item in value]
    if isinstance(value, dict):
        return {key: expand_environment(item, environment) for key, item in value.items()}
    return value


def has_unresolved_environment(value: str) -> bool:
    return bool(_ENV_PATTERN.search(value))
