"""Compile the paper inventory into deduplicated scheduling tasks and analyses."""
from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
from pathlib import Path
import tempfile

import yaml

from .config import artifact_root, expand_experiment, load_yaml


MAIN_ITEMS = [f"figure_6{letter}" for letter in "abcdef"] + ["figure_7", "figure_8"]
REQUIRED_ITEMS = set(MAIN_ITEMS + ["figure_9", "table_1", "table_2", "table_3", "table_4",
                                  "runtime_adjustment", "search_efficiency", "estimation_error",
                                  "figure_3ab", "figure_3c_jsd", "figure_3d", "figure_4"])


def product(matrix):
    keys = list(matrix)
    if any(not isinstance(matrix[k], list) or not matrix[k] for k in keys):
        raise ValueError("matrix entries must be nonempty lists")
    return [dict(zip(keys, row)) for row in itertools.product(*(matrix[k] for k in keys))]


def compile_suite(path: Path | None = None, resolve_tasks=True):
    catalog = load_yaml(path or artifact_root() / "configs/paper_experiments.yaml")
    if set(catalog["items"]) != REQUIRED_ITEMS:
        raise ValueError(f"paper inventory differs: {set(catalog['items']) ^ REQUIRED_ITEMS}")
    cases, membership = {}, {}

    def register(workload, overrides, methods, model_set=None, variant=None):
        definition = catalog["workloads"][workload]
        params = {**definition["fixed"], **overrides}
        normalized = {**catalog["defaults"], **params}
        identity = json.dumps([workload, normalized], sort_keys=True, separators=(",", ":"))
        case_id = workload + "_" + hashlib.sha256(identity.encode()).hexdigest()[:10]
        models = model_set or definition["model_set"]
        paths = [p for group in ([models] if isinstance(models, str) else models)
                 for p in catalog["model_sets"][group]]
        paths *= params.get("model_replicate_num", 1)
        actual_methods = [definition.get("mil_method", "mil") if m == "mil" else m for m in methods]
        entry = cases.setdefault(case_id, {"workload": workload, "parameters": params,
                                           "test_case": definition["test_case"], "methods": [],
                                           "model_paths": paths, "dataset": definition["dataset"]})
        if entry["model_paths"] != paths:
            raise ValueError("same parameters select inconsistent model sets")
        for method in actual_methods:
            if method not in entry["methods"]:
                entry["methods"].append(method)
        return {"case_id": case_id, "methods": actual_methods, "variant": variant}

    for item_id, item in catalog["items"].items():
        members = []
        if item["kind"] == "scheduling":
            coordinates = item.get("cases") or [
                {"workload": item["workload"], "parameters": p}
                for p in product(item.get("matrix", {}))]
            for coordinate in coordinates:
                for variant in item.get("variants", [{"parameters": {}}]):
                    members.append(register(coordinate["workload"],
                                            {**coordinate["parameters"], **variant["parameters"]},
                                            variant.get("methods", item.get("methods", catalog["methods"])),
                                            item.get("model_set"), variant.get("label")))
        elif "reuse" in item:
            seen = set()
            for source in item["reuse"]:
                for member in membership[source]:
                    old = cases[member["case_id"]]
                    if any(old["parameters"].get(k, catalog["defaults"].get(k)) != v
                           for k, v in item.get("filter", {}).items()):
                        continue
                    for extra in product(item.get("matrix", {})):
                        methods = item.get("methods", member["methods"])
                        new = register(old["workload"], {**old["parameters"], **extra}, methods)
                        key = json.dumps(new, sort_keys=True)
                        if key not in seen:
                            members.append(new)
                            seen.add(key)
        membership[item_id] = members
    profile = {"schema_version": 1, "name": "paper", "description": "Deduplicated paper scheduling experiments; other items use CPU analyses.",
               "provisional": True, "execution_ready": True,
               "validation_pending": ["GPU execution and paper performance have not been measured during release preparation."],
               "hardware": "a100_80gb_8gpu", "execution_protocol": "camera_ready_reconstruction_v1",
               "task_timeout_seconds": None, "defaults": catalog["defaults"], "workloads": []}
    for case_id, case in cases.items():
        workload = {"id": case_id, "test_case": case["test_case"], "methods": case["methods"],
                    "matrix": {k: [v] for k, v in case["parameters"].items()}}
        if "min_r" in case["methods"] and len(case["model_paths"]) >= 8:
            workload["method_matrix"] = {"min_r": {"seed": catalog["protocol"]["new_run_seeds"]}}
        profile["workloads"].append(workload)
    if not resolve_tasks:
        return catalog, profile, {"cases": cases, "membership": membership}, None
    with tempfile.TemporaryDirectory(prefix="mil-paper-profile-") as temporary:
        generated = Path(temporary) / "paper.yaml"
        generated.write_text(yaml.safe_dump(profile, sort_keys=False))
        expanded = expand_experiment(generated)
    tasks = {t.task_id: t for t in expanded.tasks}
    item_reports = {}
    for item_id, members in membership.items():
        task_ids = list(dict.fromkeys(t.task_id for member in members for t in expanded.tasks
                                     if t.workload_id == member["case_id"] and t.method_id in member["methods"]))
        item_reports[item_id] = {"kind": catalog["items"][item_id]["kind"], "members": members,
                                 "task_ids": task_ids,
                                 "analysis_required": catalog["items"][item_id]["kind"] != "scheduling"}
    report = {"schema_version": 1, "paper_sha256": catalog["paper_sha256"], "gpu_execution": False,
              "unique_cases": len(cases), "unique_scheduling_tasks": len(tasks),
              "items": item_reports, "cases": cases,
              "timing_policy": "CPU time is not the paper GPU inference time; no equality claim for search wall clock.",
              "seed_policy": "Seeds 0/1/2 are an explicit new-run policy, not reconstructed historical RNG state.",
              "extension_profiles_preserved": catalog["extensions"]["retained_profiles"]}
    return catalog, profile, report, expanded


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    _, profile, report, expanded = compile_suite(args.config)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "paper.yaml").write_text(yaml.safe_dump(profile, sort_keys=False))
    (args.output / "coverage.json").write_text(json.dumps(report, indent=2) + "\n")
    with (args.output / "tasks.jsonl").open("x") as stream:
        for task in expanded.tasks:
            stream.write(json.dumps(task.to_dict()) + "\n")
    print(json.dumps({"paper_items": len(report["items"]), "unique_cases": report["unique_cases"],
                      "scheduling_tasks": report["unique_scheduling_tasks"], "gpu_execution": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
