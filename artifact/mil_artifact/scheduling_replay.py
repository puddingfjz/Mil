"""CPU-only replay of initial LLM-Blender scheduling from recorded input lengths.

Calls the real scheduler and cost simulator, never the inference entry point.
This checks initial plans/predicted costs, not GPU execution/runtime adjustment.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
import time
import tokenize

from .config import repo_root
from .hashing import sha256_file
from .schema import atomic_write_json


def literal_prefix(text: str):
    depth = 0
    for token in tokenize.generate_tokens(io.StringIO(text).readline):
        if token.type != tokenize.OP:
            continue
        if token.string in "[({":
            depth += 1
        elif token.string in "])}":
            depth -= 1
            if depth == 0:
                end = token.end[1]
                return ast.literal_eval(text[:end]), text[end:].strip()
    raise ValueError("missing complete literal")


def parse_plan(text: str) -> dict:
    plans, remaining = literal_prefix(text)
    times, remaining = literal_prefix(remaining)
    return {"plans": plans, "stage_times": times, "predicted_seconds": float(remaining.split()[0])}


def read_inputs(path: Path) -> dict:
    result = {"parameters": {}, "input_lengths": {}}
    reading_header = True
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if line.startswith("running in "):
                reading_header = False
            match = re.fullmatch(r"len\(inp_lens\) of model (\d+): (.*)", line)
            if match:
                result["input_lengths"][int(match[1])] = ast.literal_eval(match[2])
            elif line.startswith("real model_paths: "):
                result["model_paths"] = ast.literal_eval(line.split(": ", 1)[1])
            elif line.startswith("real in_edge_dict_with_dummy_inp_nodes: "):
                result["input_edges"] = ast.literal_eval(line.split(": ", 1)[1])
            elif line.startswith("Best group seq: "):
                result["expected"] = parse_plan(line.split(": ", 1)[1])
                break
            elif line.startswith("best_group_seq: "):
                result["expected"] = {"plans": ast.literal_eval(line.split(": ", 1)[1])}
            elif line.startswith("other hybridflow res infor: "):
                result["expected"]["predicted_seconds"] = float(line.split("(", 1)[1].split(",", 1)[0])
                break
            elif reading_header and re.match(r"^[a-z_]+: ", line):
                key, value = line.split(": ", 1)
                try:
                    value = ast.literal_eval(value)
                except (SyntaxError, ValueError):
                    pass
                result["parameters"][key] = value
    p = result["parameters"]
    if p.get("test_case") != "general" or p.get("outlen_known") is not False or p.get("specify_outlen") is not False:
        raise ValueError("replay currently supports LLM-Blender with sampled output lengths only")
    if p.get("search_method_baseline") not in {"greedy_saturn_LPT", "hybridflow_model_parallel"}:
        raise ValueError("unsupported replay method")
    count = len(result["model_paths"])
    if result.get("input_edges") != {i: [-i - 1] for i in range(count)}:
        raise ValueError("expected independent LLM-Blender model graph")
    if set(result["input_lengths"]) != set(range(count)) or not result.get("expected"):
        raise ValueError("missing logged model inputs or initial schedule")
    if any(len(lens) != p["reqnum"] or any(not isinstance(n, int) or n <= 0 for n in lens)
           for lens in result["input_lengths"].values()):
        raise ValueError("incomplete or invalid input lengths")
    return result


def compare(expected: dict, actual: dict) -> dict:
    # JSON normalizes tuples/lists without changing order or numerical values.
    plans_equal = json.loads(json.dumps(expected["plans"])) == json.loads(json.dumps(actual["plans"]))
    error = abs(expected["predicted_seconds"] - actual["predicted_seconds"])
    times_match = None
    if "stage_times" in expected or "stage_times" in actual:
        times_match = len(expected.get("stage_times", [])) == len(actual.get("stage_times", [])) and all(
            math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-9)
            for a, b in zip(expected.get("stage_times", []), actual.get("stage_times", []))
        )
    return {"plans_exact_match": plans_equal, "stage_times_match": times_match,
            "predicted_absolute_error_seconds": error,
            "passed": plans_equal and times_match is not False and error <= 1e-9}


def replay(source: Path, output: Path, cpus: int) -> dict:
    inputs = read_inputs(source)
    p = inputs["parameters"]
    root = repo_root()
    output.mkdir(parents=True, exist_ok=False)
    source_hash = sha256_file(source)
    # Force CPU visibility before importing torch/Ray/vLLM, including Ray workers.
    os.environ.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      PYTHONDONTWRITEBYTECODE="1", MIL_EXPERIMENT_SEED="0",
                      OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    paths = [str(root / "benchmarks"), str(root)]
    sys.path[:0] = paths
    os.environ["PYTHONPATH"] = os.pathsep.join(paths + [os.environ.get("PYTHONPATH", "")])
    os.chdir(output)
    with (output / "scheduler.log").open("x", buffering=1) as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        import torch
        import ray
        import numpy as np
        from experiment_reproducibility import seed_process
        import search_exec_plans as scheduler
        import output_length_sampler

        def forbid_cuda(*args, **kwargs):
            raise RuntimeError("CUDA initialization is forbidden in scheduling replay")

        torch.cuda._lazy_init = forbid_cuda
        seed_process(0)
        # A private local CPU Ray instance; never connect to an existing cluster.
        ray_directory = tempfile.mkdtemp(prefix="mil-ray-")
        ray.init(address="local", num_cpus=cpus, num_gpus=0, include_dashboard=False,
                 object_store_memory=256 * 1024**2, _temp_dir=ray_directory,
                 log_to_driver=False)
        model_paths = inputs["model_paths"]
        lengths = inputs["input_lengths"]
        n = len(model_paths)
        def input_generator(num_prompts, model_id, model_path, ids):
            return list(lengths[model_id])
        def output_generator(model_id, model_name, inp_lens):
            return np.minimum(p["max_token_num"], output_length_sampler.sample_out_len_for_given_model(
                model_name, inp_lens, p["noise_level"]))
        def merger(values):
            return [sum(v) for v in zip(*values)]
        started = time.monotonic()
        try:
            best = scheduler.get_best_model_schedule(
                search_method_baseline=p["search_method_baseline"],
                gen_execplans_baseline=p["gen_execplans_baseline"],
                check_gap=1600, sort_input=True, model_paths=model_paths,
                num_prompts=p["reqnum"], inp_seq_ids_dict={i: list(range(p["reqnum"])) for i in range(n)},
                out_req_id_mapping={}, inp_req_from_which_models={},
                inp_req_from_which_model_which_out_reqs={}, independent_srcs={i: False for i in range(n)},
                inp_generators=[input_generator] * n, inp_mergers=[merger] * n,
                outlen_generators=[output_generator] * n, prompt_templates_lens={}, out_edge_dict={},
                sample_config=(1, 1, -1, 0), trust_remote_code=True, revision=None,
                gpu_name=p["gpu_name"], tot_gpu_num=p["tot_gpu_num"], byte_per_gpu=p["byte_per_gpu"],
                data_byte=2, max_group_seq_num=p["max_group_seq_num"], top_k=p["top_k"],
                similar_threshold=p["similar_threshold"], fully_connected_gpu_unit=p["fully_connected_gpu_unit"],
                machine_name=p["machine_name"],
            )
            if "hybridflow" in p["search_method_baseline"]:
                actual = {"plans": [[str(plan) for plan in group.exec_plans] for group in best.plan_group_seq],
                          "predicted_seconds": float(best.hybridflow_result[0])}
            else:
                actual = parse_plan(best.get_str_using_model_ids())
            cuda_initialized = torch.cuda.is_initialized()
            assert not cuda_initialized
        finally:
            ray.shutdown()
    report = {"schema_version": 1, "scope": "initial_scheduling_only",
              "source_log": str(source), "source_log_sha256": source_hash,
              "parameters": p, "model_paths": model_paths,
              "input_lengths_sha256": hashlib.sha256(json.dumps(lengths, sort_keys=True).encode()).hexdigest(),
              "cpu_workers": cpus, "cuda_initialized": cuda_initialized,
              "ray_directory": ray_directory,
              "wall_seconds": time.monotonic() - started,
              "source_hashes": {name: sha256_file(root / "benchmarks" / name) for name in (
                  "search_exec_plans.py", "fake_scheduling.py", "output_length_sampler.py",
                  "get_my_cost_table_directly_zxcpu.py")},
              "expected": inputs["expected"], "actual": actual,
              "comparison": compare(inputs["expected"], actual),
              "limitations": ["No GPU inference, runtime adjustment, or actual completion-time validation.",
                              "Input token lengths are replayed from the log, not retokenized from raw data."]}
    atomic_write_json(output / "comparison.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpus", type=int, default=2)
    args = parser.parse_args()
    if args.cpus < 1:
        parser.error("--cpus must be positive")
    report = replay(args.source_log.resolve(), args.output.resolve(), args.cpus)
    print(json.dumps({"output": str(args.output), "comparison": report["comparison"]}, indent=2))
    return 0 if report["comparison"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
