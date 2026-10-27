"""CPU-only construction of cost tables and output-length distributions.

`plan` writes GPU acquisition commands as data; this module never executes them.
`build` reads existing logs and writes a new, standalone resource directory.
"""

from __future__ import annotations

import argparse
import ast
from collections import defaultdict
import json
import math
from pathlib import Path
import re
import sys

import numpy as np

from .config import load_yaml, repo_root
from .hashing import sha256_file


TABLES = ("prefill_table", "decode_table", "sample_prefill_table",
          "sample_decode_table", "prepInp_prefill_table", "prepInp_decode_table")
COST_FIELDS = TABLES + ("model_configs", "prepare_cost_table")


def read_literal_module(path: Path, names: tuple[str, ...]) -> dict:
    """Read legacy literal assignments/update/extend without importing code.

    In particular, never import the cost module's vLLM-dependent loader.
    Unsupported syntax fails rather than executing arbitrary Python.
    """
    values = {}
    for node in ast.parse(path.read_text(encoding="utf-8"), filename=str(path)).body:
        if isinstance(node, ast.Import) and [a.name for a in node.names] == ["my_per_iter_latency_estimator"]:
            continue
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            continue
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name == "cost_table" and isinstance(node.value, ast.Call) and (
                ast.unparse(node.value.func) == "my_per_iter_latency_estimator.get_cost_table_from_serialized_data"
            ):
                continue
            if name not in names:
                raise ValueError(f"{path}: unexpected assignment {name}")
            if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and (
                node.value.func.id == "dict" and not node.value.args and not node.value.keywords
            ):
                values[name] = {}
            else:
                values[name] = ast.literal_eval(node.value)
            continue
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            call = node.value
            if isinstance(call.func, ast.Attribute) and len(call.args) == 1 and not call.keywords:
                target = call.func.value
                if isinstance(target, ast.Name) and target.id in values and call.func.attr == "update":
                    update = ast.literal_eval(call.args[0])
                    if not isinstance(values[target.id], dict) or not isinstance(update, dict):
                        raise ValueError(f"{path}: update requires dictionaries")
                    values[target.id].update(update)
                    continue
                if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name) and (
                    target.value.id in values and call.func.attr == "extend"
                ):
                    index = ast.literal_eval(target.slice)
                    if index not in (0, 1):
                        raise ValueError(f"{path}: invalid serialized-table index")
                    values[target.value.id][index].extend(ast.literal_eval(call.args[0]))
                    continue
        raise ValueError(f"{path}:{node.lineno}: unsupported resource syntax")
    if set(values) != set(names):
        raise ValueError(f"{path}: missing fields {set(names) - set(values)}")
    return values


def read_cost_table(path: Path) -> dict:
    result = read_literal_module(path, COST_FIELDS)
    for field in TABLES:
        keys, rows = result[field]
        if len(keys) != len(rows):
            raise ValueError(f"{path}: mismatched keys/rows in {field}")
        result[field] = {key: {r[0]: [(r[1], r[2]), (r[3], r[4])] for r in group}
                         for key, group in zip(keys, rows)}
    return result


def native(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {native(k): native(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(native(v) for v in value)
    if isinstance(value, (list, np.ndarray)):
        return [native(v) for v in value]
    return value


def cost_module_text(data: dict) -> str:
    lines = ["# Generated offline; measurements/provenance are in report.json.",
             "import my_per_iter_latency_estimator"]
    for field in TABLES:
        keys = list(data[field])
        rows = [[(batch, *points[0], *points[1]) for batch, points in data[field][key].items()]
                for key in keys]
        lines.append(f"{field} = {native((keys, rows))!r}")
    for field in COST_FIELDS[-2:]:
        lines.append(f"{field} = {native(data[field])!r}")
    lines.append("cost_table = my_per_iter_latency_estimator.get_cost_table_from_serialized_data("
                 + ", ".join(COST_FIELDS) + ")")
    return "\n".join(lines) + "\n"


def positive_int(value, label):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def distribution_key(model, temperature):
    # Historical temperature-1 resources use a plain name, other temperatures
    # use (name, temperature). Keep the scheduler's existing lookup contract.
    name = model.rsplit("/", 1)[-1]
    return name if temperature == 1 else (name, temperature)


def finite_number(value, label, minimum=0):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < minimum:
        raise ValueError(f"{label} must be finite and >= {minimum}")
    return value


def distribution_from_pairs(pairs: list, max_model_len: int) -> tuple[list, dict]:
    """Preserve analyse_script.py's censoring-aware PDF construction.

    Requests reaching the context limit are censored, not counted as completed
    outputs at that length. Remaining probability is placed in the final bin.
    """
    positive_int(max_model_len, "max_model_len")
    if not pairs or max_model_len < 2:
        raise ValueError("nonempty length records and max_model_len >= 2 are required")
    lengths = []
    for row in pairs:
        if not isinstance(row, (list, tuple)) or len(row) < 2:
            raise ValueError("each output-length record needs input and output lengths")
        inp = positive_int(row[0], "input length")
        out = positive_int(row[1], "output length")
        if inp + out > max_model_len:
            raise ValueError("input + output exceeds max_model_len")
        lengths.append((inp, out))
    inps, outs = np.asarray(lengths, dtype=np.int64).T
    events = np.zeros(max_model_len, dtype=np.int64)
    sizes = np.zeros(max_model_len, dtype=np.int64)
    for inp in np.unique(inps):
        selected = outs[inps == inp]
        values, counts = np.unique(selected, return_counts=True)
        for out, count in zip(values, counts):
            if inp + out < max_model_len:
                events[out - 1] += count
        sizes[:max_model_len - inp - 1] += len(selected)
    sizes[events == 0] = 1
    pdf = events / sizes
    cumulative = np.cumsum(pdf)
    pdf[-1] = 1 - cumulative[-2]
    # Permit floating point roundoff only; never silently normalize bad data.
    if not np.isfinite(pdf).all() or float(pdf.min()) < -1e-12 or not math.isclose(float(sum(pdf)), 1, abs_tol=1e-12):
        raise ValueError("legacy distribution estimator produced invalid probabilities")
    if pdf[-1] < 0:
        pdf[-1] = 0.0
    return pdf.tolist(), {"requests": len(lengths), "censored_requests": int(sum(inps + outs == max_model_len)),
                         "max_model_len": max_model_len}


def read_lengths(path: Path, block: int | None, expected_requests: int | None) -> list:
    blocks = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.startswith("output_lens = "):
                blocks.append(json.loads(line[len("output_lens = "):]))
    if not blocks:
        raise ValueError(f"{path}: no output_lens block")
    if block is None and len(blocks) != 1:
        raise ValueError(f"{path}: {len(blocks)} blocks; select record_block explicitly")
    index = 0 if block is None else block
    if type(index) is not int or not 0 <= index < len(blocks):
        raise ValueError(f"{path}: invalid record_block")
    result = blocks[index]
    if expected_requests is not None and len(result) != expected_requests:
        raise ValueError(f"{path}: expected {expected_requests} requests, got {len(result)}")
    return result


def select_endpoints(records: list, profile_num: int, piece_num: int, threshold: float) -> list:
    """Same minimum-of-repeats and endpoints as CostTable's selection method."""
    if not records or len(records) % profile_num:
        raise ValueError("incomplete cost profile repetitions")
    xs = np.asarray([p[0] for p in records]).reshape((-1, profile_num))
    ys = np.asarray([p[1] for p in records]).reshape((-1, profile_num))
    # Historical records can have slightly different context totals within a
    # repeated group. Like the legacy method, retain x from the fastest record.
    indices = np.argmin(ys, axis=1)
    xs, ys = xs[np.arange(len(xs)), indices], ys[np.arange(len(ys)), indices]
    if len(np.unique(xs)) < 2:
        raise ValueError("at least two distinct workload points are required")
    if piece_num == 1:
        indices = [np.argmin(xs), np.argmax(xs)]
        return [(float(xs[i]), float(ys[i])) for i in indices]
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    for x, y in zip(xs, ys):
        if y > ys[0] + threshold and x < xs[-1]:
            slope = (ys[-1] - y) / (xs[-1] - x)
            if slope <= 0:
                raise ValueError("invalid piecewise cost slope")
            intersection = (ys[0] - y) / slope + x
            return [(float(intersection), float(ys[0])), (float(xs[-1]), float(ys[-1]))]
    return [(float(xs[0]), float(ys[0])), (float(xs[-1]), float(ys[-1]))]


def build_cost_entry(path: Path, entry: dict, model_config, coefficient: float) -> tuple[dict, dict]:
    model, plan = entry["model"], tuple(entry["exec_plan"])
    sample = tuple(entry.get("sample_config", [1, 1, -1, 0]))
    h, layers = model_config
    positive_int(h, "hidden size")
    positive_int(layers, "layer count")
    finite_number(coefficient, "FLOPs coefficient", 1)
    profile_num = positive_int(entry.get("profile_num", 2), "profile_num")
    skip = entry.get("skip_records", 2)
    if type(skip) is not int or skip < 0:
        raise ValueError("skip_records must be a nonnegative integer")
    threshold = finite_number(entry.get("threshold_seconds", 0.001), "threshold_seconds")
    groups = {name: defaultdict(list) for name in TABLES}
    count, header_tp = 0, None
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            match = re.search(r"tensor_parallel_size=(\d+)", line)
            if match:
                header_tp = int(match[1])
                if header_tp != plan[0]:
                    raise ValueError(f"{path}: tensor parallel size does not match exec_plan")
            if not line.startswith("[["):
                continue
            record = json.loads(line)
            meta, _, times, _ = record[:4]
            if len(record) > 4 and record[4] is not None and record[4] != plan[0]:
                raise ValueError(f"{path}: record tensor parallel size does not match exec_plan")
            count += 1
            if count <= skip:
                continue
            batch, context, _, max_len, is_prompt = meta
            for value in (batch, context, max_len):
                positive_int(value, "profile workload")
            if type(is_prompt) is not bool:
                raise ValueError("profile phase must be a boolean")
            for value in times[1:4]:
                finite_number(value, "measured latency")
            if len(times) < 4:
                raise ValueError("profile record is missing latency components")
            if is_prompt:
                flops = layers * (batch * max_len * coefficient + 2 * batch * max_len * h * max_len / plan[0])
            else:
                flops = layers * (batch * coefficient + 2 * h * context / plan[0])
            phase = "prefill" if is_prompt else "decode"
            groups[f"{phase}_table"][batch].append((flops / 1e12, times[1]))
            groups[f"sample_{phase}_table"][batch].append((context, times[2]))
            groups[f"prepInp_{phase}_table"][batch].append((batch * max_len, times[3]))
    result = {}
    for field, batches in groups.items():
        if not batches:
            raise ValueError(f"{path}: missing {field} measurements")
        piece_num = 2 if field in TABLES[:2] and plan[2] > 2 else 1
        result[field] = {b: select_endpoints(rows, profile_num, piece_num, threshold)
                         for b, rows in batches.items()}
    return result, {"model": model, "exec_plan": list(plan), "sample_config": list(sample),
                    "records": count, "skipped_records": skip,
                    "batch_rows": {k: len(v) for k, v in result.items()}}


def compare_values(expected, actual, atol=1e-12) -> dict:
    errors = []
    mismatches = []

    def visit(a, b, where):
        if isinstance(a, dict) and isinstance(b, dict):
            if set(a) != set(b):
                mismatches.append(where + ": keys")
            for key in a.keys() & b.keys():
                visit(a[key], b[key], f"{where}/{key}")
        elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
            if len(a) != len(b):
                mismatches.append(where + ": length")
            for i, (x, y) in enumerate(zip(a, b)):
                visit(x, y, f"{where}/{i}")
        elif isinstance(a, (float, int)) and isinstance(b, (float, int)):
            if not math.isfinite(a) or not math.isfinite(b):
                mismatches.append(where + ": nonfinite")
            else:
                errors.append(abs(a - b))
        elif a != b:
            mismatches.append(where + ": value")

    visit(expected, actual, "resource")
    maximum = max(errors, default=0.0)
    return {"passed": not mismatches and maximum <= atol, "max_absolute_error": maximum,
            "numeric_values_compared": len(errors), "structural_mismatches": mismatches[:10],
            "absolute_tolerance": atol}


def load_spec(path: Path, models=()) -> dict:
    spec = load_yaml(path)
    if set(spec) - {"schema_version", "name", "description", "cost", "output_lengths"}:
        raise ValueError("unknown resource configuration fields")
    available = set()
    for section in ("cost", "output_lengths"):
        if section not in spec:
            continue
        entries = spec[section].get("entries", [])
        if not isinstance(entries, list):
            raise ValueError(f"{section}.entries must be a list")
        seen = set()
        for entry in entries:
            model = entry.get("model")
            if not isinstance(model, str) or not model:
                raise ValueError("every resource entry requires model")
            available.add(model)
            if not isinstance(entry.get("log"), str) or not entry["log"]:
                raise ValueError("every resource entry requires log")
            if section == "cost":
                plan = entry.get("exec_plan")
                if not isinstance(plan, list) or len(plan) != 4:
                    raise ValueError("exec_plan must be [tp, gpu_ratio, weight_load_degree, cache_gpu_num]")
                positive_int(plan[0], "tp")
                if not 0 < finite_number(plan[1], "gpu ratio") <= 1:
                    raise ValueError("gpu ratio must be in (0, 1]")
                positive_int(plan[2], "weight_load_degree")
                if type(plan[3]) is not int or plan[3] < 0:
                    raise ValueError("cache_gpu_num must be a nonnegative integer")
                sample = entry.get("sample_config", [1, 1, -1, 0])
                if not isinstance(sample, list) or len(sample) != 4:
                    raise ValueError("sample_config must have four entries")
                key = (model, tuple(plan))
            else:
                positive_int(entry.get("max_model_len"), "max_model_len")
                temperature = finite_number(entry.get("temperature", 1.0), "temperature")
                if "expected_requests" in entry:
                    positive_int(entry["expected_requests"], "expected_requests")
                positive_int(entry.get("num_prompts", 10000), "num_prompts")
                positive_int(entry.get("tensor_parallel_size", 2), "tensor_parallel_size")
                key = distribution_key(model, temperature)
            if key in seen:
                raise ValueError(f"duplicate {section} entry: {key}")
            seen.add(key)
        if models:
            spec[section]["entries"] = [e for e in entries if e["model"] in models]
    if set(models) - available:
        raise ValueError(f"unknown models: {set(models) - available}")
    if not any(spec.get(s, {}).get("entries") for s in ("cost", "output_lengths")):
        raise ValueError("configuration selects no resource entries")
    return spec


def plan_acquisition(spec: dict, root: Path, python: str, output: Path | None = None) -> dict:
    jobs = []
    for section in ("cost", "output_lengths"):
        for entry in spec.get(section, {}).get("entries", []):
            if section == "cost":
                tp, ratio, degree, _ = entry["exec_plan"]
                sample = entry.get("sample_config", [1, 1, -1, 0])
                if sample[1:] != [1, -1, 0] or entry["exec_plan"][3] != 0:
                    raise ValueError("legacy acquisition only supports top_p=1/top_k=-1/min_p=0 and cache_gpu_num=0")
                if entry.get("profile_num", 2) != 2 or entry.get("threshold_seconds", 0.001) != 0.001:
                    raise ValueError("legacy profiler fixes profile_num=2 and threshold_seconds=0.001")
                temperature = sample[0]
                argv = [python, "construct_cost_model.py", "--input-len", "16", "--output-len", "16",
                        "--num-prompts", "1", "--backend", "vllm"]
            else:
                tp, ratio, degree = entry.get("tensor_parallel_size", 2), 0.9, 2
                temperature = entry.get("temperature", 1.0)
                argv = [python, "benchmark_throughput.py", "--backend", "vllm_ori",
                        "--dataset", entry.get("dataset", "no_robot.parquet"),
                        "--num-prompts", str(entry.get("num_prompts", 10000)),
                        "--max-model-len", str(entry["max_model_len"])]
            argv += ["--model", entry["model"], "-tp", str(tp), "--temperature", str(temperature),
                     "-gpuratio", str(ratio), "-wldegree", str(degree), "--enforce-eager", "--trust-remote-code"]
            log_name = Path("measurements") / f"{len(jobs):04d}_{section}.log"
            destination = str(output.resolve() / log_name) if output is not None else str(log_name)
            jobs.append({"kind": section, "model": entry["model"], "argv": argv,
                         "cwd": str(root / "benchmarks"), "required_gpus": tp,
                         "stdout": destination, "source_log": entry["log"], "executed": False})
    return {"schema_version": 1, "mode": "acquisition_plan_only", "gpu_execution": False, "jobs": jobs,
            "note": "Commands are data only. Select GPUs and a new log destination before any future acquisition."}


def build_resources(spec: dict, input_root: Path) -> tuple[dict, dict]:
    sources = {}

    def source(name):
        path = Path(name)
        path = path if path.is_absolute() else input_root / path
        path = path.resolve(strict=True)
        sources[str(path)] = sha256_file(path)
        return path

    outputs, report = {}, {"schema_version": 1, "mode": "cpu_build_from_existing_measurements",
                           "gpu_execution": False, "cost_entries": [], "output_length_entries": []}
    cost_spec = spec.get("cost", {})
    if cost_spec.get("entries"):
        data = (read_cost_table(source(cost_spec["base_table"])) if cost_spec.get("base_table")
                else {field: {} for field in COST_FIELDS})
        expected = (read_cost_table(source(cost_spec["compare_table"])) if cost_spec.get("compare_table") else None)
        coefficients = (read_literal_module(source(cost_spec["coefficient_table"]), ("model_coeffs",))["model_coeffs"]
                        if cost_spec.get("coefficient_table") else {})
        for entry in cost_spec["entries"]:
            model, plan = entry["model"], tuple(entry["exec_plan"])
            model_key = (model, True, entry.get("revision"))
            sample = tuple(entry.get("sample_config", [1, 1, -1, 0]))
            config = entry.get("model_config", data["model_configs"].get(model_key))
            coefficient = entry.get("flops_coefficient", coefficients.get((model, plan[0])))
            if config is None or coefficient is None:
                raise ValueError(f"{model}: model_config and FLOPs coefficient must be supplied; no model download is attempted")
            built, details = build_cost_entry(source(entry["log"]), entry, config, coefficient)
            data["model_configs"][model_key] = tuple(config)
            short_name = model.rsplit("/", 1)[-1]
            prepare_key = (short_name, plan)
            if "prepare_seconds" in entry:
                data["prepare_cost_table"][prepare_key] = finite_number(entry["prepare_seconds"], "prepare_seconds")
            if prepare_key not in data["prepare_cost_table"]:
                raise ValueError(f"{model}: supply prepare_seconds or a base table with model loading cost")
            comparisons = {}
            for field in TABLES:
                key = (model, plan, sample) if field.startswith("sample_") else (model, plan)
                data[field][key] = built[field]
                if expected is not None:
                    comparisons[field] = compare_values(expected[field].get(key, {}), built[field])
            details["comparison"] = comparisons or None
            report["cost_entries"].append(details)
        report["inherited_cost_entries"] = len(data["prefill_table"]) - len(cost_spec["entries"])
        outputs["cost_table.py"] = cost_module_text(data)
    length_spec = spec.get("output_lengths", {})
    if length_spec.get("entries"):
        pdfs = (read_literal_module(source(length_spec["base_distribution"]), ("pdf_dict",))["pdf_dict"]
                if length_spec.get("base_distribution") else {})
        expected = (read_literal_module(source(length_spec["compare_distribution"]), ("pdf_dict",))["pdf_dict"]
                    if length_spec.get("compare_distribution") else None)
        for entry in length_spec["entries"]:
            pairs = read_lengths(source(entry["log"]), entry.get("record_block"), entry.get("expected_requests"))
            pdf, details = distribution_from_pairs(pairs, entry["max_model_len"])
            temperature = entry.get("temperature", 1.0)
            key = distribution_key(entry["model"], temperature)
            pdfs[key] = pdf
            details.update(model=entry["model"], temperature=temperature,
                           comparison=compare_values(expected.get(key, []), pdf) if expected is not None else None)
            report["output_length_entries"].append(details)
        report["inherited_output_distributions"] = len(pdfs) - len(length_spec["entries"])
        outputs["output_lengths.py"] = "# Generated offline; see report.json.\npdf_dict = " + repr(pdfs) + "\n"
    report["sources_sha256"] = sources
    comparisons = [v for e in report["cost_entries"] for v in (e["comparison"] or {}).values()]
    comparisons += [e["comparison"] for e in report["output_length_entries"] if e["comparison"] is not None]
    report["comparisons_passed"] = all(v["passed"] for v in comparisons) if comparisons else None
    return outputs, report


def write_bundle(output: Path, files: dict, report: dict):
    # Deliberately no --force and no append: previous resources stay immutable.
    output.mkdir(parents=True, exist_ok=False)
    for name, content in files.items():
        with (output / name).open("x", encoding="utf-8") as stream:
            stream.write(content)
    report["outputs_sha256"] = {name: sha256_file(output / name) for name in files}
    with (output / "report.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["plan", "build"])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, default=repo_root())
    parser.add_argument("--output", type=Path, required=True, help="new directory; existing paths are rejected")
    parser.add_argument("--model", action="append", default=[], help="exact model ID; repeat to select several")
    parser.add_argument("--python", default="python3", help="interpreter recorded in plan, never invoked")
    args = parser.parse_args(argv)
    try:
        if args.output.exists() or args.output.is_symlink():
            raise ValueError(f"output already exists: {args.output}")
        spec = load_spec(args.config, args.model)
        if args.action == "plan":
            report = plan_acquisition(spec, repo_root(), args.python, args.output)
            files = {"acquisition.json": json.dumps(report, indent=2) + "\n"}
            build_spec = json.loads(json.dumps(spec))
            jobs = iter(report["jobs"])
            for section in ("cost", "output_lengths"):
                for entry in build_spec.get(section, {}).get("entries", []):
                    entry["log"] = next(jobs)["stdout"]
            files["build_config.json"] = json.dumps(build_spec, indent=2) + "\n"
        else:
            files, report = build_resources(spec, args.input_root)
        report["config_sha256"] = sha256_file(args.config)
        report["resolved_config"] = spec
        report["builder_sha256"] = sha256_file(Path(__file__))
        report["python_version"] = sys.version.split()[0]
        report["numpy_version"] = np.__version__
        report["gpu_modules_imported"] = [name for name in ("torch", "vllm", "ray") if name in sys.modules]
        write_bundle(args.output, files, report)
        print(json.dumps({"output": str(args.output), "gpu_execution": False,
                          "cost_entries": len(report.get("cost_entries", [])),
                          "output_length_entries": len(report.get("output_length_entries", [])),
                          "comparisons_passed": report.get("comparisons_passed")}, indent=2))
        return 1 if report.get("comparisons_passed") is False else 0
    except (ValueError, OSError, KeyError, TypeError, SyntaxError) as exc:
        print(f"resource build failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
