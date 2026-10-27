"""Read-only historical log indexing, with bounded memory and no model imports."""
from __future__ import annotations

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import mmap
from pathlib import Path
import re
import time

from .config import PARAM_FLAGS, repo_root
from .collect import _PATTERNS
from .schema import atomic_write_json


HEADER_KEYS = set(PARAM_FLAGS) | {"test_case", "search_method_baseline", "gen_execplans_baseline",
                                "use_real_dataset", "version", "sample_size", "seed"}
ERROR_MARKERS = {b"Traceback (most recent call last)": "traceback",
                 b"CUDA out of memory": "cuda_out_of_memory", b"AssertionError": "assertion_error",
                 b"License has expired": "expired_solver_license"}
EVENT = re.compile(rb"(?m)^(?:MAIN PROCESS: (?:next iter plans: )?\[|all stage times: time lengths: ).*$")
METRIC_MARKERS = (b'total running time:', b'Total search time:', b'Total time for preparation before search:',
                  b'begin fusing similar models:', b'finish fusing similar models:', b'begin search:',
                  b'other hybridflow res infor:', b'Best group seq:', b'update_workload_dict min_cost:',
                  b'Cost_Query_', b'Plan_Group_', b'Iter_Search_Time', b'Select_Best_Time',
                  b'real model_paths: ')


def scalar(value):
    try:
        parsed = ast.literal_eval(value)
        return parsed if isinstance(parsed, (str, int, float, bool, type(None))) else None
    except (ValueError, SyntaxError):
        return value.strip()


def bounded_metadata(head: bytes, tail: bytes) -> dict:
    params = {}
    text = (head + b"\n" + tail).decode("utf-8", errors="replace")
    for line in head.decode("utf-8", errors="replace").splitlines():
        key, sep, value = line.partition(": ")
        if sep and key in HEADER_KEYS:
            params[key] = scalar(value)
    values = {}
    for name, pattern in _PATTERNS.items():
        matches = list(pattern.finditer(text))
        if matches:
            values[name] = float(matches[-1][1])
    prep = values.get("preparation_time_seconds")
    if prep is None and "fuse_end_seconds" in values and "begin_search_seconds" in values:
        prep = values["fuse_end_seconds"] - values["begin_search_seconds"]
        values["preparation_time_seconds"] = prep
    parts = [values.get(k) for k in ("real_running_time_seconds", "search_time_seconds", "preparation_time_seconds")]
    if all(p is not None for p in parts):
        values["total_time_seconds"] = parts[0] + parts[1]
        values["extra_time_seconds"] = parts[1]
        values["search_after_preparation_seconds"] = parts[1] - parts[2]
    best = re.findall(r"(?m)^Best group seq: .*", text)
    if best and "LightRule" not in str(params.get("search_method_baseline", "")):
        try:
            values["initial_predicted_seconds"] = float(best[0].split()[-2])
        except (ValueError, IndexError):
            pass
    elif "hybridflow_predicted_seconds" in values:
        values["initial_predicted_seconds"] = values["hybridflow_predicted_seconds"]
    for key in ("Cost_Query_Time", "Cost_Query_num", "Plan_Group_Enu_Time", "Plan_Group_Enu_num",
                "Plan_Group_num_After_Pruning", "Iter_Search_Time", "Select_Best_Time"):
        found = re.findall(rf"\b{key}\s*[:=]\s*([\d.eE+-]+)", text)
        if found:
            try:
                values[key] = float(found[-1])
            except ValueError:
                pass
    for line in text.splitlines():
        if line.startswith('Cost_Query_Time, Cost_Query_num, '):
            try:
                labels, raw = line.split(': ', 1)
                values.update(zip(labels.split(', '), ast.literal_eval(raw)))
            except (ValueError, SyntaxError):
                pass
    model_paths = None
    found = re.findall(r"(?m)^real model_paths: (\[.*\])$", text)
    if found:
        try:
            model_paths = ast.literal_eval(found[-1])
        except (ValueError, SyntaxError):
            pass
    return {"parameters": params, "metrics": values, "model_paths": model_paths,
            "metadata_scan": "first_2MiB_and_last_4MiB",
            "complete_timing": all(p is not None and p >= 0 for p in parts),
            "exit_status": "not_recorded"}


def index_log(path: Path, root: Path) -> dict:
    if path.is_symlink():
        raise ValueError(f"symlink log refused: {path}")
    before = path.stat()
    digest = hashlib.sha256()
    errors, events = set(), []
    overlap = b""
    with path.open("rb") as stream:
        head = stream.read(2 * 1024**2)
        stream.seek(max(0, before.st_size - 4 * 1024**2))
        tail = stream.read()
        stream.seek(0)
        while chunk := stream.read(8 * 1024**2):
            digest.update(chunk)
            # All-file fatal diagnostics are independent of bounded timing extraction.
            joined = overlap + chunk
            for marker, label in ERROR_MARKERS.items():
                if marker in joined:
                    errors.add(label)
            overlap = joined[-128:]
            # Event extraction uses a separate streaming pass for selected logs;
            # hashing all 343 GB should not retain request text or giant lines.
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"log changed during scan: {path}")
    record = bounded_metadata(head, tail)
    record.update(path=path.relative_to(root).as_posix(), sha256=digest.hexdigest(),
                  size_bytes=before.st_size, mtime_ns=before.st_mtime_ns,
                  diagnostics=sorted(errors), paper_membership="unassigned")
    record["status"] = "complete_timing" if record["complete_timing"] and not errors else (
        "failed" if errors else "incomplete")
    return record


def read_execution_events(path: Path) -> dict:
    """Actual runtime stage records only; no inference from initial plans."""
    stages, durations = [], None
    with path.open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            if line.startswith("MAIN PROCESS: next iter plans: "):
                raw = line.split(": ", 2)[2]
            elif line.startswith("MAIN PROCESS: [") and not stages:
                raw = line[len("MAIN PROCESS: "):]
            else:
                raw = None
            if raw is not None:
                try:
                    stages.append(ast.literal_eval(raw))
                except (ValueError, SyntaxError):
                    stages.append({"unparsed_plan": raw.strip()})
            if line.startswith("all stage times: time lengths: "):
                durations = ast.literal_eval(line.split(": ", 2)[2])
    return {"source": str(path), "kind": "actual_runtime_events", "stages": stages,
            "durations_seconds": durations,
            "complete": bool(stages) and durations is not None and len(stages) == len(durations)
                and all(isinstance(stage, list) for stage in stages)
                and all(isinstance(t, (int, float)) and math.isfinite(t) and t >= 0 for t in durations)}


def full_metadata(path: Path) -> dict:
    """Scan numeric markers throughout a file, including its large middle section.

    mmap.find avoids decoding or retaining request text. This independent pass can
    enrich older bounded indexes without changing their source hashes.
    """
    size = path.stat().st_size
    if size == 0:
        return bounded_metadata(b'', b'')
    with path.open('rb') as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as data:
        lines = {}
        for marker in METRIC_MARKERS:
            start = 0
            while True:
                pos = data.find(marker, start)
                if pos < 0:
                    break
                end = data.find(b'\n', pos)
                if end < 0:
                    end = size
                # Plan strings can be long, but numeric diagnostic lines cannot
                # contain unbounded request payloads in this format.
                if end - pos <= 1024**2:
                    lines[pos] = data[pos:end]
                start = end + 1
        head = data[:min(size, 2 * 1024**2)]
    numeric = b'\n'.join(lines[k] for k in sorted(lines))
    result = bounded_metadata(head, numeric)
    result['metadata_scan'] = 'all_file_numeric_markers'
    updates = re.findall(rb'update_workload_dict min_cost:\s*([\d.eE+-]+)', numeric)
    if result['metrics'].get('initial_predicted_seconds') is not None:
        result['metrics']['incremental_predicted_seconds'] = result['metrics']['initial_predicted_seconds'] + sum(float(v) for v in updates)
    return result


def final_models_and_counters(path: Path) -> dict:
    """Recover composed Mixed models and tuple counters from the search prefix.

    Builders print component models before the final graph. The search header
    terminates this prefix, avoiding a full reread of GPU request traces.
    """
    result = {'model_paths': None, 'search_counters': {}}
    # MAX/MIN/LightRule logs may omit Total search time. Their first emitted
    # schedule (or runtime event) also ends the initial search prefix. Looking
    # only for Total search time would reread gigabytes of unrelated GPU text.
    boundaries = (b'Total search time:', b'Best group seq:', b'best_group_seq:', b'MAIN PROCESS:')
    with path.open('rb') as stream:
        for line in stream:
            if line.startswith(boundaries):
                break
            try:
                if line.startswith(b'real model_paths: '):
                    result['model_paths'] = ast.literal_eval(line.decode().split(': ', 1)[1])
                elif line.startswith(b'Cost_Query_Time, Cost_Query_num, '):
                    names, values = line.decode().split(': ', 1)
                    result['search_counters'].update(zip(names.split(', '), ast.literal_eval(values)))
            except (ValueError, SyntaxError, UnicodeError):
                pass
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=repo_root() / "benchmarks/test_end2end_schedule")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 4:
        parser.error("workers must be between 1 and 4")
    args.output.mkdir(parents=True, exist_ok=False)
    paths = sorted(p for p in args.directory.glob("*.log") if p.is_file() and not p.is_symlink())
    start, last = time.monotonic(), time.monotonic()
    records = []
    with (args.output / "index.jsonl").open("x") as output, ThreadPoolExecutor(args.workers) as pool:
        for record in pool.map(lambda p: index_log(p, repo_root()), paths):
            records.append(record)
            output.write(json.dumps(record, allow_nan=False) + "\n")
            output.flush()
            if time.monotonic() - last >= 20:
                print(f"indexed {len(records)}/{len(paths)} logs; {sum(r['size_bytes'] for r in records)/1e9:.1f} GB", flush=True)
                last = time.monotonic()
    counts = {status: sum(r["status"] == status for r in records) for status in ("complete_timing", "failed", "incomplete")}
    summary = {"schema_version": 1, "file_count": len(records), "total_bytes": sum(r["size_bytes"] for r in records),
               "status_counts": counts, "elapsed_seconds": time.monotonic() - start,
               "note": "No exact paper membership or exit-code success is inferred from complete timing alone."}
    atomic_write_json(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
