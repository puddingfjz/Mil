# Result schema

[简体中文](result_schema.zh-CN.md)

Every task directory contains:

- `metadata.json`: provenance, execution-protocol ID, task parameters, seed
  provenance, command, environment subset, timestamps, and exit status;
- `resolved_config.yaml`: exact expanded task configuration;
- `command.txt`: shell-escaped command for inspection;
- `stdout.attempt-N.log`: immutable combined stdout/stderr for each attempt;
- `metrics.json`: parsed metrics and parser diagnostics.

The run root contains `run.json`, an `inputs/` snapshot, and a generated
`metrics.jsonl`. `run.json` records the Git state (or source-package identity), host/GPU inventory, resource
roots, GPU-topology output/checksum, and SHA-256 for every copied config,
manifest, and runtime input. The run ID includes the Git short commit and the
first eight characters of a combined Artifact-input digest. Without Git, the
commit label is `nogit`; the input digest includes the embedded source snapshot.
Each
JSONL record uses `schema_version: 1` and the following stable fields:

```json
{
  "schema_version": 1,
  "parser_schema_version": 1,
  "run_id": "...",
  "experiment_id": "...",
  "task_id": "...",
  "profile": "main",
  "execution_protocol": "camera_ready_reconstruction_v1",
  "workload_id": "router",
  "method_id": "mil",
  "status": "succeeded",
  "comparison_to_control": null,
  "parameters": {},
  "seed_provenance": {
    "experiment_seed": 0,
    "workload_ratio_seed": 0,
    "legacy_test_id": 0,
    "python_hash_seed": "0"
  },
  "metrics": {
    "predicted_running_time_seconds": null,
    "real_running_time_seconds": null,
    "search_time_seconds": null,
    "preparation_time_seconds": null,
    "total_time_seconds": null,
    "completed_request_count": null,
    "completion_event_count": null,
    "failed_request_count": null,
    "gpu_idle_time_seconds": null,
    "gpu_capacity_time_seconds": null,
    "gpu_idle_ratio": null
  },
  "errors": [],
  "timed_out": false,
  "task_timeout_seconds": null,
  "git_commit": "...",
  "source_snapshot_sha256": null,
  "artifact_input_digest_sha256": "...",
  "task_metadata_checksum_sha256": "...",
  "resolved_config_checksum_sha256": "...",
  "source_log": "tasks/.../stdout.attempt-1.log",
  "source_checksum_sha256": "...",
  "source_size_bytes": 123
}
```

New execution records require either a Git commit or `source_snapshot_sha256`.
For source-package runs, `git_commit` is `null`, the snapshot hash is populated,
and verification checks the copied `inputs/artifact/source_snapshot.json`
against it. The snapshot records the distributed source-file hashes; its contents
are checked against the source tree before execution. This is not a substitute
for the independent task, configuration and log checksums, which remain required.

`summary.json` includes the checksum of `metrics.jsonl` and a sorted list of
every source log path, checksum, and byte size. Re-running the collector
replaces derived JSON/JSONL atomically and produces the same bytes when its
inputs are unchanged.

Missing values remain JSON `null`; they are never converted to zero. A result
with a successful process exit but a traceback/known fatal diagnostic or
missing required timing fields is collected with `status: invalid` and rejected
by verification. Verification also rejects negative timings, nonzero
exit codes attached to successful records, and totals that do not equal
execution + inclusive search within numerical precision. Preparation is a subset of search. Each baseline
comparison reports actual/expected values, absolute and relative error, the
configured tolerance, and the pass/fail decision.

Execution automatically invokes collection and verification. A run whose
profile has no applicable frozen numerical baseline receives an explicit
warning rather than being compared against another profile's expectation.
Verification rejects empty results and duplicate task IDs. When `run.json`
declares an execution task list, the records must cover exactly that list and
the run must be marked `succeeded`; an incomplete/interrupted export cannot pass
merely because its available tasks succeeded. Historical reference exports
without an execution task list remain supported.
Any verification issue marks `run.json` failed even when every child process
returned zero. On resume, only a zero-exit task whose log reparses without
diagnostics and with all required timing fields is skipped; other tasks receive
a new immutable attempt log.

`run.json` records `task_timeout_seconds`, including explicit `null`. A
positive profile value or `--task-timeout-seconds` override is immutable across
resume. Expiration sends SIGTERM and, after the bounded cleanup grace period,
SIGKILL only to the task-owned process group; task and attempt metadata record
`timed_out: true` and exit code 124.

Ablation variants set `comparison_to_control` to the stable control task ID and
the exact `parameter_diff` and `legacy_diff`. Control tasks and ordinary
experiments use `null`; the comparison is generated from versioned method and
experiment configuration rather than inferred from labels after execution.

`total_time_seconds` is derived as real running time plus inclusive search time;
preparation is already included in search (see `timing_convention.md`). Total
is reconstructed from logged components, not independently parsed from a final
total field.

`completed_request_count` counts distinct request IDs in `finish one req`
events, while `completion_event_count` retains the raw event count. GPU idle
time is the sum of all intervals in the logged `non_computation_ranges`;
capacity time is real running time multiplied by the number of GPUs represented
by those ranges, and their quotient is `gpu_idle_ratio`. The legacy logs do not
emit an authoritative failed-request signal, so `failed_request_count` remains
`null` rather than being reported as zero.

Plotting reads only successful structured records. A figure selector fixes all
non-axis parameters and aggregates repetitions only within an identical
method/coordinate. The default is the median; individual figures may declare a
per-method mean and min/max error range. A coordinate may also select a
different workload, which is how the final combined Router/Mixed ablation is
represented without copying values into Python source. A figure may select one
metric (for example `gpu_idle_ratio`) instead of the default stacked timing
pair.

`parameters.seed`, `seed_provenance`, and the snapshotted
`manifests/execution_protocol.yaml` distinguish the unified experiment seed
from `ratio_seed` and the non-seed legacy `test_id`. The unified seed covers
Python/NumPy process state, PyTorch model-process state, workload sampling, and
output-length sampling. GPU deterministic algorithms are intentionally not
enabled, so this is RNG provenance rather than a claim of bitwise-deterministic
GPU timing.

# CPU preparation schemas

The normal-input CPU path uses `schema_version: 1` with an explicit `kind` or
scope, separately from the legacy GPU-run JSONL schema above:

- `inputs/frozen/*.json` (`kind: normal_request_set`) contains ordered model
  nodes, input lengths/IDs, dependency edges, workload parameters, available
  known output lengths and source hashes. There is no original request text.
- A CPU `result.json` records the input SHA-256, method/parameters, resource
  hashes, imported module paths, CPU worker count, wall time, GPU guard state
  and `actual` initial plans/stage costs/total predicted seconds. LightRule's
  unavailable initial cost is null. `failure.json` records failures and any
  explicitly requested timeout. `resource_hashes` is keyed by `cost_table`,
  `output_lengths` and `model_configurations`; the absolute-path `resources`
  map supports legacy readers. Comparisons match resources by role and hash.

`compare_schedules --reference` reads an explicitly supplied CPU baseline;
`--sources` reads a user-provided log index. Neither treats predicted costs as
measured GPU times. No baseline or source index is bundled. Default CPU acceptance
checks structure, identity and internal cost consistency.

External CPU baselines use schema 2 and `kind: cpu_regression_reference`, with
request identity, parameters, resource hashes, ordered plans and available costs.
`cpu_reference.validate_references` defines the schema; `tests/synthetic.py`
contains an artificial example, not historical measurements.

`expected/cpu/source_contract.json` uses schema 1 and `source_semantics`.
It pins executable Python structure, docstrings, special directives and selected
byte-level file hashes; it does not contain schedule outputs.
`manifests/cpu_preparation.yaml` pins this contract and the inputs.
`verify_preparation` writes schema 3 with checks/issues, `gpu_execution: false`,
`new_scheduling_searches: 0` and `historical_audits_read: false`.

## Explicit-data plotting

`plot.sh results --run DIR --output NEW_DIR` reads that run's `metrics.jsonl`
and plots successful records selected by `configs/figures.yaml`.
`plot.sh chart --input JSON --output NEW_DIR` reads a schema-1 `figures` mapping,
validates specifications with `published_plot.validate_chart`, then renders.
`plot_manifest.json` records input hashes, mode, outputs and no-GPU status.
Chart mode also writes `plot_data.json` with rendered coordinates, bar heights
and MIN-R statistics. There is no paper-number or historical-pixel comparison,
nor an implicit result fallback.

`candidates` and `supporting` require an explicit `--snapshot` analysis bundle.
Its `results.json`, `analyses.json`, `timelines.json` or `supporting.json`
must be generated from the user's data; none is distributed with the source.
Missing measurements must not be treated as successful zero values.
