# CPU quickstart

[简体中文](cpu_quickstart.zh-CN.md)

Run commands from the repository root. Each output path must be new. These
commands do not run GPU inference or require Docker. No prior experimental
results are distributed.

## 1. Install and check the tools

Use the [lightweight environment](../env/README.md#tooling-and-plotting) for
configuration, resource and plotting checks. Real scheduling in Section 2
requires the [CPU scheduling environment](../env/README.md#actual-cpu-scheduling).

```bash
bash artifact/scripts/validate.sh
python -m artifact.mil_artifact.paper_suite --output /tmp/mil-paper-plan-01
```

Validation checks source and input integrity, synthetic runtime events, parsers,
resource construction, plots and failure handling. It does not start new
scheduling searches. The paper plan lists 20 items, 72 settings and 323 tasks;
it does not execute them. The 512-token Ensembling extension is separate from
the 256-token paper setting.

## 2. Run the actual CPU scheduler

Frozen request sets contain model sets, application dependencies, request IDs,
token lengths and resource hashes, not raw request text or execution results.

| Input under `artifact/inputs/frozen/` | Workload |
| --- | --- |
| `ensemble.json` | 9 models, 5000 requests, maximum output length 256 |
| `routing.json` | 5 models, 6856 routed requests, unknown output lengths |
| `chain.json` | 100 documents, 4 evaluations, maximum output 900 |
| `mixed.json` | Chain workload plus 5000 Ensembling requests |
| `oracle.json`, `exact.json`, `rand.json` | 1000-request Oracle, Exact and Rand inputs |

`inputs/model_configurations.json` provides metadata for 14 models without
weights. CPU scheduling uses CPU-only PyTorch, Transformers and local Ray
workers. GPU operators are forbidden; model weights and a populated Hugging
Face cache are not required.

```bash
python -m artifact.mil_artifact.cpu_schedule run \
  --requests artifact/inputs/frozen/ensemble.json \
  --method mil --cpus 2 --output /tmp/mil-cpu-ensemble-01

python -m artifact.mil_artifact.validate_cpu \
  --cases chain/light_rule_star --workers 1 --output /tmp/mil-cpu-check-01
```

The first entry writes `result.json` with the plan, predicted costs, parameters,
seeds and input/resource/source hashes. The second runs a selected case and
checks completion, method identity, input hashes, plan structure, stage-cost
consistency, source identity and GPU guards. It does **not** claim agreement
with historical results. Omit `--cases` to run all eight representative cases.
CPU cost predictions are not measured GPU runtimes.

Use `cpu_batch --help` for repeated `--requests` and `--method` options.
Single and batch searches have no outer time limit by default. An explicit
`--timeout SECONDS` is a diagnostic override, not a paper failure criterion.
Saturn* retains the algorithm's per-MILP 300-second limit and requires a valid
Gurobi license; no alternative solver is substituted. Ordinary LightRule does
not return a predicted initial-plan cost; a null cost is not a zero measurement.

Optional comparisons require your own baseline:
`validate_cpu --reference FILE` accepts a schema-2 CPU reference.
`--history-root ACTIVE_REPO --sources INDEX_JSONL` additionally compares eligible
CPU stages from explicitly supplied logs, including CPU-only records.
Both history options must be supplied together. Never point them at a backup.
A historical mismatch is evidence of a difference, not by itself proof of an
algorithm error. File dates, method names and equal outputs do not identify the
generating code version.

## 3. Rebuild running inputs

Acquire raw data and tokenizer snapshots using the
[models/data guide](data_and_models.md), then:

```bash
python -m artifact.mil_artifact.cpu_schedule prepare \
  --spec artifact/configs/inputs/routing.yaml \
  --output /tmp/mil-prepared-routing-01
```

Preparation calls the original workload builders without loading weights.
Configuration fields include `application`, `model_set`, `parameters`,
`benchmark_inputs`, `data_root` and explicit `known_output_lengths`.
Unknown downstream lengths are cost estimates. Known-length settings fail if
the required input records are absent.

## 4. Resources and runtime logic

```bash
bash artifact/scripts/build_resources.sh build \
  --config artifact/configs/resources/simulated.yaml \
  --output /tmp/mil-synthetic-resources-01
python -m artifact.mil_artifact.runtime_events --output /tmp/mil-runtime-events-01
```

Synthetic inputs check formats and algorithms, not paper performance.
Six artificial completion-event cases call the real runtime adjustment and
placement functions. For measured resource acquisition and offline conversion,
see [resource generation](resource_generation.md).
Scheduling accepts `--cost-table`, `--distribution` and `--model-configs`
overrides without replacing the default files.

## 5. Plot your results

After running and collecting an experiment:

```bash
bash artifact/scripts/plot.sh results \
  --run /path/to/your/run --output /tmp/mil-plots-01
```

This reads `metrics.jsonl` from the supplied run. Repeat `--figure` to select
IDs from `configs/figures.yaml`, for example `llm_blender_256`.
There is no cached result fallback.

The `chart` mode accepts your own paper-style chart JSON:
`{"schema_version": 1, "figures": {"figure_id": chart_spec}}`.
`published_plot.validate_chart` defines the chart schema; the small artificial
example in `tests/synthetic.py` demonstrates it. This mode does not compare
against undistributed paper numbers or pixel references.

```bash
bash artifact/scripts/plot.sh chart \
  --input /path/to/your/charts.json --output /tmp/mil-charts-01
bash artifact/scripts/plot.sh candidates \
  --snapshot /path/to/your/candidate-bundle --output /tmp/mil-candidates-01
bash artifact/scripts/plot.sh supporting \
  --snapshot /path/to/your/supporting-bundle --output /tmp/mil-supporting-01
```

Candidate/supporting bundles must be explicitly generated from your own inputs.
Their plotting interfaces do not certify membership in the paper's original
sample. Output directories must be new; existing files, backup paths, source
directories and ordinary symlinks are rejected. Within the repository use
`artifact/results/` or `results/`. Shell entries accept `MIL_PYTHON`.

## 6. Compare saved outputs

`compare_schedules` compares existing CPU outputs with an explicit external
reference or log index, without rerunning search. `scheduling_replay` is a
separate log-based diagnostic, not the normal-input scheduling route.
See [replay](scheduling_replay.md), [result schema](result_schema.md) and
[timing conventions](timing_convention.md).
