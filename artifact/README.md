# Mil Artifact

[简体中文](README.zh-CN.md)

Mil schedules applications composed of multiple LLMs. This release provides the
scheduler, cost simulator, modified vLLM runtime, running inputs, experiment
configurations, collection and plotting tools. It does not include experimental
logs, measured results, published figure data or historical CPU result baselines.

## Quick start

Follow the [CPU quickstart](docs/cpu_quickstart.md) to install the environment.
From the repository root:

```bash
bash artifact/scripts/validate.sh
python -m artifact.mil_artifact.validate_cpu \
  --cases chain/light_rule_star --workers 1 --output /tmp/mil-cpu-check-01
```

The first command checks inputs, source integrity, synthetic behavior and tooling.
The second explicitly runs the actual CPU scheduler and checks its output
structure, input identity and GPU guards. Neither runs GPU inference.
Neither compares results with undistributed historical measurements.
Omit `--cases` to run the eight representative CPU cases; there is no default
outer search deadline.

## Experiments and plotting

| Purpose | Entry |
| --- | --- |
| List paper experiments | `python -m artifact.mil_artifact.paper_suite --output NEW_DIR` |
| Plan GPU experiments | `bash artifact/scripts/run.sh --profile paper --dry-run` |
| Run GPU experiments | Follow [GPU installation and execution](env/README.md#gpu-runtime) |
| Build running resources | `bash artifact/scripts/build_resources.sh build --config CONFIG --output NEW_DIR` |
| List profiling commands without running them | Same resource entry with `plan` |
| Plot your collected measurements | `bash artifact/scripts/plot.sh results --run RUN_DIR --output NEW_DIR` |
| Render supplied paper-style chart data | `bash artifact/scripts/plot.sh chart --input CHART_JSON --output NEW_DIR` |
| Plot an external candidate or supporting-analysis bundle | `plot.sh candidates --snapshot BUNDLE --output NEW_DIR` or `plot.sh supporting --snapshot BUNDLE --output NEW_DIR` |
| Compare saved CPU schedules against an external baseline | `python -m artifact.mil_artifact.compare_schedules --help` |

Output directories must be new. The catalog contains 20 paper items, 72 unique
settings and 323 expanded scheduling tasks. These are configurations, not
measurements. Workload inputs are the model set, application type and request
set; hardware descriptions, cost tables and output-length distributions supply
the scheduling resources.

## Source and input layout

- `benchmarks/`: scheduling, simulation, runtime coordination, profiling and numeric resources.
- `vllm/`, `csrc/`: Mil-modified inference runtime and native extensions.
- `artifact/configs/`: experiments, methods, hardware and resource generation.
- `artifact/inputs/`: seven frozen request sets and model configurations.
- `artifact/expected/cpu/source_contract.json`: source signatures, not experiment results.
- `artifact/manifests/`: resource access, integrity and source-distribution specifications.

The runtime requires this repository's modified sources and, for GPU execution,
extensions built from those sources. The retained package label is `0.3.0`;
source and build checksums identify the implementation.

## Documentation

- [CPU quickstart](docs/cpu_quickstart.md) and [result formats](docs/result_schema.md)
- [Experiment scope](docs/paper_experiment_scope.md), [paper mapping](docs/paper_mapping.md) and [validation scope](docs/paper_coverage.md)
- [Resource generation](docs/resource_generation.md) and [models/data](docs/data_and_models.md)
- [Scheduler](docs/core_scheduler_guide.md) and [runtime](docs/vllm_runtime_guide.md)
- [Timing](docs/timing_convention.md), [troubleshooting](docs/troubleshooting.md) and [distribution](docs/publication.md)

Model/data terms and Saturn's Gurobi license are separate from the code license.
GPU execution interfaces are provided; CPU checks do not certify GPU performance.
