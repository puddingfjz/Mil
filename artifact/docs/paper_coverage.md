# Experiment coverage and validation limits

[简体中文](paper_coverage.zh-CN.md)

Configuration coverage, functional correctness and measured GPU performance are
different claims. This distribution includes code, running inputs and commands,
not prior experimental results.

## Configurations and inputs

The catalog contains 20 paper items, 72 unique settings, 301 setting/method cells
and 323 expanded tasks. It includes the 512-token extension and an optional
MIN-R sample-size scan. Twenty-four method IDs include aliases; thirteen are
used by the paper matrix. `run_i` labels repetitions, not random seeds.

Seven normal request sets, 14 model configurations, cost tables and output-length
distributions support scheduling. These are running inputs, not end-to-end
experiment outputs. No paper figure arrays, measured timing tables, log indexes,
historical schedules or result baselines are included.

## Meaning of a pass

`validate.sh` checks source/input integrity and synthetic functionality without
new scheduling searches. `validate_cpu` explicitly executes representative
scheduling cases and checks completion, method and input identity, schedule
structure, cost consistency, source paths and GPU guards.

These checks do not claim agreement with old experimental outputs. External
baselines or logs can be provided explicitly for comparisons. CPU-only logs can
contain valid completed schedules even without GPU timing. Method labels,
modification time and matching outputs do not establish source-version identity.

GPU execution paths are included, but CPU tests do not certify target-hardware
builds, inference, solver licenses, full-search-matrix correctness or performance.
Use the [environment guide](../env/README.md#gpu-runtime) to run GPU experiments
and the [quickstart](cpu_quickstart.md) to plot newly collected measurements.
