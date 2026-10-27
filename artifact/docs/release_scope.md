# Release scope

[简体中文](release_scope.zh-CN.md)

The release provides scheduling, cost simulation, runtime execution,
resource preparation, experiment configuration, result collection and plotting.
CPU and GPU execution interfaces are included. The automated checks cover the
CPU path and GPU entrypoint connections without executing GPU operations.
Target-hardware GPU build, inference and paper performance validation remain
unverified.

## Methods and runtime behavior

The method registry contains 24 IDs and their dispatch mappings. Ensembling,
Routing and Mixed use the LPT-backed Mil method; Chain Summary uses its distinct
greedy implementation. HybridFlow's `old` selector selects a separate active
variant. Runtime adjustment, preemption, output-length noise and MIN-R sampling
are configured per method and experiment.

There is no default outer experiment deadline. Saturn uses Gurobi with a
300-second limit for each MILP solve and requires a valid Gurobi license.

Single-replica workloads use the general per-replica simulator. Memory
accounting uses the model-size table. Scheduling and inference load fixed local
model snapshots described in [models and data](data_and_models.md).

The source contract pins executable structure, docstrings and special
directives. Numeric resources and native/build files use byte-level checks.
Copyright and license notices are included with the distributed sources.

## Distribution boundary

`manifests/source_distribution.yaml` explicitly lists included files. Packaging
does not traverse backups, follow source symlinks or include arbitrary untracked
files. Raw logs, external datasets, model weights and generated results are
outside this distribution. Packaging leaves the source workspace unchanged.

## Acceptance

Run `artifact/scripts/validate.sh` from the independent source distribution.
It checks configurations, numeric inputs, source contracts, CPU runtime events,
resource construction, plotting and failure cases. Representative real CPU
searches are a separate explicit command, described in the quickstart.
Passing these checks means CPU correctness/usability, not complete GPU
performance reproduction. See [coverage limits](paper_coverage.md).
