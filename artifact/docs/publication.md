# Source distribution

[简体中文](publication.zh-CN.md)

The `codex/release` publication contains source code, running inputs, tests and
experiment instructions, without historical experimental results.

## Build and verify

From the repository root in the documented Python environment:

```bash
python -m artifact.mil_artifact.package --output /tmp/mil-release-01
cd /tmp/mil-release-01/source
bash artifact/scripts/validate.sh
```

The new output directory contains `source/`, `mil-source.tar.gz` and
`manifest.json` with file/archive checksums.
`artifact/manifests/source_distribution.yaml` is the explicit file list;
packaging does not recursively copy the working tree or visit backup directories.

A generated `artifact/source_snapshot.json` inside the source package lets
preflight and execution verify sources without Git. It excludes its own hash.
Changed or missing sources fail verification. The runner copies this identity
record to its run inputs and carries its hash through collection and verification.

## Contents and access

Cost tables, output-length distributions, numeric request sets, known-output-length
inputs and model configurations are retained. Prior logs, historical result
indexes, CPU schedule baselines, performance tables, plotted numbers and result
images are not included. Synthetic unit-test fixtures are artificial data, not
reported measurements. Source-integrity signatures are retained.

Private datasets, weights, credentials, Git metadata, backups and generated
results are outside the source archive. Raw data, model weights and new profiling
measurements must be obtained under their own terms. See [models/data](data_and_models.md)
and [resource generation](resource_generation.md). Saturn* requires Gurobi
authorization and does not silently substitute another solver.

The published release history starts from a source-only root commit.
Other branches or separately retained Git objects are not part of this release.
For a result-free checkout, select `codex/release` explicitly; repository-default
branches are independent.

## Validation scope

`cpu_preparation.yaml` and `release.yaml` describe source/input checks and
functional testing, not historical performance certification. Plotting requires
explicit measurements from the user. For isolation checks, use an independent
source package in a fresh Conda environment without user-site packages or
preinstalled vLLM. See [CPU quickstart](cpu_quickstart.md) and
[GPU installation](../env/README.md#gpu-runtime).
