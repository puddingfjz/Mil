# Environments

[简体中文](README.zh-CN.md)

The primary installation route is native Conda. Docker is optional; CPU
validation and the native GPU workflow do not require it.

## Required vLLM runtime

Mil requires the modified vLLM Python and CUDA sources included in this
repository. CPU scheduling imports the local Python sources; GPU execution
requires building the native extensions from the same source tree.

`runtime.yaml` lists this custom runtime under `custom_vllm`, separately from
standard package dependencies. The package version is `0.3.0`; source checksums
and the local build record identify the implementation.

## Tooling and plotting

```bash
conda env create --file artifact/env/environment.yml
conda activate mil-artifact
bash artifact/scripts/validate.sh
```

The lightweight environment and `requirements-lock.txt` pin configuration,
numeric analysis and plotting dependencies. Rendering uses Matplotlib 3.9.4, NumPy 1.26.3 and Pillow 11.3.0.
Plots require explicit data; no paper values or historical pixel references are bundled.

## Actual CPU scheduling

For the frozen numeric workloads on Linux x86-64, create a fresh environment
and install the CPU-only PyTorch wheel before the scheduling requirements:

```bash
conda env create --name mil-artifact-cpu --file artifact/env/environment.yml
conda activate mil-artifact-cpu
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=""
python -m pip install --no-cache-dir --no-deps \
  --index-url https://download.pytorch.org/whl/cpu torch==2.1.2
python -m pip install --no-cache-dir --index-url https://pypi.org/simple \
  -r artifact/env/requirements-cpu.txt
python -m pip check
python -c 'import torch; assert torch.version.cuda is None; print(torch.__version__)'
bash artifact/scripts/validate.sh
python -m artifact.mil_artifact.validate_cpu \
  --cases ensemble/mil chain/light_rule_star --workers 1 \
  --output /tmp/mil-clean-cpu-check-01
```

The [official PyTorch CPU wheel index](https://pytorch.org/get-started/previous-versions/#v212)
provides `torch==2.1.2+cpu`. `requirements-cpu.txt` uses `runtime-freeze.txt` as
version constraints, not as an installation list: CUDA libraries, xFormers and
Triton are not installed. Do not install the repository as an editable GPU
package or build its CUDA extension for this route. Run from the source root.

The launcher forbids GPU operators, reconstructs model configurations from the
numeric fixtures and starts a private local Ray CPU service. No model weights,
external model cache or Gurobi license are needed for these two cases. Dataset
preparation and Saturn require additional inputs and dependencies and are not
covered by these two cases. See [CPU quickstart](../docs/cpu_quickstart.md).

Use a new output directory for each run. An existing environment can be used
for routine work, but is not evidence of a fresh installation. `runtime.yaml`
records CPU acceptance separately from pending GPU validation.

## GPU runtime

On a compatible host with CUDA 12.1, prepare a separate environment:

```bash
conda create --yes --name mil-artifact-runtime \
  python=3.9.20 pip=24.2 setuptools=75.1.0 wheel=0.44.0
conda activate mil-artifact-runtime
unset PYTHONPATH PYTHONHOME MIL_CPU_SCHEDULING VLLM_USE_PRECOMPILED PYTORCH_NVCC
export PYTHONNOUSERSITE=1
python -m pip install --no-cache-dir -r artifact/env/runtime-freeze.txt
# Use the actual toolkit directory, not a symlink to another installation.
export CUDA_HOME=/usr/local/cuda-12.1
export TORCH_CUDA_ARCH_LIST=8.0
# Choose a NEW directory; retain it while this editable installation is in use.
python -m artifact.mil_artifact.package --output /tmp/mil-runtime-source-01
cd /tmp/mil-runtime-source-01/source
python -m artifact.mil_artifact.runtime_identity build
python -m artifact.mil_artifact.runtime_identity verify
bash artifact/scripts/doctor.sh --profile smoke
```

For GPU execution, build the extension from the packaged Mil sources using the
command above. CPU validation does not cover GPU compilation or inference.

The explicit build entry runs a local editable installation with dependency
downloads disabled. It refuses old binaries/build directories and the
precompiled-extension bypass. It records the exact source hashes, compiler and
CUDA target, Python/PyTorch ABI, actual module origins and native binary hashes
in `artifact/results/runtime-build/build.json`, only after a successful build.
Compilation/import checks hide GPU devices and do not execute inference.
Keep using this new source directory for GPU preflight and experiments; the
original checkout and its existing compiled files are not overwritten.
The package embeds `artifact/source_snapshot.json`; preflight verifies its
source hashes instead of requiring `.git`. Keep this file with an extracted
archive. Build outputs and new results do not change the source inventory.

After preparing the external datasets and pinned model snapshots described in
[models and data](../docs/data_and_models.md), select and inspect a task before
explicit execution, for example:

```bash
bash artifact/scripts/run.sh --profile smoke --suite router --method mil --dry-run
bash artifact/scripts/run.sh --profile smoke --suite router --method mil \
  --execute --allow-provisional
```

The same entry accepts `main`, `ablation`, `scalability`, `reference`, `full`
and `paper`. GPU build and inference validation are pending for this release.
`--allow-provisional` acknowledges unvalidated measurements; it
does not bypass source/build, resource, hardware or selected-solver preflight.
Method, workload and parameter filters also apply to preflight, so selecting
only Mil does not require a Saturn license. The runner normalizes its Python
search path to this source and disables user-site/CPU-shim imports for both
preflight and actual tasks. GPU visibility is set from the selected hardware.

Verification rejects missing records, changed sources/binaries, foreign import
paths and mismatched runtime environments. The GPU doctor performs this check
before running the inference entrypoint or hardware/solver probes. This is local
provenance, not a signed attestation; editing both records and files can defeat
it. Successful provenance checks do not certify CUDA operator correctness or
paper performance. The release marks actual GPU build/inference validation
as pending. CPU-only users do not run the GPU build/verify commands.

Model weights, datasets and their access terms are external requirements.
Saturn uses PuLP 3.3.0/Gurobi 12.0.2 and needs a valid user-provided
license, selected through `GRB_LICENSE_FILE` where applicable. Never include
license secrets in Git or logs, and do not replace Gurobi with another solver.

The optional container recipe is `artifact/env/Dockerfile`; its base digest and
build parameters are pinned in `runtime.yaml`. Build its context only from a
freshly packaged source tree. The container build and GPU execution are unvalidated.
Keep `.dockerignore` exclusions
for local logs and backups. Shell entrypoints honor `MIL_PYTHON` when explicitly
set to the intended Python executable.
