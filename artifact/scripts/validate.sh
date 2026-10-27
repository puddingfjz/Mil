#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd)"
temporary_root="$(mktemp -d "${TMPDIR:-/tmp}/mil-artifact-validate.XXXXXX")"
cleanup() {
    rm -rf -- "${temporary_root}"
}
trap cleanup EXIT

cd "${repo_root}"
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="${temporary_root}/matplotlib"
python_bin="${MIL_PYTHON:-python3}"

"${python_bin}" -m unittest discover -s artifact/tests -v
"${python_bin}" -m artifact.mil_artifact.run \
    --profile smoke --parameter ratio_seed=0 --dry-run >/dev/null
"${python_bin}" -m artifact.mil_artifact.paper_suite \
    --output "${temporary_root}/paper-plan" >/dev/null
"${python_bin}" -m artifact.mil_artifact.runtime_events \
    --output "${temporary_root}/runtime-events"

"${python_bin}" -m artifact.mil_artifact.resource_build build \
    --config artifact/configs/resources/simulated.yaml \
    --output "${temporary_root}/simulated-resources" >/dev/null
"${python_bin}" -m artifact.mil_artifact.resource_build plan \
    --config artifact/configs/resources/local_measurements.yaml \
    --output "${temporary_root}/resource-plan" >/dev/null

"${python_bin}" -m artifact.mil_artifact.verify_preparation \
    --output "${temporary_root}/cpu-bundle"
"${python_bin}" -m artifact.mil_artifact.release --json >/dev/null

echo "Artifact no-GPU validation passed."
