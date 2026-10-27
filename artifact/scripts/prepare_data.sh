#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd)"
cd "${repo_root}"

python_bin="${MIL_PYTHON:-python3}"
exec "${python_bin}" -m artifact.mil_artifact.prepare data "$@"
