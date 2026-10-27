# Troubleshooting

[简体中文](troubleshooting.zh-CN.md)

## `doctor` reports unresolved paths

Set `MIL_DATA_ROOT` and, when needed, `MIL_MODEL_ROOT`. Do not edit paths into
Python source or commit machine-local locations.

## GPU name or count does not match

Use the hardware profile corresponding to the machine. The default is to stop.
`--allow-profile-mismatch` downgrades GPU count/model/memory mismatches only;
the run metadata is conservatively marked non-comparable. A missing or broken
NVIDIA driver remains an error and cannot be bypassed by this flag.

If `nvidia-smi` fails in a restricted execution environment, verify device
access permissions and compare its output with a check on the host. A device
visibility problem in that environment does not by itself identify a host
driver failure.

## A GPU is locked

Another Artifact run using the same result root owns one or more requested
devices. Use `--gpu-lock-timeout SECONDS` to wait, or choose a disjoint hardware
profile/result root according to the site's coordination policy. Lock release
is handled by the kernel when the runner exits. Do not delete another live
run's lock or kill unrelated GPU processes.

## `doctor` reports a disk-space error

The checker compares filesystem capacity/free bytes against the provisional
storage policy in the hardware profile. Missing manifest resources are added to
the repository, data, or model free-space requirement with 5% download/staging
overhead. Results use the expanded profile task count and a 400 MB-per-task
estimate derived from 1,762 historical logs with a 2.05 safety factor. Put
`MIL_RESULTS_ROOT` on a larger filesystem or prepare verified resources first;
use storage capacity appropriate for the selected task matrix. The result-size
estimate is provisional; monitor actual usage during large runs.

## `doctor` reports a Gurobi license failure

Profiles containing Saturn* require the frozen `gurobipy`/PuLP packages and a
working Gurobi license. Provision the license according to your institution's
Gurobi setup, then rerun `doctor` with the intended profile. Do not switch to
CBC: the historical caller explicitly selects Gurobi, and changing the solver
would change baseline semantics. Profiles without Saturn* do not perform this
license check.

Select your effective license with `GRB_LICENSE_FILE` when required by the
local installation. Keep credentials out of logs and tracked files. Mil-only
CPU scheduling does not require Gurobi.

## Docker is unavailable

Use the native Conda procedure in `artifact/env/README.md`. Docker is optional;
the native installation and validation commands do not require a Docker daemon.

## A method is blocked

The method mapping is incomplete or the current dispatcher lacks that method.
Do not substitute another method under the same paper label. Update
`paper_mapping.md`, implement/recover the exact variant, and add a regression
case first.

## A run was interrupted

Use `--resume <run-directory>`. Successful task directories are preserved;
failed tasks receive another attempt record and their old logs are retained.
Resume fails closed if the Git commit, Artifact-input digest, expanded task
matrix, profile, Python executable, or hardware-mismatch policy differs from
the original run. A task is placed in its own process group, so SIGINT/SIGTERM
cleanup is limited to that task. Start a new run instead of mixing provenance
when any immutable input changes.

## A task exceeds its wall-clock timeout

The runner records `timed_out: true`, uses exit code 124, and terminates only
the current task process group. Fix the cause or start a new run with a larger
positive `--task-timeout-seconds`; resume deliberately rejects a changed
timeout because it would mix failure policies inside one run. Profiles declare
`task_timeout_seconds: null` by default, so no automatic timeout occurs unless
explicitly requested.

## A zero-exit task is reported invalid

The legacy process can return zero after printing an internal failure. The
collector therefore requires execution, search, preparation, and total timing
fields and rejects tracebacks, assertions, and CUDA out-of-memory diagnostics.
Inspect the immutable attempt log, correct the underlying problem, then use
`--resume <run-directory>`; the invalid task is reparsed and attempted again.
Do not edit the log or replace the missing value with zero.

## Repeated runs differ despite the same seed

Check `metadata.json` under `seed_provenance` for `experiment_seed`,
`workload_ratio_seed`, and `legacy_test_id`; they have different roles. The
unified experiment seed
controls process RNGs and sampling, but deterministic GPU algorithms are not
forced because they could change performance semantics. Exact bitwise GPU
results are therefore not promised. Record timing dispersion and use the
aggregation declared by `configs/figures.yaml`.

## Legacy files appear to be changing

Stop the run. Artifact execution should use its staged working directory. Check
`metadata.json` for `working_directory`; it must be inside the selected results
root, not `benchmarks/`.
