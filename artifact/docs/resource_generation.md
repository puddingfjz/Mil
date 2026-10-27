<a id="成本表与输出长度分布参数化入口与纯-cpu-验证"></a>

# Cost tables and output-length distributions: parameterized generation and CPU-only validation

[简体中文](resource_generation.zh-CN.md)

This entry generates cost tables and output-length distributions from
measurement logs. It does not start GPUs, load weights, connect to Ray or
download data/models. It requires Python 3.9+, NumPy and PyYAML; the artifact
environment is suitable. Implementation: `artifact/mil_artifact/resource_build.py`;
shell entry: `artifact/scripts/build_resources.sh`.

<a id="两个动作"></a>

## Two actions

- `plan`: generate an acquisition list with models, arguments, required GPU
  counts and new log locations. It writes JSON only and executes no commands;
  there is no `--execute` option.
- `build`: read existing measurement logs and compute tables/distributions on
  CPU, producing standalone resource files without GPU initialization or source edits.

`--config` selects YAML/JSON configuration; `--input-root` resolves relative
input paths (repository root by default); repeated `--model` selects a subset
by full model ID. `--output` must be new. Existing directories and symlinks are
rejected; overwrite and append are not supported.

Run all commands from the repository root and choose a new output path each time.

<a id="1-不需要模型gpu-或历史日志的模拟测试"></a>

## 1. Synthetic tests without models, GPUs or historical logs

```bash
bash artifact/scripts/build_resources.sh build \
  --config artifact/configs/resources/simulated.yaml \
  --output /tmp/mil-resources-simulation-01
```

This configuration reads small versioned fixtures simulating profiling records
and request lengths. All measurements are test data, not paper results. It tests
parsing, repeated-measurement selection, table serialization, truncated requests
and distribution generation. Unit tests compare results with the original
`CostTable` CPU calculations.

<a id="2-从保留的测量日志重建资源"></a>

## 2. Rebuild resources from retained measurement logs

```bash
bash artifact/scripts/build_resources.sh build \
  --config artifact/configs/resources/local_measurements.yaml \
  --output /tmp/mil-resources-from-logs-01
```

This example covers four cost profiles: TP=1/2 for ChatGLM3-6B and StableLM
tuned alpha 7B, plus their temperature=1 No Robots output-length records.
Original logs are under `benchmarks/Cost_Model_per_iter_zxcpu/` and
`benchmarks/collect_output_lengths/no_robot/` in the active repository.
They are not required for clean-clone tests. Missing logs cause an explicit
error; acquisition is not started automatically.

Generated files:

| File | Purpose |
| --- | --- |
| `cost_table.py` | Complete serialized data compatible with `get_cost_table_from_serialized_data` |
| `output_lengths.py` | Distributions compatible with the sampler's `pdf_dict` key conventions |
| `report.json` | Resolved configuration, input/output SHA-256, Python/NumPy versions, rebuilt entries, inherited counts and comparison results |

The example explicitly selects committed tables/distributions as `base_*`.
Only selected entries are rebuilt; others inherit base data. Reports distinguish
rebuilt and inherited counts. Existing source resources and the scheduler's
default resource choices are not replaced automatically.

Use `read_cost_table(path)` and `read_literal_module(path, ("pdf_dict",))`
to read outputs on CPU. They parse allowed literals/update statements only,
without executing resource Python code or triggering vLLM imports in serialized modules.

Process only one model:

```bash
bash artifact/scripts/build_resources.sh build \
  --config artifact/configs/resources/local_measurements.yaml \
  --model THUDM/chatglm3-6b \
  --output /tmp/mil-resources-chatglm-01
```

<a id="3-生成采集任务清单"></a>

## 3. Generate an acquisition plan

```bash
bash artifact/scripts/build_resources.sh plan \
  --config artifact/configs/resources/local_measurements.yaml \
  --output /tmp/mil-resource-plan-01
```

`acquisition.json` stores argv arrays, working directories, GPU counts and
target log paths; all tasks are `executed: false`. Targets are under
`measurements/` in the new output directory, not historical logs.
`build_config.json` points to these targets for use with `build` after acquisition.
`plan` does not create or start model processes.

The original profiler fixes two measurements per point, a 1 ms threshold and
some sampling parameters. `plan` rejects acquisition settings the profiler
cannot express rather than silently ignoring them. `build` can interpret
other existing records with `profile_num` and `threshold_seconds`.

<a id="配置约定"></a>

## Configuration conventions

Each `cost.entries` item declares:

- `model`, `log`, `exec_plan: [tp, gpu_ratio, weight_load_degree, cache_gpu_num]`;
- `sample_config`, default `[1, 1, -1, 0]`;
- `profile_num` default 2, `skip_records` default 2 and `threshold_seconds` default 0.001;
- `model_config: [hidden_size, num_layers]`, `flops_coefficient` and `prepare_seconds`,
  supplied directly or read from explicit base/coefficient tables.

`cost.base_table` and `cost.coefficient_table` select existing resources.
`cost.compare_table` optionally selects an independent reference to compare
six cost categories of rebuilt entries. Missing model configurations fail
without contacting model services or loading Hugging Face models.

`output_lengths.entries` declares model, log, `max_model_len`, temperature and counts:

- `num_prompts` is the acquisition request limit; `expected_requests` is the
  expected retained count after filtering. `sample_requests` filters very short
  or long inputs: this example's 10000-request limit yields 9907 ChatGLM and 9889 StableLM records.
- Multiple `output_lens` blocks require an explicit zero-based `record_block`;
  the first/last block is not selected automatically.
- Context-limit outputs retain the original truncation handling, with remaining
  probability assigned to the final length. Temperature 1 uses a model-short-name
  key; other temperatures use `(short_name, temperature)`.
- `base_distribution` preserves unrebuilt entries; `compare_distribution`
  compares selected entry probabilities.

Reconstruction failures or mismatches return nonzero. Mismatches retain a failed
report; format errors, missing inputs and similar construction errors stop
before creating the result directory.

<a id="验证与资源范围"></a>

## Validation and resource scope

`artifact/scripts/validate.sh` includes synthetic builds, serialization comparisons
and acquisition-plan-only checks, without historical logs, weights or GPUs.

The default zxcpu table has 26 fixed entries covering 14 models. Recomputed
and inherited entries are distinguished; offline equality does not constitute
new GPU measurements. `configs/resources/additional_measurements.yaml` lists
optional other-host measurements, which do not automatically replace the default
table. Lengths exceeding model windows are rejected, not clipped to pass.
CPU scheduling accepts explicit `--cost-table` and `--distribution` and records hashes.
