<a id="result-schema"></a>

# 结果格式

[English](result_schema.md)

每个任务目录包含：

- `metadata.json`：来源、执行协议 ID、任务参数、种子来源、命令、环境子集、时间戳和退出状态。
- `resolved_config.yaml`：精确展开后的任务配置。
- `command.txt`：经过 shell 转义、便于检查的命令。
- `stdout.attempt-N.log`：每次尝试不可变的合并 stdout／stderr 日志。
- `metrics.json`：解析后的指标及解析器诊断。

运行根目录包含 `run.json`、`inputs/` 快照和生成的 `metrics.jsonl`。
`run.json` 记录 Git 状态或源码包身份、主机／GPU 清单、资源根目录、GPU 拓扑输出及校验值，
以及每份复制配置、清单和运行时输入的 SHA-256。运行 ID 包含 Git 短提交号和合并输入摘要
的前 8 位。无 Git 时提交标签为 `nogit`，输入摘要包含内嵌源码快照。
每条 JSONL 记录使用 `schema_version: 1`，稳定字段如下：

```json
{
  "schema_version": 1,
  "parser_schema_version": 1,
  "run_id": "...",
  "experiment_id": "...",
  "task_id": "...",
  "profile": "main",
  "execution_protocol": "camera_ready_reconstruction_v1",
  "workload_id": "router",
  "method_id": "mil",
  "status": "succeeded",
  "comparison_to_control": null,
  "parameters": {},
  "seed_provenance": {
    "experiment_seed": 0,
    "workload_ratio_seed": 0,
    "legacy_test_id": 0,
    "python_hash_seed": "0"
  },
  "metrics": {
    "predicted_running_time_seconds": null,
    "real_running_time_seconds": null,
    "search_time_seconds": null,
    "preparation_time_seconds": null,
    "total_time_seconds": null,
    "completed_request_count": null,
    "completion_event_count": null,
    "failed_request_count": null,
    "gpu_idle_time_seconds": null,
    "gpu_capacity_time_seconds": null,
    "gpu_idle_ratio": null
  },
  "errors": [],
  "timed_out": false,
  "task_timeout_seconds": null,
  "git_commit": "...",
  "source_snapshot_sha256": null,
  "artifact_input_digest_sha256": "...",
  "task_metadata_checksum_sha256": "...",
  "resolved_config_checksum_sha256": "...",
  "source_log": "tasks/.../stdout.attempt-1.log",
  "source_checksum_sha256": "...",
  "source_size_bytes": 123
}
```

新运行记录必须包含 Git 提交或 `source_snapshot_sha256`。源码包运行的 `git_commit`
为 `null`，并填写快照哈希；验证时核对复制的 `inputs/artifact/source_snapshot.json`。
快照记录分发源码文件的哈希，在执行前与源码树核验。任务、配置和日志仍需各自的独立校验值。

`summary.json` 保存 `metrics.jsonl` 校验值，以及按顺序排列的日志路径、校验值和字节数。
重复收集会原子替换派生 JSON／JSONL；输入不变时输出字节也不变。

缺失值保留为 JSON `null`，不转换为零。即使进程正常退出，出现 traceback、已知致命诊断
或缺少必需计时字段时，也会收集为 `status: invalid` 并被验证拒绝。
负计时、成功记录附带非零退出码、总时间在数值精度内不等于执行时间加包含准备阶段的搜索时间，
也会失败。Preparation 是 search 的子项。每次基线比较报告实际值／期望值、绝对与相对误差、
配置容差和通过／失败判定。

执行后自动收集和验证。若所选 profile 没有适用的固定数值基线，会明确警告，不使用其他
profile 的期望值。验证拒绝空结果和重复任务 ID。`run.json` 声明任务列表时，记录必须精确
覆盖该列表，运行状态须为 `succeeded`；部分任务成功不足以让不完整或中断导出通过。
不含执行任务列表的历史参考导出仍受支持。
任何验证问题都会将 `run.json` 标记为失败，即使所有子进程均返回零。
恢复运行时，仅跳过退出码为零、重新解析无诊断且计时字段完整的任务；其他任务产生新的不可变尝试日志。

`run.json` 记录 `task_timeout_seconds`，包括显式 `null`。
配置中的正数限时或 `--task-timeout-seconds` 覆盖值在恢复运行时不可更改。
到期后仅向该任务拥有的进程组发送 SIGTERM，清理宽限期后发送 SIGKILL；
任务和尝试元数据记录 `timed_out: true` 与退出码 124。

消融变体的 `comparison_to_control` 指向稳定对照任务 ID，并记录精确的
`parameter_diff` 和 `legacy_diff`。对照任务与普通实验使用 `null`。
比较关系从版本化的方法和实验配置生成，不在执行后根据标签推断。

`total_time_seconds` 由实际运行时间加包含准备阶段的搜索时间得到；preparation 已包含在
search 中，见 `timing_convention.md`。总时间由日志分量重建，不单独解析最终总数字段。

`completed_request_count` 对 `finish one req` 事件的请求 ID 去重计数，
`completion_event_count` 保留原始事件数。GPU 空闲时间为日志
`non_computation_ranges` 内所有区间之和；容量时间为实际运行时间乘这些区间涉及的 GPU 数，
二者比值为 `gpu_idle_ratio`。历史日志没有权威的请求失败信号，
因此 `failed_request_count` 保持 `null`，不填零。

绘图只读取成功的结构化记录。图选择器固定非横轴参数，只在相同方法／坐标内聚合重复运行。
默认取中位数，单图可声明逐方法均值及 min/max 误差范围。坐标也可选择不同工作负载，
用于表达最终 Router／Mixed 组合消融，无需把数值复制进 Python 源码。
图可选择单项指标（如 `gpu_idle_ratio`），而非默认堆叠计时对。

`parameters.seed`、`seed_provenance` 和快照中的 `manifests/execution_protocol.yaml`
区分统一实验种子、`ratio_seed` 与非种子的历史 `test_id`。
统一种子控制 Python／NumPy 进程状态、PyTorch 模型进程状态、工作负载及输出长度采样。
未启用 GPU 确定性算法，因此记录的是随机数来源，不保证 GPU 计时逐位确定。

<a id="cpu-preparation-schemas"></a>

# CPU 输入与验证格式

正常输入的 CPU 路径使用 `schema_version: 1`，并显式标记 `kind` 或范围，
与上述历史 GPU 运行 JSONL 格式分开：

- `inputs/frozen/*.json`（`kind: normal_request_set`）：有序模型节点、输入长度／ID、
  依赖边、工作负载参数、可用的已知输出长度与来源哈希，不含原请求文本。
- CPU `result.json`：输入 SHA-256、方法／参数、资源哈希、导入模块路径、CPU worker 数、
  墙钟时间、GPU 保护状态和 `actual` 初始计划／阶段成本／总预测秒数。
  LightRule 未提供的初始成本为 null。`failure.json` 记录失败及显式请求的超时。
  `resource_hashes` 以 `cost_table`、`output_lengths`、`model_configurations` 为键；
  绝对路径 `resources` 映射兼容旧读取器。比较按资源角色与哈希匹配。

`compare_schedules --reference` 读取用户显式提供的 CPU 基线；`--sources`
读取用户提供的日志索引。两者均不把预测成本等同于 GPU 实测时间。
发布包不附带基线或来源索引，默认 CPU 验收只检查结构、身份与成本内部一致性。

外部 CPU 基线使用 schema 2、`kind: cpu_regression_reference`，记录请求身份、
参数、资源哈希、有序计划和可用成本；格式由 `cpu_reference.validate_references` 定义。
`tests/synthetic.py` 提供人工样例，不是历史测量。

`expected/cpu/source_contract.json` 使用 schema 1 和 `source_semantics`，
固定 Python 可执行结构、文档字符串、特殊指令及部分文件的字节哈希，
不包含调度结果。契约及输入哈希由 `manifests/cpu_preparation.yaml` 固定。
`verify_preparation` 输出 schema 3、checks/issues、`gpu_execution: false`、
`new_scheduling_searches: 0` 和 `historical_audits_read: false`。

## 显式数据绘图

`plot.sh results --run DIR --output NEW_DIR` 读取该运行的 `metrics.jsonl`，
按 `configs/figures.yaml` 选取成功测量并绘图。
`plot.sh chart --input JSON --output NEW_DIR` 读取 schema 1 的 `figures` 映射，
按 `published_plot.validate_chart` 验证图表规格后渲染。
`plot_manifest.json` 记录输入哈希、模式、输出及 GPU 未执行状态；
chart 模式另写 `plot_data.json`，记录实际渲染的坐标、柱高和 MIN-R 统计。
不提供论文数字、旧图片像素比较或隐式数据回退。

`candidates` 与 `supporting` 模式需要显式 `--snapshot` 分析包；
其 `results.json`、`analyses.json`、`timelines.json` 或 `supporting.json`
均来自用户自己的数据处理，不随源码分发。缺失测量不得作为成功的零值。
