# CPU 快速开始

[English](cpu_quickstart.md)

所有命令在仓库根目录执行，输出路径必须是新路径。这些命令不运行 GPU 推理，
不需要 Docker。发布版不分发历史实验结果。

## 1. 安装并检查工具

配置、资源和绘图检查使用 [轻量环境](../env/README.zh-CN.md#tooling-and-plotting)；
第 2 节真实调度需要 [CPU 调度环境](../env/README.zh-CN.md#actual-cpu-scheduling)。

```bash
bash artifact/scripts/validate.sh
python -m artifact.mil_artifact.paper_suite --output /tmp/mil-paper-plan-01
```

验证检查源码及输入完整性、合成运行时事件、解析器、资源构造、绘图和错误处理，
不启动新的调度搜索。论文计划列出 20 个项目、72 个设置和 323 个任务而不执行它们。
512-token Ensembling 扩展与论文的 256-token 设置分开。

## 2. 运行真实 CPU 调度器

冻结请求集包含模型集合、应用依赖、请求 ID、token 长度与资源哈希，
不包含请求原文或执行结果。

| `artifact/inputs/frozen/` 中的输入 | 工作负载 |
| --- | --- |
| `ensemble.json` | 9 模型、5000 请求、最大输出 256 |
| `routing.json` | 5 模型、6856 个路由后请求、未知输出长度 |
| `chain.json` | 100 文档、4 次评价、最大输出 900 |
| `mixed.json` | Chain 工作负载加 5000 个 Ensembling 请求 |
| `oracle.json`、`exact.json`、`rand.json` | 1000 请求的 Oracle、Exact、Rand 输入 |

`inputs/model_configurations.json` 提供 14 个模型的配置，不含权重。
CPU 调度使用 CPU-only PyTorch、Transformers 和本地 Ray worker，
禁止 GPU 算子，不需要模型权重或预先填充的 Hugging Face 缓存。

```bash
python -m artifact.mil_artifact.cpu_schedule run \
  --requests artifact/inputs/frozen/ensemble.json \
  --method mil --cpus 2 --output /tmp/mil-cpu-ensemble-01

python -m artifact.mil_artifact.validate_cpu \
  --cases chain/light_rule_star --workers 1 --output /tmp/mil-cpu-check-01
```

第一个入口生成 `result.json`，记录计划、预测成本、参数、种子及输入/资源/源码哈希。
第二个入口运行所选用例，检查完成状态、方法身份、输入哈希、计划结构、阶段成本
一致性、源码身份及 GPU 保护，**不宣称与历史结果一致**。省略 `--cases` 可运行
全部八个代表用例。CPU 预测成本不是 GPU 实测时间。

`cpu_batch --help` 介绍批量的 `--requests`、`--method` 选项。
单项和批量搜索默认没有外层时间限制，显式 `--timeout SECONDS` 只是诊断选项，
不是论文失败判据。Saturn* 保留算法原有的每个 MILP 300 秒限制，需要有效的
Gurobi 授权，不自动替换求解器。普通 LightRule 不返回初始计划预测成本，
null 不能解释为零秒测量。

可选对比需要自行提供基线：`validate_cpu --reference FILE` 接受 schema 2 的
CPU 参考；同时提供 `--history-root ACTIVE_REPO --sources INDEX_JSONL` 可对比
指定日志中的合法 CPU 阶段，包括仅运行 CPU 的记录。两个历史选项必须一起提供，
不得指向备份目录。历史不一致说明存在差异，不直接证明算法错误；文件日期、
方法名称或结果相同均不能证明生成源码版本相同。

## 3. 重建运行输入

按 [模型与数据指南](data_and_models.zh-CN.md) 获取数据和 tokenizer 快照，再执行：

```bash
python -m artifact.mil_artifact.cpu_schedule prepare \
  --spec artifact/configs/inputs/routing.yaml \
  --output /tmp/mil-prepared-routing-01
```

准备过程调用原始工作负载构造器，不加载权重。配置字段包括 `application`、
`model_set`、`parameters`、`benchmark_inputs`、`data_root` 和显式
`known_output_lengths`。未知下游长度是成本估计；已知长度设置缺失必要输入时
会明确报错。

## 4. 资源与运行时逻辑

```bash
bash artifact/scripts/build_resources.sh build \
  --config artifact/configs/resources/simulated.yaml \
  --output /tmp/mil-synthetic-resources-01
python -m artifact.mil_artifact.runtime_events --output /tmp/mil-runtime-events-01
```

合成输入检查格式和算法，不代表论文性能。六个合成完成事件用例调用真实运行时
调整及放置函数。实测资源获取和离线转换见 [资源生成](resource_generation.zh-CN.md)。
调度支持 `--cost-table`、`--distribution`、`--model-configs` 覆盖参数，
不替换默认资源文件。

## 5. 绘制自己的结果

执行并收集实验后：

```bash
bash artifact/scripts/plot.sh results \
  --run /path/to/your/run --output /tmp/mil-plots-01
```

入口读取指定目录的 `metrics.jsonl`，可重复 `--figure` 选择
`configs/figures.yaml` 中的 ID，例如 `llm_blender_256`；不会回退读取旧结果。

`chart` 模式接受自行提供的论文样式 JSON：
`{"schema_version": 1, "figures": {"figure_id": chart_spec}}`。
格式由 `published_plot.validate_chart` 定义，`tests/synthetic.py` 提供简短的
人工示例。该模式不与未分发的论文数字或像素基准比较。

```bash
bash artifact/scripts/plot.sh chart \
  --input /path/to/your/charts.json --output /tmp/mil-charts-01
bash artifact/scripts/plot.sh candidates \
  --snapshot /path/to/your/candidate-bundle --output /tmp/mil-candidates-01
bash artifact/scripts/plot.sh supporting \
  --snapshot /path/to/your/supporting-bundle --output /tmp/mil-supporting-01
```

候选/支撑分析包必须从自己的输入显式生成，绘图接口不能证明数据属于论文原始样本。
输出目录必须是新目录，已有路径、备份路径、源码目录和普通符号链接会被拒绝。
仓库内使用 `artifact/results/` 或 `results/`，shell 入口支持 `MIL_PYTHON`。

## 6. 比较已保存的输出

`compare_schedules` 读取已有 CPU 输出与显式外部基线或日志索引进行比较，
不重新搜索。`scheduling_replay` 是独立的日志诊断入口，不是正常输入调度路径。
见 [回放](scheduling_replay.zh-CN.md)、[结果格式](result_schema.zh-CN.md)
及 [计时约定](timing_convention.zh-CN.md)。
