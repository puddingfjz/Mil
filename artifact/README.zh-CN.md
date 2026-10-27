# Mil Artifact

[English](README.md)

Mil 调度由多个 LLM 组成的应用。本发布版提供调度器、成本模拟器、改造后的
vLLM 运行时、运行输入、实验配置、结果收集与绘图工具，不附带实验日志、
实测结果、论文绘图数据或历史 CPU 结果基线。

## 快速开始

先按 [CPU 快速开始](docs/cpu_quickstart.zh-CN.md) 安装环境，在仓库根目录执行：

```bash
bash artifact/scripts/validate.sh
python -m artifact.mil_artifact.validate_cpu \
  --cases chain/light_rule_star --workers 1 --output /tmp/mil-cpu-check-01
```

第一条命令检查输入、源码完整性、合成行为与工具链；第二条显式运行真实 CPU
调度器，检查输出结构、输入身份和 GPU 禁用保护。两者均不执行 GPU 推理，
也不与未分发的历史实测结果对比。省略 `--cases` 可运行八个代表性 CPU 用例；
搜索默认没有外层时间限制。

## 实验与绘图

| 用途 | 入口 |
| --- | --- |
| 列出论文实验 | `python -m artifact.mil_artifact.paper_suite --output NEW_DIR` |
| 预览 GPU 实验命令 | `bash artifact/scripts/run.sh --profile paper --dry-run` |
| 执行 GPU 实验 | 按 [GPU 安装与执行说明](env/README.zh-CN.md#gpu-runtime) 操作 |
| 构造运行资源 | `bash artifact/scripts/build_resources.sh build --config CONFIG --output NEW_DIR` |
| 只生成 profiling 命令 | 同一资源入口使用 `plan` |
| 绘制自己收集的结果 | `bash artifact/scripts/plot.sh results --run RUN_DIR --output NEW_DIR` |
| 绘制自行提供的论文样式图表数据 | `bash artifact/scripts/plot.sh chart --input CHART_JSON --output NEW_DIR` |
| 绘制外部候选或支撑分析包 | `plot.sh candidates --snapshot BUNDLE --output NEW_DIR` 或 `plot.sh supporting --snapshot BUNDLE --output NEW_DIR` |
| 用外部基线比较已保存的 CPU 调度结果 | `python -m artifact.mil_artifact.compare_schedules --help` |

输出目录必须是新目录。实验目录包含 20 个论文项目、72 个独立设置与 323 个展开的
调度任务；这些数字表示配置覆盖，不代表已有实测结果。工作负载输入是模型集合、
应用类型与请求集合，硬件描述、成本表和输出长度分布提供调度所需资源。

## 源码与输入布局

- `benchmarks/`：调度、模拟、运行时协调、profiling 与数值资源。
- `vllm/`、`csrc/`：Mil 改造的推理运行时与原生扩展。
- `artifact/configs/`：实验、方法、硬件及资源生成配置。
- `artifact/inputs/`：七份冻结请求集与模型配置。
- `artifact/expected/cpu/source_contract.json`：源码签名，不是实验结果。
- `artifact/manifests/`：资源获取、完整性与源码分发规范。

运行时必须使用本仓库改造的源码，GPU 执行还需要从这些源码构建扩展。
保留的包版本标签为 `0.3.0`，实际实现由源码及构建校验值识别。

## 文档

- [CPU 快速开始](docs/cpu_quickstart.zh-CN.md) 与 [结果格式](docs/result_schema.zh-CN.md)
- [实验范围](docs/paper_experiment_scope.zh-CN.md)、[论文映射](docs/paper_mapping.zh-CN.md) 与 [验证范围](docs/paper_coverage.zh-CN.md)
- [资源生成](docs/resource_generation.zh-CN.md) 与 [模型和数据](docs/data_and_models.zh-CN.md)
- [调度器](docs/core_scheduler_guide.zh-CN.md) 与 [运行时](docs/vllm_runtime_guide.zh-CN.md)
- [计时](docs/timing_convention.zh-CN.md)、[故障排查](docs/troubleshooting.zh-CN.md) 与 [分发](docs/publication.zh-CN.md)

模型、数据及 Saturn 的 Gurobi 授权独立于代码许可证。
发布版提供 GPU 执行接口；CPU 检查不能证明 GPU 性能。
