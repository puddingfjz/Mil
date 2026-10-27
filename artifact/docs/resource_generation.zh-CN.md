# 成本表与输出长度分布：参数化入口与纯 CPU 验证

[English](resource_generation.md)

本入口从测量日志生成成本表和输出长度分布，不启动 GPU，不加载模型权重，不连接 Ray，也不下载
模型或数据。需要 Python 3.9+、NumPy、PyYAML；可使用现有 Artifact 环境。
实现为 `artifact/mil_artifact/resource_build.py`，shell 入口为
`artifact/scripts/build_resources.sh`。

## 两个动作

- `plan`：从配置生成采集任务清单，列出模型、执行参数、所需 GPU 数和新日志位置。
  **只写 JSON，不运行命令**。没有 `--execute` 开关。
- `build`：读取已经存在的测量日志，在 CPU 上计算成本表和输出长度分布。
  输出独立资源文件，不初始化 GPU 或修改源码。

`--config` 指定 YAML/JSON 配置，`--input-root` 指定相对输入路径的根目录（默认
仓库根目录），`--model` 可重复，用完整模型 ID 选择子集。`--output` 必须是新目录，
已存在的目录或符号链接会被拒绝；没有覆盖或追加选项。

以下命令均从仓库根目录执行。示例结果目录每次应选一个新路径。

## 1. 不需要模型、GPU 或历史日志的模拟测试

```bash
bash artifact/scripts/build_resources.sh build \
  --config artifact/configs/resources/simulated.yaml \
  --output /tmp/mil-resources-simulation-01
```

该配置读取已纳入版本控制的小型合成 fixture，模拟 profiling 记录及请求长度。
所有测量值都是测试数据，不能当作论文结果。它检验解析、重复测量选择、成本表
序列化、截断请求处理和概率分布生成；单元测试还将结果与旧 `CostTable` 的 CPU
计算方法逐项比较。

## 2. 从保留的测量日志重建资源

```bash
bash artifact/scripts/build_resources.sh build \
  --config artifact/configs/resources/local_measurements.yaml \
  --output /tmp/mil-resources-from-logs-01
```

示例覆盖 ChatGLM3-6B、StableLM tuned alpha 7B 各自 TP=1/2 的四组成本测量，
以及两者 temperature=1 的 No Robots 输出长度记录。原始日志位于当前仓库的
`benchmarks/Cost_Model_per_iter_zxcpu/` 和
`benchmarks/collect_output_lengths/no_robot/`；这些本机日志并非 clean clone
的必需测试输入。缺失日志会明确报错，不会自动启动采集。

生成文件：

| 文件 | 用途 |
| --- | --- |
| `cost_table.py` | 与现有 `get_cost_table_from_serialized_data` 兼容的完整序列化数据 |
| `output_lengths.py` | 与现有 sampler 的 `pdf_dict` 键约定兼容的分布数据 |
| `report.json` | 解析后的配置、输入/输出 SHA-256、Python/NumPy 版本、重建项、继承项数、比较结果 |

该示例显式使用已提交成本表和分布作为 `base_*`：只重建选中的条目，其他条目从
基础资源继承。报告分别列出重建与继承数量；继承项直接使用基础表数据。
原有源码资源不被替换，正式调度器的默认资源选择也不会自动改变。

可用 `read_cost_table(path)` 和 `read_literal_module(path, ("pdf_dict",))`
在纯 CPU 环境读取生成文件。这两个函数只解析允许的字面量/更新语句，不执行
资源中的 Python 代码，也不触发序列化成本模块里的 vLLM 导入。

只处理一个模型：

```bash
bash artifact/scripts/build_resources.sh build \
  --config artifact/configs/resources/local_measurements.yaml \
  --model THUDM/chatglm3-6b \
  --output /tmp/mil-resources-chatglm-01
```

## 3. 生成采集任务清单

```bash
bash artifact/scripts/build_resources.sh plan \
  --config artifact/configs/resources/local_measurements.yaml \
  --output /tmp/mil-resource-plan-01
```

`acquisition.json` 保存 argv 数组、工作目录、GPU 数量和目标日志路径；所有任务
均标记 `executed: false`。目标位于新输出目录下的 `measurements/`，不指向旧
日志。`build_config.json` 将输入日志位置设置为这些目标，采集完成后可直接用于
`build` 入口。`plan` 命令不创建或启动模型进程。

旧 profiler 固定每点两次测量、1 ms 阈值以及部分采样参数；如果配置要求的
采集参数无法由旧入口表达，`plan` 会报错，不会静默忽略。`build` 可以通过
`profile_num`、`threshold_seconds` 解释符合其他设置的已有记录。

## 配置约定

`cost.entries` 中每个条目声明：

- `model`、`log`、`exec_plan: [tp, gpu_ratio, weight_load_degree, cache_gpu_num]`；
- `sample_config`，默认 `[1, 1, -1, 0]`；
- `profile_num` 默认 2，`skip_records` 默认 2，`threshold_seconds` 默认 0.001；
- `model_config: [hidden_size, num_layers]`、`flops_coefficient` 和
  `prepare_seconds` 可以直接提供；也可从显式基础表/系数表读取。

`cost.base_table` 与 `cost.coefficient_table` 指向已有资源。
`cost.compare_table` 可选择一个独立参照表，检验重建条目的六类成本数据。
模型配置缺失时直接报错，不会查询模型服务或加载 Hugging Face 模型。

`output_lengths.entries` 声明模型、日志、`max_model_len`、温度和记录数：

- `num_prompts` 是采集命令的请求上限；`expected_requests` 是过滤后实际应保存
  的记录数，两者不能混为一谈。旧 `sample_requests` 会过滤过短或过长输入，
  因此本例 10000 的请求上限对应 ChatGLM 的 9907、StableLM 的 9889 条记录。
- 日志有多个 `output_lens` 块时，必须用从零开始的 `record_block` 明确选择。
  不会自动取首块或末块，以免误用不完整运行。
- 达到上下文上限的输出沿用旧代码的截断处理；剩余概率保留在最后一个长度。
  temperature=1 使用模型短名作为键，其他温度使用 `(短名, temperature)`。
- `base_distribution` 保留未重建条目；`compare_distribution` 比较选定条目的概率。

重建失败或比较不一致时返回非零状态。比较不一致会保留带失败结论的报告；
格式错误、缺失输入等构建异常在创建结果目录前停止。

## 验证与资源范围

`artifact/scripts/validate.sh` 包含合成资源构建、序列化比较和仅生成采集清单的检查。
这些测试无需历史日志、模型权重或 GPU。

默认 zxcpu 成本表包含覆盖 14 个模型的 26 个固定条目。构建工具明确区分从日志
重新计算的条目和从基础表继承的条目；离线相等不证明重新测量了 GPU 性能。
`configs/resources/additional_measurements.yaml` 列出可选的其他主机测量，不能
用这些条目自动覆盖默认表。长度超出模型窗口的记录会被拒绝，不通过裁剪来放行。
CPU 调度可显式选择 `--cost-table` 和 `--distribution`，结果记录资源哈希。
