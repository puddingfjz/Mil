# Artifact 实验范围

[English](paper_experiment_scope.md)

本范围确定实验内容和参数；配置覆盖不表示全部性能实验已经复现。

论文：**Mil: Cost-guided Minimum Makespan Scheduling for
Applications of Multiple LLMs**。本范围使用的论文版本共 14 页，
SHA-256：`7c554ef32ece500046e4999f898ca40904f3b3b7af489540e8cfd0f3c6a7c1d0`。
以下页码均为 PDF 页序号。论文配置对应原文的实验设置，历史脚本和日志提供
实现与数据来源的辅助信息。

实验目录覆盖论文中的全部实验项，并提供 512-token Ensembling、八变体 ablation
等扩展配置。论文标准设置与扩展设置分别标识，结果分别统计。

### 标准范围与扩展

- Figures 6–8 对应 28 个去重 workload 设置，与 Round2 主实验脚本的设置一致。
- `minRandom-differentSampleSize` 的 `SAMPLE_SIZE=10/12/14/16/18/20/22`
  是可选扩展。论文 §5 定义 MIN-R 最多采样 N 个 ready LLM，
  本实验 N=8，标准配置的 `sample_size=8` 与此一致。
- 较早脚本中的 5000 请求噪声扫描、`noise_level=1e-6`、1200 文档及
  15000/20000 请求压力测试不属于论文标准范围。
- 模型数不少于 N 时，MIN-R 每点运行三次，报告均值和最小/最大范围。
  paper 配置为新运行指定三个种子；历史 `run_i=4/5/6` 仅是运行次数标识，
  不记录随机种子或最终样本归属。CPU 验收检查配置和统计功能，
  MIN-R 实际重复运行及其性能统计需单独执行。
- 整项调度/实验没有额外墙钟时间上限。CPU 单项、批量、Oracle 及 paper 配置
  默认无限时；Saturn 每次 MILP 的 5 分钟限制独立保留。显式限时的诊断运行
  与论文标准设置分开统计。

## 1. 正式实验输入

Workload 使用以下三个输入接口：

| 输入 | 为复现而需要明确的内容 |
| --- | --- |
| Model set | 模型集合、版本、tokenizer；模型权重位置由环境配置提供 |
| Application 类型 | Ensembling、Routing、Chain Summary、Mixed；对应依赖图、prompt 模板、路由关系、分块与评价次数 |
| Request set | 数据集版本、所选样本及顺序、请求到模型的分配、复制次数、最大输出长度或指定输出长度 |

硬件信息、可选执行计划、成本表和输出长度采样分布是运行环境与调度器的配套
资源。论文 Definition 1（PDF 第 3 页）的形式化输入为
`(N, M, I, P, ρ, E)`：GPU 数、模型、初始请求长度、执行计划、processing functions、
依赖关系。上述 workload 输入与配套资源共同提供这些字段。

历史日志用于实验结果核对。`scheduling_replay --source-log` 从日志取 token 长度，
用于隔离输入差异、核对历史初始调度。正式调度入口从上述三项输入构造长度与
依赖数据，再调用调度器，无需历史日志。纯 CPU 回放能够验证初始计划和预测成本；它不能独立
验证真实推理耗时或 runtime adjustment 的加速效果。

## 2. 共用实验设置

来源：§5，PDF 第 7–8 页。

- 主机：8 × A100 80GB，论文描述每对 GPU 间有 NVLink；2 × 24 核 CPU、
  1510 GB RAM，Ubuntu，vLLM 推理引擎。
- 主结果比较 9 种方法：Mil、MAX、MIN、MIN-R、Saturn*、LightRule、
  LightRule*、HybridFlow、HybridFlow*。
- Saturn* 的每次 MILP 求解超时为 5 分钟；这不是整个实验任务的超时。
- 指标：inference time、extra time（生成调度的时间）、两者之和 end-to-end time。
  Figures 6–8 同时展示相对 Mil 的 inference 和 end-to-end 比值。
- MIN-R：当模型数不少于 GPU 数 N 时，每个测试点运行 **3 次**，报告 inference
  和 end-to-end 的均值，误差线为 end-to-end 的最小/最大值。
  论文没有给其他所有方法规定统一重复次数或统一取中位数的规则。

### 模型集合与请求来源

| Application | Model set | Request set / 构造 |
| --- | --- | --- |
| Ensembling | vicuna-13b-v1.5、oasst-sft-4-pythia-12b-epoch-3.5、alpaca-13b、baize-v2-13b、koala-13B-HF、dolly-v2-12b、mpt-7b-chat、chatglm3-6b、stablelm-tuned-alpha-7b | MixInstruct；9 个模型处理相同请求，无模型依赖 |
| Routing | Llama-2-70b-chat-hf、Mixtral-8x7B-Instruct-v0.1、WizardLM-13B-V1.2、CodeLlama-34b-Instruct-hf、Mistral-7B-Instruct-v0.2 | ROUTERBENCH；每条请求分配给一个模型 |
| Chain Summary | vicuna-13b-v1.5 做摘要，Llama-2-70b-chat-hf 做评价 | BOOOOKSCORE 与 BookSum；2048-token 分块，逐步摘要后评价 |
| Mixed | Chain Summary 与 Ensembling 的组合 | 同时调度两类应用；请求数量分别配置 |

Table 1 是 Routing 输入统计：上述 5 个模型的请求数依次为
408、1267、2068、456、2657，共 **6856** 条；复制 r 次即 6856 × r。
旧 CLI 中 `reqnum=10000` 不代表该 workload 实际用了 10000 条请求：
`_get_router_bench_data` 使用准备好的各模型请求列表并按复制次数扩展。

## 3. 主结果：Figures 6–8

来源：§5.1，设置在 PDF 第 8 页，图在第 9 页。下表各行分别是一条参数扫描，
各扫描维度独立变化，不构成笛卡尔积。

| 论文项 | 实验 | 变化参数 | 固定参数 |
| --- | --- | --- | --- |
| Figure 6(a) | Ensembling 请求规模 | 请求数 1000 / 2000 / 5000 / 10000 | 9 模型，max output 256 |
| Figure 6(b) | Routing，未知输出长度 | 数据复制 1 / 2 / 3 / 4 次 | 5 模型，max output 4096 |
| Figure 6(c) | Routing，已知输出长度 | 数据复制 1 / 2 / 3 / 4 次 | 强制生成数据集 response 对应的准确 token 数 |
| Figure 6(d) | Chain Summary 文档规模 | 文档数 100 / 200 / 300 / 400 / 500 | evaluation times 4，max output 900 |
| Figure 6(e) | Chain Summary 评价次数 | evaluation times 2 / 4 / 8 | 文档数 300，max output 900 |
| Figure 6(f) | Chain Summary 输出长度 | max output 100 / 500 / 900 | 文档数 400，evaluation times 4 |
| Figure 7 | Mixed 工作负载组合 | 摘要文档数 100 / 200 / 300 / 400 / 500 | Ensembling 请求数 5000；评价 4 次；摘要/Ensembling max output 为 900/256 |
| Figure 8 | Mixed 应用数量扩展性 | 应用数 2 / 4 / 8 | 基础组合为 200 篇摘要文档 + 5000 条 Ensembling 请求，整组复制 1 / 2 / 4 次；其余设置同 Figure 7 |

Figure 8 横轴是 **application number**。GPU 数保持 8；旧参数
`model_replicate_num=1/2/4` 对应 2/4/8 个应用，不能解释为 2/4/8 个模型。
部分基线在大规模设置下搜索数小时仍未完成、MILP 失败或无法同时容纳模型；
图中的叉号应保留为明确失败状态，不能作为零秒结果或静默丢弃。

上述图共 31 个横轴坐标。去除三条 Chain Summary 扫描之间的两个重合点，
以及 Figure 8 的 2-app 点与 Figure 7 的重合后，共 28 个不同 workload 设置。
乘以 9 种方法得到 252 个名义配置。该计数不包括重复运行、组件实验和失败任务，
不表示论文实际运行次数。

### Figure 9：调度执行时间线

来源：PDF 第 10 页。使用 Routing、未知输出长度、复制 1 次的主实验数据，
展示不同方法如何占用 GPU、何时加载/执行/抢占/调整模型。共有 8 个面板，
其中 MIN 和 MIN-R 共用一个面板，覆盖上述 9 种方法。

这主要是同一主实验的另一种输出，不需要引入新 request set。复现实际时间线
需要执行事件及 GPU 分配记录，初始预测计划不提供实际执行时间。

### §5.1.2：搜索效率

来源：PDF 第 11 页，正文分析而非单独编号图表。应从同一组实验中导出各方法
搜索时间、初始化/估算开销、候选计划枚举数及成本估算次数，另分析扩展性设置。
CPU 配置会影响搜索墙钟时间，不能把使用更少 CPU
的回放耗时直接与论文机器上的秒数要求精确相等。

## 4. 组件分析：Tables 2–4 及正文

来源：§5.2，PDF 第 11–12 页。

| 论文项 | 比较内容 | 实验设置 | 应保留的输出 |
| --- | --- | --- | --- |
| Table 2 | Mil 允许/禁止 preemption | Ensembling 10000 请求；Routing 未知输出长度、复制 4 次；Mixed (500, 5000)、2 apps；其余沿用主实验设置 | inference + extra time，以及相对允许抢占的 end-to-end 比值 |
| Table 3 | 输出长度采样分布误差 | Ensembling 1000 / 2000 请求，**max output 4096**；Exact、0、1e-5、1e-4、1e-3、1e-2、Rand | 相对正常 Mil 的 inference time；保存各噪声设置及种子 |
| §5.2 estimation error | 预测推理时间与真实推理时间的误差 | §5.1 四类应用；另统计已知输出长度的 Routing | 每类平均/最大相对误差与逐点预测/实测值 |
| §5.2 runtime adjustment | Mil 开启/关闭 runtime adjustment | **§5.1 的全部实验**，包括 Mixed 与应用数量扩展性 | 逐点 inference/end-to-end 时间与加速比 |
| Table 4 | Greedy Mil 与 Oracle | Ensembling 仅 3 个模型：baize-v2-13b、koala-13B-HF、dolly-v2-12b；请求数 1000 / 2000，max output 256 | inference time、估计误差，以及正文报告的搜索时间、估算/枚举次数 |

Table 2 中 Chain Summary 不单列，因为论文报告有无 preemption 产生相同调度。
它不是现有 `draw_ablation.py` 的 Mil/MIN 八变体柱状图；该历史图不能替代 Table 2。

Table 3 的噪声是在输出长度概率上加 `[-a, a]` 的均匀噪声；`Rand` 是在合法
输出长度范围内均匀采样；`Exact` 使用准确输出长度。不要直接把这七列等同于
现有 `randomCost`/禁止抢占变体。Table 3 的 4096 上限也不能沿用 Figure 6(a)
的 256 上限。

计算估算误差时需明确误差分母、绝对值处理和聚合样本。

Table 4 的 Oracle 枚举调度，用 processing functions 和已知输出长度选择方案，
不是把所有方案都在 GPU 上运行一遍。论文指出超过 3 模型会 OOM，因此该实验
不能复用默认 9 模型的 Ensembling 集合。

## 5. Processing function 的支撑实验：Figures 3–4

来源：§3，PDF 第 4–5 页。配置和分析入口覆盖以下支撑实验，输入须显式获取；
发布包不包含支撑实验结果，见 [覆盖限制](paper_coverage.zh-CN.md)。

| 论文项 | 要验证的内容 | 数据与模型 |
| --- | --- | --- |
| Figures 3(a), 3(b) | 输出长度 eCDF 对输入长度分组、请求类别的稳定性 | vicuna-13b-v1.5；No Robots 10000 请求、10 类任务 |
| Figure 3(c) 及 JSD 正文 | 跨数据集的输出长度分布 | 8 个数据集；图画 vicuna-13b-v1.5，正文另测试 chatglm3-6b、WizardLM-13B-V1.2 |
| Figure 3(d) | 单次 generation iteration 模型计算延迟与 FLOPs 的关系，按 batch size 分组 | Llama 7B，1 张 GPU，vLLM profiling |
| §3 input preparation / output sampling | 迭代另外两部分的延迟拟合 | 同一 profiling 流程，正文描述，未单独画图 |
| Figures 4(a), 4(b) | 实际与模拟推理中每个 iteration 的 running request 数 | vicuna-13b-v1.5，ShareGPT 1000 请求 |
| §3 完整成本估算示例 | 预测与实测耗时，包括/不包括模型加载 | 同 Figure 4，分别记录推理与含加载的总耗时 |

Figure 3(c) 涉及 No Robots、dolly-15k、Natural Questions、TLDR、GSM8K、
HumanEval、mbpp、ShareGPT。论文表述为各采样 10000 条；对于原始规模较小的数据集，
如何重复/抽样需从实验代码确定，不能推定。

Table 1 是输入统计；Figures 1、2、5 是问题示例、依赖图或框架图，不是独立的
性能评测任务。

## 6. 配置与资源入口

`configs/paper_experiments.yaml` 将以上论文项映射到 72 个设置和 323 个展开任务，
方法由 `configs/methods.yaml` 显式指定。512-token Ensembling 和其他额外配置
作为扩展保留，不与论文标准设置混合。

资源采集使用 `benchmarks/construct_cost_model.py`、`benchmark_throughput.py`
和 `comp_model_size.py`。采集需要真实模型与 GPU，默认 CPU 验收不启动它们。
`artifact/mil_artifact/resource_build.py` 从显式配置和已有日志构建成本表与输出
长度分布，写入新目录而非追加到源码。现成的数字资源随发布包保留；采样器为
`benchmarks/output_length_sampler.py`。完整用法见 [资源生成](resource_generation.zh-CN.md)。

配置覆盖、CPU 功能验证、历史候选结果和实际 GPU 性能必须分别解释，见
[论文映射](paper_mapping.zh-CN.md) 与 [覆盖限制](paper_coverage.zh-CN.md)。
