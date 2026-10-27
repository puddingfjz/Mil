# 论文到 Artifact 的映射

[English](paper_mapping.md)

论文原文范围见 [paper_experiment_scope.md](paper_experiment_scope.zh-CN.md)。机器可读目录为
`configs/paper_experiments.yaml`；`paper_suite` 检查 20 个项目并展开 72 个去重设置、
323 个调度任务。重复图坐标引用同一个任务；全组合与 512-token 设置作为扩展配置提供。

| 论文项目 | 稳定 ID / 处理入口 |
| --- | --- |
| Figure 6(a) Ensembling 1k/2k/5k/10k、256 tokens | `figure_6a` / `paper_results` |
| Figure 6(b/c) Routing 未知/已知输出，复制 1–4 次 | `figure_6b`, `figure_6c` / `paper_results` |
| Figure 6(d–f) 摘要文档数、评价次数、输出长度三个独立扫描 | `figure_6d`, `figure_6e`, `figure_6f` / `paper_results` |
| Figure 7 Mixed | `figure_7` / `paper_results` |
| Figure 8 2/4/8 个应用，固定 8 个逻辑 GPU | `figure_8` / `paper_results` |
| Figure 9 Routing 未知输出、复制 1 次的实际时间线 | `figure_9` / `paper_results --timelines` |
| Table 2 抢占 | `table_2` / 三个工作负载的 Mil 对照 |
| Table 3 Exact / 0 / 1e-5 / 1e-4 / 1e-3 / 1e-2 / Rand | `table_3` / 1k、2k、4096 tokens |
| Table 4 三模型 Oracle | `table_4` / Baize、Koala、Dolly，1k/2k、256 tokens |
| §5.2 runtime adjustment | `runtime_adjustment` / 复用所有 §5.1 设置及关闭调整的对照 |
| §5.1.2 搜索效率，§5.2 预测误差 | `search_efficiency`, `estimation_error` / `analyses.json` |
| Figure 3(a/b) 输入长度分组、请求类别 | `figure_3ab` / `supporting_analyses` |
| Figure 3(c) 三模型八数据集分布与 JSD | `figure_3c_jsd` / `supporting_analyses` |
| Figure 3(d) FLOPs/迭代延迟、准备与采样成本 | `figure_3d` / `supporting_analyses` |
| Figure 4 真实与模拟 trace | `figure_4` / `supporting_analyses` |
| Table 1 Routing 输入统计 | `table_1` / 从正常 request set 统计 |

Figures 1、2、5 是说明图，不新增性能任务。配置齐全不等于已经执行实验。
发布包不附带实验结果，验证范围见 [覆盖说明](paper_coverage.zh-CN.md)。

## 方法对应

`configs/methods.yaml` 保存全部 dispatcher、开关和证据。主比较包含 Mil、MAX、
MIN、MIN-R、Saturn*、LightRule、LightRule*、HybridFlow、HybridFlow*。

Ensembling、Routing、Mixed 的 Mil 使用 `greedy_saturn_LPT`。
Chain Summary 的最终绘图源选择 `greedy_saturn`，对应原源码
`get_candidate_plan_groups_greedy_best_exec_plan_first`；LPT 变体调用独立的
`get_candidate_plan_groups_greedy_best_exec_plan_first_LPT`，先按拓扑层和最少 GPU
方案下的最长完成时间考虑模型。两个实现分别标为 `mil` 与
`mil_chain_summary`，结果独立统计。
CPU 可选回归比较需要显式提供外部参考，不包含历史结果基线。

HybridFlow 的旧/新实现由历史提交 9011731 区分：旧版本保留为
`hybridflow_model_parallel_old`，新版本逐个加载 colocated set 中的模型，CLI
仍叫 `hybridflow_model_parallel`。历史日志没有源码提交，候选匹配中的 Round2
名称只是代际线索，不能单独证明用于论文。两种方法的结果保持独立。

LightRule 使用期望输出长度（`noise_level=-2`）；LightRule* 使用原成本模型。
初始 plan 不提供数值成本时保留 null，不填零。Saturn* 继续使用 PuLP/Gurobi，
原求解上限 300 秒，无 CBC 自动替代。使用者需配置有效 Gurobi 授权；默认 CPU 验收不启动该求解器。

## 对照与统计协议

- Table 2 的禁止抢占使用 `greedy_saturn_LPT_no_preemption`；旧 `randomCost`
  同时带禁止抢占语义，不能用作 Table 3 的 Rand。
- Table 3 Exact 使用已有输出长度。Rand 在未知输出长度路径中均匀采样合法区间，
  其他噪声参数使用输出长度概率扰动函数。
- Table 4 的模型子集对应 builder 的 `[3:6]`。Oracle 搜索默认无外层时间上限；
  显式设置诊断超时时，结果记录该限制及超时状态。
- 论文对模型数不少于 8 的 MIN-R 要求三次运行、均值及 min/max。新配置显式给出
  seed 0/1/2。每次运行应保存实际种子、重复标记和样本数。
- 其他方法按显式重复配置执行，只在相同参数与方法内聚合；不得在运行后挑选最小耗时作为结果。
- `search_time_seconds` 包含 preparation；`total = inference + search`。计时口径
  见 [timing_convention.md](timing_convention.zh-CN.md)。

## 输入、模拟与实测的边界

正式输入是 model set、application 和 request set。`cpu_schedule prepare` 从原始
数据调用原 builder，`cpu_schedule run` 读取独立数字化 request set 并调用真实
scheduler/cost simulator。日志回放是单独的 `scheduling_replay` 诊断入口。

CPU 输出只说明初始模型/GPU 分配、阶段顺序和预测成本。运行时逻辑由原函数加
合成完成事件检查；实际加速比需要真实测量。Figure 9 只使用日志中的实际
阶段事件。Figure 4 缺失真实/模拟成对 trace 时报告缺口，不能用预测 plan 替代。

分发内容仅包含源码、运行输入、测试与实验指令，不包含实验日志或结果。
数据与模型获取方式见 manifests；BooookScore 原文、模型权重和许可证密钥不在源码包中。
