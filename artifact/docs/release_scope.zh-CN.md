<a id="release-scope"></a>

# 发布范围

[English](release_scope.md)

发布版提供调度、成本模拟、运行时执行、资源准备、实验配置、结果收集与绘图，包含 CPU
及 GPU 执行接口。自动检查覆盖 CPU 路径和 GPU 入口连接，不执行 GPU 操作。
目标硬件上的 GPU 构建、推理与论文性能仍未验证。

<a id="methods-and-runtime-behavior"></a>

## 方法与运行时行为

方法注册表含 24 个 ID 及分发映射。Ensembling、Routing 和 Mixed 使用基于 LPT 的
Mil，Chain Summary 使用独立贪心实现。HybridFlow 的 `old` selector 对应另一活跃变体。
运行时调整、抢占、输出长度噪声和 MIN-R 采样按方法与实验配置。

实验默认没有外层截止时间。Saturn 使用 Gurobi，每次 MILP 求解上限 300 秒，
并需要有效 Gurobi 授权。

单副本工作负载使用通用逐副本模拟器；内存计算使用模型大小表。
调度与推理加载固定本地模型快照，见[模型与数据](data_and_models.zh-CN.md)。

源码契约固定可执行结构、文档字符串和特殊指令；数值资源及原生／构建文件使用字节校验。
分发源码包含版权和许可证声明。

<a id="distribution-boundary"></a>

## 分发边界

`manifests/source_distribution.yaml` 明确列出包含文件。打包不遍历备份、不跟随源码符号链接，
也不纳入任意未追踪文件。原始日志、外部数据集、模型权重和生成结果不在分发范围内。
打包不会修改源工作树。

<a id="acceptance"></a>

## 验收

在独立源码包中运行 `artifact/scripts/validate.sh`，检查配置、数字化输入、源码契约、
CPU 运行时事件、资源构建、绘图及失败场景。代表性真实 CPU 搜索需要单独显式调用，
命令见快速开始。通过表示 CPU 正确性／可用性检查成功，不表示完整 GPU 性能复现，
见[覆盖限制](paper_coverage.zh-CN.md)。
