<a id="mil-implementation"></a>

# Mil 实现

[English](README.md)

`search_exec_plans.py` 搜索执行计划，`fake_scheduling.py` 预测成本，
`schedule_multi_model.py` 协调放置与运行时调整。
`my_llm_infer_worker_multiprocessing.py` 承载推理 worker。
注册方法中包含名称带 `old` 的兼容变体，分发映射见 `artifact/configs/methods.yaml`。

参数化实验、CPU 验证、资源生成和绘图见 [artifact 指南](../artifact/README.zh-CN.md)。
仅执行 CPU 检查时，不要通过导入 GPU 运行时模块来启动检查。
搜索、模拟和 Saturn 求解器模块作为库使用；配置实验通过 artifact 入口运行。

`construct_cost_model.py`、`benchmark_throughput.py` 和 `comp_model_size.py`
提供 GPU 测量功能，需要显式准备模型与 GPU 环境。
artifact 资源规划器只生成采集命令，不执行命令。
数字化成本与分布模块是固定输入，不表示其原始测量已经重新运行。
