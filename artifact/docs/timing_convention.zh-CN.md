# 计时边界与参考结果

[English](timing_convention.md)

Artifact 的计时口径遵循
`benchmarks/search_exec_plans.py:get_best_model_schedule` 的日志边界。

源码在模型对象构造、依赖修正和融合之前保存 `time1`，融合结束后保存
`time_before_search`，搜索完成后保存 `time2`。日志打印：

- `Total search time = time2 - time1`，包含初始化/准备和搜索。
- `Total time for preparation before search = time_before_search - time1`，是前者子项。
- `total running time` 是后续执行阶段，包含原有的冷启动/加载边界。

因此采用 `extra = Total search time`、`total = inference + extra`。
单独考察搜索算法时可使用 `search_after_preparation = search - preparation`；
它不应与包容性的 search 再相加。输入分词发生在此计时区间之前，不应把该
分词耗时解释为日志中已包含的 GPU 推理时间。

每次运行分别保存原始计时及源码身份，不拼接不同运行的计时分量。
发布包不附带参考测量；外部日志缺失源码版本时，不能仅凭字段名证明计时边界相同。
