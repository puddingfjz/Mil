# 核心调度源码阅读指南

[English](core_scheduler_guide.md)

三个核心模块分别负责计划搜索、成本模拟和运行时协调。候选搜索、初始 GPU 放置和
运行时计划选择使用各自的排序规则，下文分别说明。

权重搬运、KV cache 与 worker 生命周期见
[vLLM 定制运行路径指南](vllm_runtime_guide.zh-CN.md)。

## 建议阅读顺序

| 模块 | 先看哪些入口 | 负责什么 |
| --- | --- | --- |
| [search_exec_plans.py](../../benchmarks/search_exec_plans.py) | `get_best_model_schedule`、`_get_best_model_schedule`、`MyModelSystem.get_candidate_plan_groups_dispatch` | 选择成本资源，构造模型状态，按方法生成候选并搜索阶段计划序列 |
| [fake_scheduling.py](../../benchmarks/fake_scheduling.py) | `fake_FCFS_schedule`、`fake_FCFS_schedule_vertical_fuse`、`estimate_cost_from_predicted_logs` | 根据请求长度、到达时间和缓存限制模拟进度，再用成本表估计迭代延迟和完成时间 |
| [schedule_multi_model.py](../../benchmarks/schedule_multi_model.py) | `search_best_scheduling`、`prepare_exec_plan_states`、`initialize_SHARED_CONTECT`、`get_the_next_round_exec_plan_schedule` | 将搜索结果包装为运行状态，分配具体 GPU，协调模型进程及阶段切换 |

`schedule_multi_model.py` 也包含应用请求图和长度回调的构造代码。它的
`get_schedule_setting` 区分数据集输入与合成输入；`search_best_scheduling`
构造长度回调并调用搜索模块，随后将结果转换成 `MyExecPlanState`。
完整运行入口 `main_with_preemption` 在取得计划后再组织模型进程。

这些模块依赖 Ray、vLLM 等组件。无 GPU 功能检查使用
`artifact/scripts/validate.sh`；运行 CPU 调度搜索的命令见
[CPU 使用指南](cpu_quickstart.zh-CN.md)。

## 搜索模块中的数据对象

- `MyModelInfor` 和 `MyFusedModelInfor`：模型、依赖及请求进度。
- `MyExecPlan` 和 `MyVerticalFusedExecPlan`：单个计划的资源设置、预测成本与阶段进度。
- `MyExecPlanGroup`：同一阶段中的计划组合及依赖到达关系。
- `MyExecPlanGroupSeq`：阶段组合序列、阶段时长与保存的模型状态。
- `MyModelSystem`：模型系统状态，以及不同方法的候选生成入口。

方法名通过 `get_candidate_plan_groups_dispatch` 分发。
`hybridflow_model_parallel_old` 映射到 `get_candidate_plan_groups_HybridFlow_old`。
论文方法与 selector 的对应关系见 `artifact/configs/methods.yaml` 和分发测试。

`get_one_stage_exec_plans_sorted` 根据方法选择排序路径，部分方法提前返回。
`_get_best_model_schedule` 处理累计预测成本、状态去重/剪枝和阶段扩展。
算法内部的搜索计数规则独立于外层墙钟时间限制；CPU 调度入口默认不设外层时间上限。

## 成本预测不是实测运行时间

`fake_FCFS_schedule` 根据到达时间选择对应模拟路径，形成 prefill/decode 迭代记录，
再调用成本表得到累计延迟、请求完成时间和吞吐量信息。纵向融合有独立模拟入口。
请求 ID、长度数组和进度范围按索引对应，更新或排序时需保持这种对应关系。

`get_info_at_stop_time` 的快速路径使用传入的 `stop_iter_i` 计算完成范围。
`update_fake_FCFS_schedule_metadata` 负责阶段后的进度更新。
这些时间值是成本模型的预测值，区别于实际 GPU 测量值。

## 三种排序和放置场景

| 场景 | 代码入口 | 当前行为 |
| --- | --- | --- |
| 搜索候选 | `MyModelSystem` 的各方法及 `get_one_stage_exec_plans_sorted` | 方法各自决定候选构造、成本计算和排序，不能一概等同于 GPU 数排序 |
| 初始 GPU 放置 | `initialize_SHARED_CONTECT` | 对首阶段计划按 `num_worker`、总 GPU 数是否为互联单元大小的整数倍、`dp_size` 的元组降序处理，再分配 GPU ID |
| 运行时待启动计划 | `_try_to_load_exec_plans` 及两个 root selector | 先考虑候选依赖；通常在依赖根中优先选择 `num_worker * dp_size` 较大的计划，普通 LightRule 路径保留根节点的输入列表顺序 |

运行时 selector 中的“根”是相对于当前候选列表定义的：计划的上游模型 ID 与
当前候选模型 ID 没有交集，才会进入根集合。selector 在该集合内选择待启动计划。
`LightRule_our_cost_model` 使用 GPU 优先分支，普通 LightRule 保持输入顺序。

`_get_the_first_plan_state_when_sorted_by_topology` 返回输入顺序中的第一个依赖根；
GPU 优先 selector 则在依赖根中按 GPU 数降序选择。两种行为分别有 CPU 功能测试。

`no_runtime_adjustment` 使用阶段屏障推进函数，根据尚在运行的计划决定何时
考虑下阶段。该路径调用 `_try_to_load_exec_plans` 分配 GPU，外层
`get_the_next_round_exec_plan_schedule` 处理 GPU 互联相关的放置调整。
