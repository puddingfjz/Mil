# vLLM 运行路径

[English](vllm_runtime_guide.md)

多模型计划由 `benchmarks/schedule_multi_model.py` 产生并交给 vLLM 执行。
以下为 GPU 执行路径，其构建与推理验证独立于 CPU 调度测试。

## 请求与执行

`vllm/entrypoints/llm.py` 中的 `LLM._run_engine` 从 communicator 获取后续输入，
按间隔或请求完成状态回传输出。运行队列为空时仍可能等待依赖模型的输出。
`vllm/core/multimodel_scheduler.py` 负责跨模型请求和共享状态。

`LLMEngine.step` 调用 `Scheduler.schedule`；worker 先处理缓存操作，再调用
`ModelRunner.execute_model` 准备输入、执行模型和采样，最后由 engine 更新请求状态。
这条路径消费选定计划，不重新搜索整个多模型调度。

## 初始化和重新调度

`USE_VLLM == 'True'` 或 `SOFT_RESCHEDULE == 'False'` 选择普通初始化路径；
其他情况使用已有 actor 池或首次创建 actor。复用 actor 不等于复用原 Worker、
模型张量或 KV cache：内部 Worker、模型装载和调度器仍有重新初始化步骤。

`_init_cache_no_profile` 依赖固定模型/并行度的缓存容量数据，并按显存比例调整；
不能当作任意模型、任意硬件的通用测量。`no_profile` 和 `fake_profile` 都不是
纯 CPU 入口，它们仍涉及 CUDA 内存查询、缓存初始化或模型 warmup。

## 权重和 KV 缓存

Llama 普通模式使用 `load_weights_ori`；卸载模式使用
`preprocess_for_offloading_weights_fast`。缓存 GPU 选择结合
`TOT_ORDERED_GPUS` 与 `CUDA_VISIBLE_DEVICES`，权重驻留设置需经过实际 interval、
degree 与缓存块分配，不能直接把环境变量值视为驻留层数。

`weight_cache_cpu` 可保存 GPU 上分配张量的视图。
`forward_ours` 依次应用驻留层变化、等待加载事件、计算、记录事件并触发后续搬运。
`load_layer_params` 的地址更新、CUDA stream 等待和事件顺序不可分离。
原生实现在 `csrc/cache_kernels.cu`，使用当前 stream 上的异步设备间拷贝。

动态增加驻留层会同时改变逻辑块表、KV 可用区间、引用计数、张量视图和物理搬移。
相关实现位于 `core/scheduler.py`、`core/block_manager.py` 和
`worker/cache_engine.py`。actor、缓存和进程组释放逻辑位于 engine、worker 及
`model_executor/parallel_utils/` 中。
