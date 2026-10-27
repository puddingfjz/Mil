<a id="parallel-utilities"></a>

# 并行工具

[English](README.md)

本目录的文件移植自 [Megatron-LM](https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core)，
仅保留推理使用的代码。

本目录包含并行状态管理、通信操作、自定义 all-reduce 及辅助工具。
它们在 Mil 执行流程与进程组生命周期中的作用见
[运行路径指南](../../../artifact/docs/vllm_runtime_guide.zh-CN.md)。
