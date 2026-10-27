<a id="add-a-model"></a>

# 添加模型

[English](add_a_new_model.md)

按照模型的访问和许可条款获取权重与 tokenizer。在 artifact 模型清单中添加明确的
revision 和模型配置；不要把凭据或特定机器的缓存路径纳入版本控制。

在目标 GPU 硬件上使用 `comp_model_size.py` 中的函数采集模型大小、FLOPs 系数和
初始化成本；使用 `construct_cost_model.py` 采集逐迭代延迟，使用
`benchmark_throughput.py` 采集输出长度。CPU 验证不会执行这些 GPU 测量。

在资源配置中声明采集日志和模型元数据，再通过
[参数化构建器](../artifact/docs/resource_generation.zh-CN.md) 生成独立的成本与分布模块。
重建延迟资源前先更新初始化元数据。构建器不会向源码文件追加内容。
显式选择新资源，完成 CPU 检查后再启动 GPU 实验。
