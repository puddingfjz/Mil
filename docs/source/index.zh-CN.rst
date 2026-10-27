欢迎使用 vLLM！
====================

:doc:`English <index>`

.. note::


   这些页面介绍上游 vLLM API 与用法。Mil 需要本仓库的定制源码，
   安装与实验说明见 ``artifact/env/README.zh-CN.md`` 和 ``artifact/README.zh-CN.md``。


.. figure:: ./assets/logos/vllm-logo-text-light.png
  :width: 60%
  :align: center
  :alt: vLLM
  :class: no-scaled-link

.. raw:: html

   <p style="text-align:center">
   <strong>面向所有人的简单、快速、低成本 LLM 服务
   </strong>
   </p>

   <p style="text-align:center">
   <script async defer src="https://buttons.github.io/buttons.js"></script>
   <a class="github-button" href="https://github.com/vllm-project/vllm" data-show-count="true" data-size="large" aria-label="Star">Star</a>
   <a class="github-button" href="https://github.com/vllm-project/vllm/subscription" data-icon="octicon-eye" data-size="large" aria-label="Watch">Watch</a>
   <a class="github-button" href="https://github.com/vllm-project/vllm/fork" data-icon="octicon-repo-forked" data-size="large" aria-label="Fork">Fork</a>
   </p>

vLLM 是一个快速、易用的 LLM 推理与服务库。

上游项目理念是让所有人都能使用简单、快速、低成本的 LLM 服务。

性能特性：

* 高服务吞吐量
* 使用 **PagedAttention** 高效管理注意力键值内存
* 连续批处理
* 使用 CUDA／HIP graph 加速模型执行
* 支持 `GPTQ <https://arxiv.org/abs/2210.17323>`_、
  `AWQ <https://arxiv.org/abs/2306.00978>`_、
  `SqueezeLLM <https://arxiv.org/abs/2306.07629>`_ 和 FP8 KV Cache 量化
* 优化的 CUDA 内核

使用特性：

* 集成常见 Hugging Face 模型
* 支持并行采样、束搜索等解码算法的高吞吐服务
* 用于分布式推理的张量并行
* 流式输出
* OpenAI 兼容 API 服务
* 支持 NVIDIA 与 AMD GPU
* 实验性前缀缓存
* 实验性多 LoRA 支持

更多信息：

* `vLLM 发布博客 <https://vllm.ai>`_：PagedAttention 介绍。
* `vLLM 论文 <https://arxiv.org/abs/2309.06180>`_：SOSP 2023。
* `连续批处理的吞吐量与延迟分析 <https://www.anyscale.com/blog/continuous-batching-llm-inference>`_：Cade Daniel 等人撰写。

文档
----

.. toctree::
   :maxdepth: 1
   :caption: 入门

   getting_started/installation.zh-CN
   getting_started/amd-installation.zh-CN
   getting_started/quickstart.zh-CN

.. toctree::
   :maxdepth: 1
   :caption: 服务部署

   serving/distributed_serving.zh-CN
   serving/run_on_sky.zh-CN
   serving/deploying_with_triton.zh-CN
   serving/deploying_with_docker.zh-CN
   serving/serving_with_langchain.zh-CN
   serving/metrics.zh-CN

.. toctree::
   :maxdepth: 1
   :caption: 模型

   models/supported_models.zh-CN
   models/adding_model.zh-CN
   models/engine_args.zh-CN

.. toctree::
   :maxdepth: 1
   :caption: 量化

   quantization/auto_awq.zh-CN
   quantization/fp8_e5m2_kv_cache.zh-CN

.. toctree::
   :maxdepth: 2
   :caption: 开发者文档

   dev/engine/engine_index.zh-CN

索引与表格
==========

* :ref:`genindex`
* :ref:`modindex`
