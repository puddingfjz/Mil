生产监控指标
============

:doc:`English <metrics>`

vLLM 提供用于监测系统健康状态的指标，通过 OpenAI 兼容 API 服务的 ``/metrics`` 暴露。

可用指标如下：

.. literalinclude:: ../../../vllm/engine/metrics.py
    :language: python
    :start-after: begin-metrics-definitions
    :end-before: end-metrics-definitions
