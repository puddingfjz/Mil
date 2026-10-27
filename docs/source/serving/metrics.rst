Production Metrics
====================================

:doc:`简体中文 <metrics.zh-CN>`

vLLM exposes a number of metrics that can be used to monitor the health of the
system. These metrics are exposed via the `/metrics` endpoint on the vLLM
OpenAI compatible API server.

The following metrics are exposed:

.. literalinclude:: ../../../vllm/engine/metrics.py
    :language: python
    :start-after: begin-metrics-definitions
    :end-before: end-metrics-definitions
