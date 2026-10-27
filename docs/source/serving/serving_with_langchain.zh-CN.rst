.. _run_on_langchain_zh_cn:

通过 Langchain 提供服务
==================================

:doc:`English <serving_with_langchain>`

也可通过 `Langchain <https://github.com/langchain-ai/langchain>`_ 使用 vLLM。

安装 Langchain：

.. code-block:: console

    $ pip install langchain -q

使用 ``langchain`` 的 ``VLLM`` 类运行单卡或多卡推理。
下例为 Hugging Face 模型启用 ``trust_remote_code=True``；多卡可设置 ``tensor_parallel_size``。

.. code-block:: python

    from langchain.llms import VLLM

    llm = VLLM(model="mosaicml/mpt-7b",
               trust_remote_code=True,  # mandatory for hf models
               max_new_tokens=128,
               top_k=10,
               top_p=0.95,
               temperature=0.8,
               # tensor_parallel_size=... # for distributed inference
    )

    print(llm("What is the capital of France ?"))

更多信息见 `Langchain 集成教程 <https://github.com/langchain-ai/langchain/blob/master/docs/docs/integrations/llms/vllm.ipynb>`_。
