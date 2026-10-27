.. _quickstart_zh_cn:

快速开始
========

:doc:`English <quickstart>`

本指南介绍数据集离线批量推理、LLM API 服务和 OpenAI 兼容服务。
继续前先完成 :ref:`安装说明 <installation_zh_cn>`。

.. note::


    vLLM 默认从 `Hugging Face <https://huggingface.co/>`_ 下载模型。
    若使用 `ModelScope <https://www.modelscope.cn>`_ 模型，设置以下环境变量：

    .. code-block:: shell

        export VLLM_USE_MODELSCOPE=True


离线批量推理
------------

以下示例为一组输入 prompt 生成文本，展示数据集离线批量推理。

从 vLLM 导入 ``LLM`` 和 ``SamplingParams``：前者是离线推理主要入口，后者配置采样参数。

.. code-block:: python

    from vllm import LLM, SamplingParams

定义输入列表和生成参数：温度 0.8，核采样概率 0.95。其他参数见
`SamplingParams 类定义 <https://github.com/vllm-project/vllm/blob/main/vllm/sampling_params.py>`_。

.. code-block:: python

    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

使用 ``LLM`` 类和 OPT-125M 初始化离线引擎；模型列表见 :ref:`支持的模型 <supported_models_zh_cn>`。

.. code-block:: python

    llm = LLM(model="facebook/opt-125m")

调用 ``llm.generate`` 将 prompt 加入等待队列并执行生成，返回包含全部输出 token 的
``RequestOutput`` 对象列表。下面逐项打印输入和生成文本。

.. code-block:: python

    outputs = llm.generate(prompts, sampling_params)

    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

完整代码见 `examples/offline_inference.py <https://github.com/vllm-project/vllm/blob/main/examples/offline_inference.py>`_。

OpenAI 兼容服务
----------------------

vLLM 可部署为实现 OpenAI API 协议的服务，替代应用的 OpenAI API 后端。
默认地址为 ``http://localhost:8000``，可用 ``--host`` 和 ``--port`` 修改。
一次托管一个模型（下例为 OPT-125M），提供模型列表、聊天补全和文本补全接口。

启动服务：

.. code-block:: console

    $ python -m vllm.entrypoints.openai.api_server \
    $     --model facebook/opt-125m

默认使用 tokenizer 中的聊天模板；可通过 ``--chat-template`` 覆盖：

.. code-block:: console

   $ python -m vllm.entrypoints.openai.api_server \
   $     --model facebook/opt-125m \
   $     --chat-template ./examples/template_chatml.jinja

请求格式与 OpenAI API 一致，例如列出模型：

.. code-block:: console

    $ curl http://localhost:8000/v1/models

使用 ``--api-key`` 或 ``VLLM_API_KEY`` 可启用请求头 API key 检查。

使用 OpenAI 文本补全 API
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

向模型发送输入 prompt：

.. code-block:: console

    $ curl http://localhost:8000/v1/completions \
    $     -H "Content-Type: application/json" \
    $     -d '{
    $         "model": "facebook/opt-125m",
    $         "prompt": "San Francisco is a",
    $         "max_tokens": 7,
    $         "temperature": 0
    $     }'

也可通过 ``openai`` Python 包访问兼容服务。将 API key 和服务地址设为 vLLM 配置：

.. code-block:: python

    from openai import OpenAI

    # Modify OpenAI's API key and API base to use vLLM's API server.
    openai_api_key = "EMPTY"
    openai_api_base = "http://localhost:8000/v1"
    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
    completion = client.completions.create(model="facebook/opt-125m",
                                          prompt="San Francisco is a")
    print("Completion result:", completion)

更完整的客户端示例见
`examples/openai_completion_client.py <https://github.com/vllm-project/vllm/blob/main/examples/openai_completion_client.py>`_。

使用 OpenAI 聊天 API
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

聊天 API 支持多轮交互，可将交流内容保存在聊天历史中，适用于需要上下文或详细解释的任务。

通过 OpenAI 聊天 API 查询模型：

使用 `create chat completion <https://platform.openai.com/docs/api-reference/chat/completions/create>`_
接口发送聊天请求：

.. code-block:: console

    $ curl http://localhost:8000/v1/chat/completions \
    $     -H "Content-Type: application/json" \
    $     -d '{
    $         "model": "facebook/opt-125m",
    $         "messages": [
    $             {"role": "system", "content": "You are a helpful assistant."},
    $             {"role": "user", "content": "Who won the world series in 2020?"}
    $         ]
    $     }'

Python 客户端示例：

也可通过 ``openai`` Python 包进行聊天交互：

.. code-block:: python

    from openai import OpenAI
    # Set OpenAI's API key and API base to use vLLM's API server.
    openai_api_key = "EMPTY"
    openai_api_base = "http://localhost:8000/v1"

    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    chat_response = client.chat.completions.create(
        model="facebook/opt-125m",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Tell me a joke."},
        ]
    )
    print("Chat response:", chat_response)

更多示例和高级聊天 API 功能见 OpenAI 官方文档。
