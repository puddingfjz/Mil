.. _on_cloud_zh_cn:

使用 SkyPilot 在云上运行
==================================

:doc:`English <run_on_sky>`

.. raw:: html

    <p align="center">
        <img src="https://imgur.com/yxtzPEu.png" alt="vLLM"/>
    </p>

可借助开源 SkyPilot 在云上运行 vLLM 并扩展到多张 GPU。
SkyPilot 用于在不同云平台运行 LLM。

安装 SkyPilot 并检查云凭据：

.. code-block:: console

    $ pip install skypilot
    $ sky check

服务配置见 SkyPilot 的
`serving.yaml <https://github.com/skypilot-org/skypilot/blob/master/llm/vllm/serve.yaml>`_。
以下示例定义资源、环境、安装和运行步骤。

.. code-block:: yaml

    resources:
        accelerators: A100

    envs:
        MODEL_NAME: decapoda-research/llama-13b-hf
        TOKENIZER: hf-internal-testing/llama-tokenizer

    setup: |
        conda create -n vllm python=3.9 -y
        conda activate vllm
        git clone https://github.com/vllm-project/vllm.git
        cd vllm
        pip install .
        pip install gradio

    run: |
        conda activate vllm
        echo 'Starting vllm api server...'
        python -u -m vllm.entrypoints.api_server \
                        --model $MODEL_NAME \
                        --tensor-parallel-size $SKYPILOT_NUM_GPUS_PER_NODE \
                        --tokenizer $TOKENIZER 2>&1 | tee api_server.log &
        echo 'Waiting for vllm api server to start...'
        while ! `cat api_server.log | grep -q 'Uvicorn running on'`; do sleep 1; done
        echo 'Starting gradio server...'
        python vllm/examples/gradio_webserver.py

在 A100 上启动 LLaMA-13B 服务：

.. code-block:: console

    $ sky launch serving.yaml

检查命令输出，可找到可分享的 Gradio 链接（如下所示）。在浏览器中打开即可进行文本补全。

.. code-block:: console

    (task, pid=7431) Running on public URL: https://<gradio-hash>.gradio.live

可选：将默认 13B 改为 65B 模型，并使用更多 GPU：

.. code-block:: console

    sky launch -c vllm-serve-new -s serve.yaml --gpus A100:8 --env MODEL_NAME=decapoda-research/llama-65b-hf
