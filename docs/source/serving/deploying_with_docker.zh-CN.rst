.. _deploying_with_docker_zh_cn:

使用 Docker 部署
========================

:doc:`English <deploying_with_docker>`

vLLM 提供官方 Docker 镜像，用于运行 OpenAI 兼容服务。
镜像位于 Docker Hub 的
`vllm/vllm-openai <https://hub.docker.com/r/vllm/vllm-openai/tags>`_ 仓库。

.. code-block:: console

    $ docker run --runtime nvidia --gpus all \
        -v ~/.cache/huggingface:/root/.cache/huggingface \
        --env "HUGGING_FACE_HUB_TOKEN=<secret>" \
        -p 8000:8000 \
        --ipc=host \
        vllm/vllm-openai:latest \
        --model mistralai/Mistral-7B-v0.1

.. note::


        ``--ipc=host`` 或 ``--shm-size`` 可让容器获得共享内存。
        vLLM 使用的 PyTorch 需要共享内存在进程间共享数据，尤其是张量并行推理。


也可用提供的 Dockerfile 从源码构建并运行。以下为构建命令，
可选传入 ``max_jobs``、``nvcc_threads`` 等构建参数：

.. code-block:: console

    $ DOCKER_BUILDKIT=1 docker build . --target vllm-openai --tag vllm/vllm-openai # optionally specifies: --build-arg max_jobs=8 --build-arg nvcc_threads=2

.. note::


        默认针对所有 GPU 类型构建以兼容更多设备。若仅针对当前机器的 GPU，
        可添加 ``--build-arg torch_cuda_arch_list=""``，自动检测并针对当前 GPU 构建。


运行 vLLM：

.. code-block:: console

    $ docker run --runtime nvidia --gpus all \
        -v ~/.cache/huggingface:/root/.cache/huggingface \
        -p 8000:8000 \
        --env "HUGGING_FACE_HUB_TOKEN=<secret>" \
        vllm/vllm-openai <args...>
