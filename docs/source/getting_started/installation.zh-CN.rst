.. _installation_zh_cn:

安装
====

:doc:`English <installation>`

vLLM 是 Python 库，也包含预编译的 C++ 和 CUDA（12.1）二进制。

.. note::


   本页是上游安装参考。Mil 应使用本地定制源码及 ``artifact/env/README.zh-CN.md`` 中的流程。


要求
----

* 操作系统：Linux
* Python：3.8–3.11
* GPU：计算能力至少 7.0，如 V100、T4、RTX20xx、A100、L4、H100 等。

通过 pip 安装
------------------

可选地创建 Conda 环境，然后用 pip 安装 CUDA 12.1 版 vLLM：

.. code-block:: console

    $ # (Optional) Create a new conda environment.
    $ conda create -n myenv python=3.9 -y
    $ conda activate myenv

    $ # Install vLLM with CUDA 12.1.
    $ pip install vllm

.. note::


    二进制默认使用 CUDA 12.1 编译。以下命令安装 CUDA 11.8 版 vLLM，
    并重新安装对应版本的 PyTorch 和 xFormers：

    .. code-block:: console

        $ # Install vLLM with CUDA 11.8.
        $ export VLLM_VERSION=0.2.4
        $ export PYTHON_VERSION=39
        $ pip install https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+cu118-cp${PYTHON_VERSION}-cp${PYTHON_VERSION}-manylinux1_x86_64.whl

        $ # Re-install PyTorch with CUDA 11.8.
        $ pip uninstall torch -y
        $ pip install torch --upgrade --index-url https://download.pytorch.org/whl/cu118

        $ # Re-install xFormers with CUDA 11.8.
        $ pip uninstall xformers -y
        $ pip install --upgrade xformers --index-url https://download.pytorch.org/whl/cu118


.. _build_from_source_zh_cn:

从源码构建
----------

也可克隆源码后构建安装，通常需要 5–10 分钟：

.. code-block:: console

    $ git clone https://github.com/vllm-project/vllm.git
    $ cd vllm
    $ pip install -e .  # This may take 5-10 minutes.

.. tip::


    若构建遇到问题，可使用 NVIDIA PyTorch Docker 镜像。
    ``--ipc=host`` 用于提供足够的共享内存。

    .. code-block:: console

        $ # Use `--ipc=host` to make sure the shared memory is large enough.
        $ docker run --gpus all -it --rm --ipc=host nvcr.io/nvidia/pytorch:23.10-py3
