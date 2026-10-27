.. _installation_rocm_zh_cn:

使用 ROCm 安装
====================

:doc:`English <amd-installation>`

.. note::


   本页是上游 ROCm 参考。Mil 的安装与实验环境见 ``artifact/env/README.zh-CN.md``。


vLLM 从 0.2.4 起支持 AMD GPU 的 ROCm 推理与服务。本参考版本不支持 ROCm AWQ，
但已移植 SqueezeLLM；支持 FP16 和 BF16。

要求
----

* 操作系统：Linux
* Python：3.8–3.11
* GPU：MI200 系列（gfx90a）、MI300（gfx942）
* PyTorch：2.0.1／2.1.1／2.2
* ROCm：5.7（Python 3.10 验证）或 6.0（Python 3.9 验证）

安装选项：

#. :ref:`预装 vLLM 的 Docker 镜像（推荐） <quick_start_docker_rocm_zh_cn>`
#. :ref:`本地源码构建 <build_from_source_rocm_zh_cn>`
#. :ref:`Docker 源码构建 <build_from_source_docker_rocm_zh_cn>`

.. _quick_start_docker_rocm_zh_cn:

推荐选项 1：使用预装 vLLM 的 Docker 镜像
--------------------------------------------------------

此选项仅适用于 ROCm 5.7：

.. code-block:: console

    $ docker pull embeddedllminfo/vllm-rocm:vllm-v0.2.4
    $ docker run -it \
       --network=host \
       --group-add=video \
       --ipc=host \
       --cap-add=SYS_PTRACE \
       --security-opt seccomp=unconfined \
       --device /dev/kfd \
       --device /dev/dri \
       -v <path/to/model>:/app/model \
       embeddedllminfo/vllm-rocm \
       bash

.. _build_from_source_rocm_zh_cn:

选项 2：从源码构建
--------------------

可从源码构建并安装 vLLM。

以下流程仅针对 ROCm 5.7；本上游文档对应时期，PyTorch 网站尚未提供 ROCm 6.0 wheel。

先安装依赖；若环境或容器中已有相应依赖，可跳过此步。

- `ROCm <https://rocm.docs.amd.com/en/latest/deploy/linux/index.html>`_
- `Pytorch <https://pytorch.org/>`_

    .. code-block:: console

        $ pip install torch==2.2.0.dev20231206+rocm5.7 --index-url https://download.pytorch.org/whl/nightly/rocm5.7 # tested version

按 `AMD GPU／ROCm 指南 <https://github.com/ROCmSoftwarePlatform/flash-attention/tree/flash_attention_for_rocm#amd-gpurocm-support>`__
安装 ROCm Flash Attention（v2.0.4）。

.. note::


    - ROCm 5.7 配合 PyTorch 2.1.0 或更高版本，无需应用 ``hipify_python.patch``，可直接构建。
    - 若安装失败，可尝试提交 ``6fd2f8e572805681cd67ef8596c7e2ce521ed3c6``。
    - ROCm Flash-attention-2（v2.0.4）不支持滑动窗口注意力。
    - 必要时将 ninja 降至 1.10，例如 ``pip install ninja==1.10.2.4``。


无依赖安装 ``xformers==0.0.23``，再应用 ROCm Flash Attention 兼容补丁。

    .. code-block:: console

        $ pip install xformers==0.0.23 --no-deps
        $ bash patch_xformers.rocm.sh

构建 vLLM，通常需要 5–10 分钟。本参考流程使用 ``python setup.py install``。

此处的 ROCm 安装流程不支持直接使用 ``pip install .``。

    .. code-block:: console

        $ cd vllm
        $ pip install -U -r requirements-rocm.txt
        $ python setup.py install # This may take 5-10 minutes. Currently, `pip install .`` does not work for ROCm installation

.. _build_from_source_docker_rocm_zh_cn:

选项 3：在 Docker 中从源码构建
----------------------------------------

可从源码构建并安装 vLLM。

使用 ``Dockerfile.rocm`` 构建镜像并启动容器。

``Dockerfile.rocm`` 支持 ROCm 5.7、6.0 及更高版本，可通过以下构建参数定制。

* ``BASE_IMAGE``：PyTorch ROCm 基础镜像；已测试 ROCm 5.7 和 6.0。
  默认 ``rocm/pytorch:rocm6.0_ubuntu20.04_py3.9_pytorch_2.1.1``。
* ``FX_GFX_ARCHS``：Flash Attention 构建架构，例如 MI200／MI300 使用 ``gfx90a;gfx942``；
  默认同样为 ``gfx90a;gfx942``。
* ``FA_BRANCH``：ROCm Flash Attention 仓库的构建分支，默认 ``3d2b6f5``。

通过 ``docker build`` 的 ``--build-arg`` 传入这些值。

例如构建 ROCm 5.7 镜像：

.. code-block:: console

    $ docker build --build-arg BASE_IMAGE="rocm/pytorch:rocm5.7_ubuntu22.04_py3.10_pytorch_2.0.1" \
       -f Dockerfile.rocm -t vllm-rocm .

ROCm 6.0 可使用默认配置，按以下命令构建并运行：

.. code-block:: console

    $ docker build -f Dockerfile.rocm -t vllm-rocm .
    $ docker run -it \
       --network=host \
       --group-add=video \
       --ipc=host \
       --cap-add=SYS_PTRACE \
       --security-opt seccomp=unconfined \
       --device /dev/kfd \
       --device /dev/dri \
       -v <path/to/model>:/app/model \
       vllm-rocm \
       bash

若直接在本机安装或使用新 Docker 基础镜像（如 rocm/pytorch），可按以下步骤操作。

先安装依赖；若环境或容器中已有相应依赖，可跳过此步。

- `ROCm <https://rocm.docs.amd.com/en/latest/deploy/linux/index.html>`_
- `Pytorch <https://pytorch.org/>`_
- `hipBLAS <https://rocm.docs.amd.com/projects/hipBLAS/en/latest/install.html>`_

依赖包括 ROCm、PyTorch 和 hipBLAS。

按 `AMD GPU／ROCm 指南 <https://github.com/ROCmSoftwarePlatform/flash-attention/tree/flash_attention_for_rocm#amd-gpurocm-support>`__
安装 ROCm Flash Attention（v2.0.4）。

.. note::


    - ROCm 5.7 配合 PyTorch 2.1.0 或更高版本，无需应用 ``hipify_python.patch``，可直接构建。
    - 若安装失败，可尝试提交 ``6fd2f8e572805681cd67ef8596c7e2ce521ed3c6``。
    - ROCm Flash-attention-2（v2.0.4）不支持滑动窗口注意力。
    - 必要时将 ninja 降至 1.10，例如 ``pip install ninja==1.10.2.4``。


无依赖安装 ``xformers==0.0.23``，再应用 ROCm Flash Attention 兼容补丁。

    .. code-block:: console

        $ pip install xformers==0.0.23 --no-deps
        $ bash patch_xformers.rocm.sh

构建 vLLM，通常需要 5–10 分钟。本参考流程使用 ``python setup.py install``。

    .. code-block:: console

        $ cd vllm
        $ pip install -U -r requirements-rocm.txt
        $ python setup.py install # This may take 5-10 minutes.

.. note::


    用 ``benchmark_thoughput.py`` 测试安装时若进程挂起，可尝试 ``--enforce-eager``。
