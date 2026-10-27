.. _distributed_serving_zh_cn:

分布式推理与服务
================

:doc:`English <distributed_serving>`

vLLM 支持分布式张量并行推理与服务，采用 Megatron-LM 张量并行算法，并使用 Ray
管理分布式运行时。先安装 Ray：

.. code-block:: console

    $ pip install ray

通过 ``LLM`` 类运行多 GPU 推理时，将 :code:`tensor_parallel_size` 设为 GPU 数；
下面示例使用 4 张 GPU：

.. code-block:: python

    from vllm import LLM
    llm = LLM("facebook/opt-13b", tensor_parallel_size=4)
    output = llm.generate("San Franciso is a")

启动多 GPU 服务时传入 :code:`--tensor-parallel-size`，下面使用 4 张 GPU：

.. code-block:: console

    $ python -m vllm.entrypoints.api_server \
    $     --model facebook/opt-13b \
    $     --tensor-parallel-size 4

扩展到多机前，先通过命令行启动 Ray 集群。分别在头节点和工作节点运行：

.. code-block:: console

    $ # On head node
    $ ray start --head

    $ # On worker nodes
    $ ray start --address=<ray-head-address>

随后在头节点启动 vLLM 进程，将 :code:`tensor_parallel_size` 设为所有机器的 GPU 总数，
即可进行多机推理与服务。
