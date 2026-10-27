.. _adding_a_new_model_zh_cn:

添加新模型
==========

:doc:`English <adding_model>`

本指南概述如何将 `Hugging Face Transformers <https://github.com/huggingface/transformers>`_
模型接入 vLLM。

.. note::


    接入复杂度取决于模型架构。与已有模型架构相似时通常较简单；
    若包含新算子（如新的注意力机制），则需要更多实现工作。


.. tip::


    模型接入遇到问题时，可在
    `GitHub 问题页面 <https://github.com/vllm-project/vllm/issues>`_ 提交 issue 寻求帮助。


0. Fork vLLM 仓库
------------------------------

先 fork `vLLM 仓库 <https://github.com/vllm-project/vllm>`_，再
:ref:`从源码构建 <build_from_source_zh_cn>`，以便修改和测试模型。

1. 引入模型代码
------------------

从 Hugging Face Transformers 获取 PyTorch 模型代码，放入 ``vllm/model_executor/models``。
vLLM 的 `OPT 实现 <https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/opt.py>`_
即改编自 Hugging Face 的
`modeling_opt.py <https://github.com/huggingface/transformers/blob/main/src/transformers/models/opt/modeling_opt.py>`_。

.. warning::


    复制模型代码时，应审阅并遵守其版权和许可证条款。


2. 改写 forward 方法
--------------------------------

按以下步骤改写模型的 :code:`forward` 方法：

1. 删除只用于训练等不需要的代码。
2. 按下面差异修改输入参数。

.. code-block:: diff

    def forward(
        self,
        input_ids: torch.Tensor,
    -    attention_mask: Optional[torch.Tensor] = None,
    -    position_ids: Optional[torch.LongTensor] = None,
    -    past_key_values: Optional[List[torch.FloatTensor]] = None,
    -    inputs_embeds: Optional[torch.FloatTensor] = None,
    -    labels: Optional[torch.LongTensor] = None,
    -    use_cache: Optional[bool] = None,
    -    output_attentions: Optional[bool] = None,
    -    output_hidden_states: Optional[bool] = None,
    -    return_dict: Optional[bool] = None,
    -) -> Union[Tuple, CausalLMOutputWithPast]:
    +    positions: torch.Tensor,
    +    kv_caches: List[KVCache],
    +    input_metadata: InputMetadata,
    +) -> Optional[SamplerOutput]:

1. 将 :code:`input_ids` 和 :code:`positions` 作为展平张量处理。
2. 按模型架构替换注意力操作，使用 :code:`PagedAttention`、:code:`PagedAttentionWithRoPE`
   或 :code:`PagedAttentionWithALiBi`。

.. note::


    vLLM 支持基本多头注意力及旋转位置编码变体；若模型采用其他注意力机制，
    需要实现新的注意力层。


3. 可选：实现张量并行和量化支持
----------------------------------

模型无法放入单张 GPU 时，可用张量并行。将线性层和嵌入层替换为并行版本：
``nn.Embedding`` 对应 :code:`VocabParallelEmbedding`，输出 LM head 对应
:code:`ParallelLMHead`。线性层可选择：

* :code:`ReplicatedLinear`：在多个 GPU 上复制输入和权重，不节省内存。
* :code:`RowParallelLinear`：输入按隐藏维度、权重按行（输入维度）切分，矩阵乘后执行 all-reduce；
  常用于第二个 FFN 层和注意力输出变换。
* :code:`ColumnParallelLinear`：复制输入，权重按列（输出维度）切分，输出也按列切分；
  常用于第一个 FFN 层及原 Transformer 中独立的 QKV 变换。
* :code:`MergedColumnParallelLinear`：合并多个列并行算子，常用于带权重激活（如 SiLU）的
  第一个 FFN 层，负责多个权重矩阵的分片加载。
* :code:`QKVParallelLinear`：多头或分组查询注意力的 Q/K/V 投影。KV 头数小于进程总数时
  正确复制 KV 头，并处理多个权重矩阵的加载和复制。

上述线性层均接收 ``linear_method``，vLLM 按量化方案设置它以支持权重量化。

4. 实现权重加载
------------------

在 :code:`*ForCausalLM` 类中实现 :code:`load_weights`，从 Hugging Face checkpoint
加载并分配权重。对于 ``MergedColumnParallelLinear`` 和 ``QKVParallelLinear``，
若原模型使用独立权重矩阵，需要分别加载各部分。

5. 注册模型
--------------

将 :code:`*ForCausalLM` 类加入 ``vllm/model_executor/models/__init__.py``，
并在 ``vllm/model_executor/model_loader.py`` 的 :code:`_MODEL_REGISTRY` 注册。
