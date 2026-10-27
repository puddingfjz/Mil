.. _supported_models_zh_cn:

支持的模型
==========

:doc:`English <supported_models>`

vLLM 支持多种 Hugging Face Transformers 生成式模型。下表列出支持的架构及代表模型。

.. list-table::
  :widths: 25 25 50
  :header-rows: 1

  * - 架构
    - 模型
    - Hugging Face 示例模型
  * - :code:`AquilaForCausalLM`
    - Aquila
    - :code:`BAAI/Aquila-7B`, :code:`BAAI/AquilaChat-7B` 等
  * - :code:`BaiChuanForCausalLM`
    - Baichuan
    - :code:`baichuan-inc/Baichuan2-13B-Chat`, :code:`baichuan-inc/Baichuan-7B` 等
  * - :code:`ChatGLMModel`
    - ChatGLM
    - :code:`THUDM/chatglm2-6b`, :code:`THUDM/chatglm3-6b` 等
  * - :code:`DeciLMForCausalLM`
    - DeciLM
    - :code:`Deci/DeciLM-7B`, :code:`Deci/DeciLM-7B-instruct` 等
  * - :code:`BloomForCausalLM`
    - BLOOM, BLOOMZ, BLOOMChat
    - :code:`bigscience/bloom`, :code:`bigscience/bloomz` 等
  * - :code:`FalconForCausalLM`
    - Falcon
    - :code:`tiiuae/falcon-7b`, :code:`tiiuae/falcon-40b`, :code:`tiiuae/falcon-rw-7b` 等
  * - :code:`GPT2LMHeadModel`
    - GPT-2
    - :code:`gpt2`, :code:`gpt2-xl` 等
  * - :code:`GPTBigCodeForCausalLM`
    - StarCoder, SantaCoder, WizardCoder
    - :code:`bigcode/starcoder`, :code:`bigcode/gpt_bigcode-santacoder`, :code:`WizardLM/WizardCoder-15B-V1.0` 等
  * - :code:`GPTJForCausalLM`
    - GPT-J
    - :code:`EleutherAI/gpt-j-6b`, :code:`nomic-ai/gpt4all-j` 等
  * - :code:`GPTNeoXForCausalLM`
    - GPT-NeoX, Pythia, OpenAssistant, Dolly V2, StableLM
    - :code:`EleutherAI/gpt-neox-20b`, :code:`EleutherAI/pythia-12b`, :code:`OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5`, :code:`databricks/dolly-v2-12b`, :code:`stabilityai/stablelm-tuned-alpha-7b` 等
  * - :code:`InternLMForCausalLM`
    - InternLM
    - :code:`internlm/internlm-7b`, :code:`internlm/internlm-chat-7b` 等
  * - :code:`InternLM2ForCausalLM`
    - InternLM2
    - :code:`internlm/internlm2-7b`, :code:`internlm/internlm2-chat-7b` 等
  * - :code:`LlamaForCausalLM`
    - LLaMA, LLaMA-2, Vicuna, Alpaca, Koala, Guanaco
    - :code:`meta-llama/Llama-2-13b-hf`, :code:`meta-llama/Llama-2-70b-hf`, :code:`openlm-research/open_llama_13b`, :code:`lmsys/vicuna-13b-v1.3`, :code:`young-geng/koala` 等
  * - :code:`MistralForCausalLM`
    - Mistral, Mistral-Instruct
    - :code:`mistralai/Mistral-7B-v0.1`, :code:`mistralai/Mistral-7B-Instruct-v0.1` 等
  * - :code:`MixtralForCausalLM`
    - Mixtral-8x7B, Mixtral-8x7B-Instruct
    - :code:`mistralai/Mixtral-8x7B-v0.1`, :code:`mistralai/Mixtral-8x7B-Instruct-v0.1` 等
  * - :code:`MPTForCausalLM`
    - MPT, MPT-Instruct, MPT-Chat, MPT-StoryWriter
    - :code:`mosaicml/mpt-7b`, :code:`mosaicml/mpt-7b-storywriter`, :code:`mosaicml/mpt-30b` 等
  * - :code:`OPTForCausalLM`
    - OPT, OPT-IML
    - :code:`facebook/opt-66b`, :code:`facebook/opt-iml-max-30b` 等
  * - :code:`PhiForCausalLM`
    - Phi
    - :code:`microsoft/phi-1_5`, :code:`microsoft/phi-2` 等
  * - :code:`QWenLMHeadModel`
    - Qwen
    - :code:`Qwen/Qwen-7B`, :code:`Qwen/Qwen-7B-Chat` 等
  * - :code:`Qwen2ForCausalLM`
    - Qwen2
    - :code:`Qwen/Qwen2-beta-7B`, :code:`Qwen/Qwen2-beta-7B-Chat` 等
  * - :code:`StableLMEpochForCausalLM`
    - StableLM
    - :code:`stabilityai/stablelm-3b-4e1t/` , :code:`stabilityai/stablelm-base-alpha-7b-v2` 等
  * - :code:`YiForCausalLM`
    - Yi
    - :code:`01-ai/Yi-6B`, :code:`01-ai/Yi-34B` 等

若模型采用上述架构，可直接用 vLLM 运行；否则参考 :ref:`添加新模型 <adding_a_new_model_zh_cn>`
实现支持，或在 `GitHub 问题页面 <https://github.com/vllm-project/vllm/issues>`_ 提交 issue。

.. note::


    ROCm 版对 Mistral 和 Mixtral 的支持限于不超过 4096 的上下文长度。


.. tip::


    可运行下面的程序检查模型是否受支持，填写模型名称或路径：

    .. code-block:: python

        from vllm import LLM

        llm = LLM(model=...)  # Name or path of your model
        output = llm.generate("Hello, my name is")
        print(output)

    成功生成文本表示该模型受支持。


.. tip::


    若使用 ModelScope 而非 Hugging Face Hub，设置以下环境变量：

    .. code-block:: shell

       $ export VLLM_USE_MODELSCOPE=True

    同时设置 :code:`trust_remote_code=True`，并填写模型名称／路径和 revision。

    .. code-block:: python

        from vllm import LLM

        llm = LLM(model=..., revision=..., trust_remote_code=True)  # Name or path of your model
        output = llm.generate("Hello, my name is")
        print(output)
