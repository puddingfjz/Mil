.. _auto_awq_zh_cn:

AutoAWQ 量化
====================

:doc:`English <auto_awq>`

.. warning::


   本参考实现的 AWQ 优化有限，未量化模型通常具有更好的精度和吞吐量。
   AWQ 可降低内存占用，更适合少量并发请求的低延迟推理；此实现的吞吐量低于未量化版本。


可使用 `AutoAWQ <https://github.com/casper-hansen/AutoAWQ>`_ 创建 4 位量化模型，将 FP16 精度降为 INT4，文件大小约减少 70%；
主要收益是降低延迟与内存占用。

可安装 AutoAWQ 量化自己的模型，或从
`Hugging Face AWQ 模型列表 <https://huggingface.co/models?sort=trending&search=awq>`_ 中选择。

.. code-block:: console

    $ pip install autoawq

安装后即可量化模型。下例加载 Vicuna 7B v1.5 及 tokenizer，执行量化并保存：

.. code-block:: python

    from awq import AutoAWQForCausalLM
    from transformers import AutoTokenizer

    model_path = 'lmsys/vicuna-7b-v1.5'
    quant_path = 'vicuna-7b-v1.5-awq'
    quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM" }

    # Load model
    model = AutoAWQForCausalLM.from_pretrained(model_path, **{"low_cpu_mem_usage": True})
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Quantize
    model.quantize(tokenizer, quant_config=quant_config)

    # Save quantized model
    model.save_quantized(quant_path)
    tokenizer.save_pretrained(quant_path)

以下命令在 vLLM 中运行 TheBloke/Llama-2-7b-Chat-AWQ：

.. code-block:: console

    $ python examples/llm_engine_example.py --model TheBloke/Llama-2-7b-Chat-AWQ --quantization awq

也可通过 LLM 入口直接运行 AWQ 模型。下例设置 prompt 和采样参数，创建模型、生成并打印结果：

.. code-block:: python

    from vllm import LLM, SamplingParams

    # Sample prompts.
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    # Create a sampling params object.
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

    # Create an LLM.
    llm = LLM(model="TheBloke/Llama-2-7b-Chat-AWQ", quantization="AWQ")
    # Generate texts from the prompts. The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    outputs = llm.generate(prompts, sampling_params)
    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
