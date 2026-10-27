.. _fp8_e5m2_kv_cache_zh_cn:

FP8 E5M2 KV 缓存
============================

:doc:`English <fp8_e5m2_kv_cache>`

int8／int4 量化需要额外 GPU 内存保存缩放系数，降低预期显存收益。
FP8 保留 2–3 位尾数，可与 float／FP16／BF16 相互转换。

下面展示启用该功能的方法：设置 ``kv_cache_dtype="fp8_e5m2"``，
创建模型及采样参数，生成并打印输出。

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
    llm = LLM(model="facebook/opt-125m", kv_cache_dtype="fp8_e5m2")
    # Generate texts from the prompts. The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    outputs = llm.generate(prompts, sampling_params)
    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
