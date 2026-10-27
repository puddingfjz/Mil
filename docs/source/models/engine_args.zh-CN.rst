.. _engine_args_zh_cn:

引擎参数
========

:doc:`English <engine_args>`

下文说明各个 vLLM 引擎参数。

.. option:: --model <model_name_or_path>
    :noindex:


    Hugging Face 模型名称或路径。


.. option:: --tokenizer <tokenizer_name_or_path>
    :noindex:


    Hugging Face tokenizer 名称或路径。


.. option:: --revision <revision>
    :noindex:


    指定模型版本，可为分支、标签或提交 ID；省略时使用默认版本。


.. option:: --tokenizer-revision <revision>
    :noindex:


    指定 tokenizer 版本，可为分支、标签或提交 ID；省略时使用默认版本。


.. option:: --tokenizer-mode {auto,slow}
    :noindex:


    tokenizer 模式。

    * ``auto``：有快速 tokenizer 时使用快速实现。
    * ``slow``：始终使用慢速实现。


.. option:: --trust-remote-code
    :noindex:


    信任 Hugging Face 远程代码。


.. option:: --download-dir <directory>
    :noindex:


    权重下载与加载目录，默认使用 Hugging Face 缓存目录。


.. option:: --load-format {auto,pt,safetensors,npcache,dummy}
    :noindex:


    待加载权重的格式。

    * ``auto``：优先 safetensors；不可用时回退到 PyTorch bin。
    * ``pt``：加载 PyTorch 权重。
    * ``safetensors``：加载 safetensors 权重。
    * ``npcache``：加载 PyTorch 权重并保存 NumPy 缓存，加速后续加载。
    * ``dummy``：随机初始化权重，主要用于性能采集。


.. option:: --dtype {auto,half,float16,bfloat16,float,float32}
    :noindex:


    模型权重和激活值的数据类型。

    * ``auto``：FP32／FP16 模型使用 FP16，BF16 模型使用 BF16。
    * ``half``：FP16，建议用于 AWQ。
    * ``float16``：等同 ``half``。
    * ``bfloat16``：在精度与范围之间折中。
    * ``float``：FP32 简写。
    * ``float32``：FP32。


.. option:: --max-model-len <length>
    :noindex:


    模型上下文长度；未指定时从模型配置推导。


.. option:: --worker-use-ray
    :noindex:


    使用 Ray 进行分布式服务；使用多张 GPU 时自动设置。


.. option:: --pipeline-parallel-size (-pp) <size>
    :noindex:


    流水线阶段数。


.. option:: --tensor-parallel-size (-tp) <size>
    :noindex:


    张量并行副本数。


.. option:: --max-parallel-loading-workers <workers>
    :noindex:


    将模型分多批顺序加载，避免张量并行大模型加载时耗尽主机内存。


.. option:: --block-size {8,16,32}
    :noindex:


    连续 token 块大小。


.. option:: --seed <seed>
    :noindex:


    操作随机种子。


.. option:: --swap-space <size>
    :noindex:


    每张 GPU 对应的 CPU 交换空间，单位 GiB。


.. option:: --gpu-memory-utilization <fraction>
    :noindex:


    模型执行器可使用的 GPU 显存比例，范围 0–1；例如 0.5 表示 50%，默认 0.9。


.. option:: --max-num-batched-tokens <tokens>
    :noindex:


    每次迭代的最大批处理 token 数。


.. option:: --max-num-seqs <sequences>
    :noindex:


    每次迭代的最大序列数。


.. option:: --max-paddings <paddings>
    :noindex:


    批次中的最大填充数量。


.. option:: --disable-log-stats
    :noindex:


    禁用统计日志。


.. option:: --quantization (-q) {awq,squeezellm,None}
    :noindex:


    权重量化方法。
