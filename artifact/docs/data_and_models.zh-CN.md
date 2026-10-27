<a id="data-and-model-preparation"></a>

# 数据与模型准备

[English](data_and_models.md)

调度器和推理入口使用与预检相同的固定本地模型快照，不会隐式下载 `main`。
数据集与成本资源路径由 `MIL_DATA_ROOT` 和仓库相对路径提供；doctor 检查活动代码中的
机器专属硬编码路径。

<a id="model-policy"></a>

## 模型规则

- 模型权重不提交到本仓库。
- 每个模型固定到不可变的 Hugging Face revision。
- 逐模型记录固定的模型卡／许可证证据、访问要求和已知许可限制；权重由运行者自行提供。
- 可以使用本地缓存，但运行元数据会记录实际解析的快照 revision。

`MIL_MODEL_ROOT` 指定 Hugging Face 缓存根目录；未设置时依次使用 `HF_HUB_CACHE`、
`HF_HOME/hub` 或默认缓存。运行前需准备快照，缓存缺失视为错误。
预检返回每个模型的 revision、快照绝对路径和配置校验值。运行器将映射固定在运行／任务
元数据中，并通过 `MIL_MODEL_SNAPSHOTS` 传给子进程。调度配置读取、请求分词、运行时
prompt tokenizer 和推理权重均使用该映射。任务启用 Hugging Face／Transformers
离线模式；快照缺失、变更或未被选中时失败，不回退到其他缓存或版本。
初始执行计划使用的引擎配置也遵循此规则。仅显式的 CPU-only 调度路径为固定配置读取器
保留模型 ID，无需模型缓存；正常实验运行器在启动运行时任务前清除 CPU-only 模式。

预检只检查展开后所选任务（`--suite`、`--method`、`--parameter`）需要的资源。
Router 检查所选问题分区和 5 个路由模型；Ensembling 检查模型集或 Oracle 子集；
Chain Summary 检查两类书籍输入及摘要／评价模型；Mixed 使用两者并集。
所有任务使用随包提供的输出长度输入，选中资源缺失仍会失败。
模型／数据磁盘需求按所选内容、初始化开销和运行预留估计，不按完整清单估计。
资源准备命令在未显式筛选资源时仍可能检查完整清单。

`manifests/models.yaml` 保存固定 revision、快照校验值和访问证据；版本来源引用固定
Git 提交中的文档。这些引用不证明每次历史运行都属于最终论文样本。

固定模型清单的访问要求如下：

| 模型 | 固定证据中的条款 | 使用要求 |
| --- | --- | --- |
| Llama-2-70B、CodeLlama-34B | Llama 2 | 完成 Meta 的受限访问联系／授权流程 |
| Mixtral-8x7B、Mistral-7B | Apache-2.0 | 固定模型卡含 Hugging Face 联系门槛元数据；需重新检查托管平台当前门槛 |
| WizardLM-13B、Vicuna-13B | Llama 2 | 遵守上游及 Llama 2 条款 |
| OASST Pythia-12B | Apache-2.0 | 遵守上游条款 |
| Alpaca-13B | 未声明 | 在 LLaMA 衍生权重／数据条款明确前，不获取或用于发布 |
| Baize-v2-13B | CC-BY-NC-4.0 | 非商业使用，并审阅底层 LLaMA 条款 |
| Koala-13B | 自定义／复合，仅学术研究 | 不可商用；审阅所引 LLaMA、OpenAI 数据及 ShareGPT 条款 |
| Dolly-v2-12B | MIT | 遵守上游条款；模型卡另将训练数据标为 CC-BY-SA |
| MPT-7B-chat | CC-BY-NC-SA-4.0 | 仅限非商业使用 |
| ChatGLM3-6B | 自定义 ChatGLM3 条款 | 商业使用需登记 |
| StableLM tuned alpha | CC-BY-NC-SA-4.0 | 仅限非商业使用 |

表格概括 `manifests/models.yaml` 中具有精确哈希的证据，不扩大上游授权。
访问门槛可独立于 Git revision 变化，因此缓存可访问不等于新运行者已获得授权。

清单还固定 `config.json` 的 SHA-256、快照／权重文件数与总字节数，以及排序后的
权重路径／大小摘要。准备命令和 doctor 核对这些值，并对所列模型卡／许可证文件计算哈希。
轻量权重布局摘要可检测缺失或截断的分片，无需每次预检重读约 0.9 TB 数据。
它不是权重内容哈希；不可变的仓库 revision 仍是上游内容标识。

<a id="dataset-policy"></a>

## 数据集规则

- `MIL_DATA_ROOT` 指定外部数据集根目录。
- 仓库自有的预处理输入使用 `${MIL_REPO_ROOT}`。
- 每份发布的预处理数据记录 SHA-256、样本数、预处理命令、上游 URL 和许可证。
- 用于强制已观测输出长度的文件作为版本化输入；新生成文件写入运行目录。

数据清单区分字节／语义来源与再分发许可。以下来源核对完成于 2026-09-06：

- MixInstruct 固定到 `c7f77a4ef0515a99d1752c6387c22366d6922da6`。
  `train_data_prepared.jsonl` 的上游 LFS OID 与仓库文件 SHA-256 完全一致，
  因此未推定额外预处理步骤。上游数据卡声明 MIT。
- 两份 RouterBench JSON 分区来自 revision `784021482c3f320c6619ed4b3bb3b41a21424fcb`
  的 `routerbench_5shot.pkl`。解析列表字面量、以 LF 连接消息并映射 5 个路由目标后，
  30,379 条记录中有 30,378 条与 `oracle_model_to_route_to` 一致。
  唯一例外为 `mbpp.dev.101`：WizardLM、Mixtral 和 CodeLlama 均得 1.0，
  本地保留 WizardLM 响应，而源 oracle 字段为 Mixtral。
  6,856 条非选择题 prompt／response 均完全一致。官方代码仓库为 MIT，但数据卡未声明
  数据许可证，因此不能假定预处理 RouterBench 数据可再分发。
- BookSum 固定到 `c62321036e5647db5767ecaff139912b554dc938`。
  `train.csv` 的 9,600 行对应 4,845 个唯一 `chapter_path`，重复路径的章节文本相同。
  将路径映射到章节并使用 BooookScore 的 2048-token `cl100k_base` 分块器后，
  4,845 份本地分块列表均完全一致。数据卡声明 BSD-3-Clause，另有研究用途和底层权利条款，
  发布时仍需审阅。
- `all_books_chunked_2048.pkl` 的文件名与命令对应 BooookScore revision
  `094bf69bc55317b728b4b2bb679c287e0176ae6c`：`python -m booookscore.chunk
  --chunk_size 2048 --input_path all_books.pkl --output_path all_books_chunked_2048.pkl`。
  分块器使用 `cl100k_base`，上游固定 `tiktoken==0.5.2`。
  私有 `all_books.pkl` 无法用于字节比较；项目说明仅通过申请向学术实验室提供数据。
  MIT 代码许可证不代表数据再分发授权。

清单也记录本地 RouterBench 文件的创建提交。BookSum 比较证明预处理对象语义一致，
不保证不同 Python 版本的 pickle 字节相同。论文计时运行仍以预处理文件的精确校验值为准。

运行 `bash artifact/scripts/prepare_data.sh` 验证所有解析目标。
缺少预处理文件时，显式传入 `--source DATASET_ID=/path/to/file`，并选择
`--strategy link` 或 `--strategy copy`。暂存前核对来源校验值；有效目标可复用，
已有不匹配文件不会覆盖。该命令准备已知预处理输入，不隐式下载大型或受限数据，
也不绕过上游条款。计时运行要求重建对象仍匹配 `manifests/datasets.yaml` 的文件校验值。

全新工作树可直接获取精确的公开 MixInstruct 对象，无需手动浏览器操作：

```bash
bash artifact/scripts/prepare_data.sh \
  --dataset mixinstruct_prepared --download-public
```

下载需要显式启用，固定 revision，并通过校验值检查。默认在仓库忽略目录中创建指向
Hugging Face 缓存的链接；`--strategy copy` 创建实际副本。
没有 `automatic_download: allowed` 的清单项会拒绝自动下载，因此该开关不能绕过
RouterBench 未解决的许可或 BookSum／BooookScore 的受限、私有数据流程。

运行 `bash artifact/scripts/prepare_models.sh` 验证缓存中的固定 revision。
只有同时指定 `--download` 和 `--acknowledge-access-terms` 才会下载；模型访问条款和
认证由运行者负责。确认条款不能覆盖未解决的清单许可：Alpaca-13B 在人工审阅前拒绝自动下载。
模型始终按 `models.yaml` 的精确 revision 获取。验证还要求有效的 `config.json` 对象、
至少一个权重文件、无失效缓存链接、匹配的快照／权重布局，以及配置和许可证据哈希。
仅目录名称与 revision 相符不足以构成完整快照。

<a id="cost-profile-policy"></a>

## 成本测量资源规则

参数化离线构建器见[资源生成](resource_generation.zh-CN.md)。它支持显式选择模型、日志、
执行计划和输出，写入新资源而不向旧源码追加。`plan` 只记录采集命令；`build` 无需 GPU
或模型装载。支持合成测试和已有 ChatGLM／StableLM 测量的重建。

默认入口选择 `--machine_name zxcpu`，使用固定的 `get_my_cost_table_directly_zxcpu.py`。
由于 `search_exec_plans.py` 在导入时立即加载两个序列化表模块，两者均需提供。
doctor 直接校验这些运行时文件及表加载器。使用固定表不需要原始 profiling 日志。

保留的命令归档对应提交 `43561cb0af9128d440aecb0f2bfdadfa6c0079a6` 的常用采集设置：
合成输入／输出长度 16/16、1 个请求、eager 模式、TP 1/2/4、temperature 1.0、
GPU ratio 0.9、weight-load degree 2、vLLM 后端。表构建器每点使用两次测量、
1 ms 阈值和 `(1, 1, -1, 0)` 采样配置。
zxcpu 序列化表最近一次更新提交为 `de1677dcb7e2a216172aedf3262b19b106f17786`。

序列化表标识实际成本数据。完整 zxcpu 原始日志及不可变的 GPU 拓扑记录不可用，
因此新的测量复现仍未验证。原始归档是可选来源证据：缺失时 doctor 给出警告；
序列化运行时表缺失或变更则属于错误。

成本采集使用常规 Hugging Face 认证（`HF_TOKEN` 或本地 token 存储），采集器不内嵌个人令牌。

<a id="cpu-scheduling-input-bundle"></a>

## CPU 调度输入包

`artifact/inputs/frozen/*.json` 包含原工作负载构造器生成的数字化请求 ID、token 长度、
应用依赖和来源哈希。7 份输入包不含请求文本或权重。
`artifact/inputs/model_configurations.json` 固定 CPU 成本模拟器所需的 14 个模型配置对象。
这些文件支持全新工作树中的离线调度，不授权再分发上游文本或权重，上述访问要求仍适用。
发布包不分发实验日志、结果索引或绘图测量。
