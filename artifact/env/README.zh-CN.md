<a id="environments"></a>

# 环境配置

[English](README.md)

主要安装方式为原生 Conda。Docker 为可选项，CPU 验证和原生 GPU 工作流均不需要 Docker。

<a id="required-vllm-runtime"></a>

## 所需 vLLM 运行时

Mil 需要本仓库中的定制 vLLM Python 和 CUDA 源码。CPU 调度导入本地 Python 源码；
GPU 执行还需要从同一源码树构建原生扩展。

`runtime.yaml` 在 `custom_vllm` 下单独列出该运行时，与普通包依赖区分。
包版本为 `0.3.0`，源码校验值与本地构建记录共同标识实际实现。

<a id="tooling-and-plotting"></a>

## 工具与绘图

```bash
conda env create --file artifact/env/environment.yml
conda activate mil-artifact
bash artifact/scripts/validate.sh
```

轻量环境与 `requirements-lock.txt` 固定配置处理、数值分析和绘图依赖。
绘图依赖 Matplotlib 3.9.4、NumPy 1.26.3 和 Pillow 11.3.0。
绘图读取显式提供的数据，发布包不附带论文数值或历史像素基准。

<a id="actual-cpu-scheduling"></a>

## 实际 CPU 调度

对于 Linux x86-64 上的固定数字化工作负载，创建新环境，先安装 CPU-only PyTorch，
再安装调度依赖：

```bash
conda env create --name mil-artifact-cpu --file artifact/env/environment.yml
conda activate mil-artifact-cpu
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=""
python -m pip install --no-cache-dir --no-deps \
  --index-url https://download.pytorch.org/whl/cpu torch==2.1.2
python -m pip install --no-cache-dir --index-url https://pypi.org/simple \
  -r artifact/env/requirements-cpu.txt
python -m pip check
python -c 'import torch; assert torch.version.cuda is None; print(torch.__version__)'
bash artifact/scripts/validate.sh
python -m artifact.mil_artifact.validate_cpu \
  --cases ensemble/mil chain/light_rule_star --workers 1 \
  --output /tmp/mil-clean-cpu-check-01
```

[PyTorch 官方 CPU wheel 索引](https://pytorch.org/get-started/previous-versions/#v212)
提供 `torch==2.1.2+cpu`。`requirements-cpu.txt` 将 `runtime-freeze.txt` 用作版本约束，
而非完整安装列表，因此不安装 CUDA 库、xFormers 或 Triton。此路径无需将仓库安装为
可编辑 GPU 包，也无需构建 CUDA 扩展；从源码根目录运行即可。

启动器禁止 GPU 算子，从数字化输入构造模型配置，并启动私有的本地 Ray CPU 服务。
上述两个用例无需模型权重、外部模型缓存或 Gurobi 授权。数据准备和 Saturn 需要额外
输入与依赖，不在这两个用例的范围内，见 [CPU 快速开始](../docs/cpu_quickstart.zh-CN.md)。

每次运行使用新输出目录。日常工作可使用已有环境，但这不构成全新安装的验证证据。
`runtime.yaml` 分开记录 CPU 验收与尚未完成的 GPU 验证状态。

<a id="gpu-runtime"></a>

## GPU 运行时

在具备 CUDA 12.1 的兼容主机上准备独立环境：

```bash
conda create --yes --name mil-artifact-runtime \
  python=3.9.20 pip=24.2 setuptools=75.1.0 wheel=0.44.0
conda activate mil-artifact-runtime
unset PYTHONPATH PYTHONHOME MIL_CPU_SCHEDULING VLLM_USE_PRECOMPILED PYTORCH_NVCC
export PYTHONNOUSERSITE=1
python -m pip install --no-cache-dir -r artifact/env/runtime-freeze.txt
# Use the actual toolkit directory, not a symlink to another installation.
export CUDA_HOME=/usr/local/cuda-12.1
export TORCH_CUDA_ARCH_LIST=8.0
# Choose a NEW directory; retain it while this editable installation is in use.
python -m artifact.mil_artifact.package --output /tmp/mil-runtime-source-01
cd /tmp/mil-runtime-source-01/source
python -m artifact.mil_artifact.runtime_identity build
python -m artifact.mil_artifact.runtime_identity verify
bash artifact/scripts/doctor.sh --profile smoke
```

GPU 执行使用上述命令从打包后的 Mil 源码构建扩展。CPU 验证不覆盖 GPU 编译或推理。

显式构建入口执行本地可编辑安装，禁用依赖下载，拒绝旧二进制、旧构建目录及预编译扩展
绕过方式。只有构建成功后，才在 `artifact/results/runtime-build/build.json` 中记录精确的
源码哈希、编译器、CUDA 目标、Python／PyTorch ABI、实际模块来源和原生二进制哈希。
编译与导入检查隐藏 GPU 设备，不执行推理。后续 GPU 预检和实验继续使用这个新源码目录，
原工作树及已有编译文件不会被覆盖。
源码包内嵌 `artifact/source_snapshot.json`，预检使用其源码哈希，无需 `.git`。
解压后需保留该文件；构建输出和新结果不改变源码清单。

按[模型与数据](../docs/data_and_models.zh-CN.md)准备外部数据集和固定模型快照后，
先选择并检查任务，再显式执行，例如：

```bash
bash artifact/scripts/run.sh --profile smoke --suite router --method mil --dry-run
bash artifact/scripts/run.sh --profile smoke --suite router --method mil \
  --execute --allow-provisional
```

同一入口支持 `main`、`ablation`、`scalability`、`reference`、`full` 和 `paper`。
本发布版的 GPU 构建与推理验证尚未完成。`--allow-provisional` 表示接受测量尚未验证，
不会绕过源码／构建、资源、硬件或所选求解器预检。
方法、工作负载和参数筛选也作用于预检，因此仅选择 Mil 不需要 Saturn 授权。
运行器统一 Python 搜索路径指向当前源码，并在预检及实际任务中禁用用户 site-packages
和 CPU 替代导入。GPU 可见性根据所选硬件配置设置。

验证拒绝缺失记录、变更后的源码／二进制、外部导入路径及不匹配的运行环境。
GPU doctor 在启动推理入口或硬件／求解器探测之前执行此检查。
这是本地来源验证，不是签名证明；同时编辑记录和文件可绕过它。
来源检查通过不保证 CUDA 算子正确性或论文性能，实际 GPU 构建／推理仍标记为待验证。
仅使用 CPU 时无需运行 GPU 构建和验证命令。

模型权重、数据集及其访问条款是外部要求。Saturn 使用 PuLP 3.3.0／Gurobi 12.0.2，
需要运行者提供有效授权；适用时通过 `GRB_LICENSE_FILE` 选择许可证。
不要把授权密钥写入 Git 或日志，也不要替换为其他求解器。

可选容器配方为 `artifact/env/Dockerfile`，基础镜像摘要及构建参数固定在 `runtime.yaml`。
构建上下文应使用新打包的源码树。容器构建和 GPU 执行尚未验证。
保留 `.dockerignore` 对本地日志和备份的排除规则。
shell 入口支持显式将 `MIL_PYTHON` 设置为目标 Python 可执行文件。
