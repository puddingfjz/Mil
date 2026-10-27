<a id="mil-paper-artifact"></a>

# Mil 论文复现包

[English](README.md)

Mil 用于调度由多个 LLM 组成的应用。本仓库提供调度与运行时实现、实验配置、
数字化输入、资源生成工具和绘图工具。

<a id="paper-and-poster"></a>

## 论文与海报

**Mil: Cost-guided Minimum Makespan Scheduling for Applications of Multiple LLMs**

[论文](https://github.com/puddingfjz/Mil/blob/main/technical_report/technical_report.pdf)
· [DOI](https://doi.org/10.14778/3819518.3819522)
· [海报 PDF](https://github.com/puddingfjz/Mil/blob/main/assets/poster/Mil_VLDB_A1_Poster.pdf)

论文与海报托管在 `main` 分支，不随仅含源码与运行输入的发布包分发。
下方提供[海报预览](#poster)。

<a id="getting-started"></a>

## 入门

- [复现指南](artifact/README.zh-CN.md)：实验、验证与绘图。
- [CPU 快速开始](artifact/docs/cpu_quickstart.zh-CN.md)：无需 GPU 的正确性与可用性检查。

发布版包含 CPU 和 GPU 执行路径。默认验证与绘图无需 GPU；GPU 构建、推理和性能验证
不属于所附 CPU 检查的覆盖范围。

<a id="code-structure"></a>

## 代码结构

论文复现包同时包含核心实现和实验工具。`artifact/` 提供复现接口；
调度算法和定制推理运行时位于该目录之外。

```text
benchmarks/              计划搜索、成本模拟、运行协调与性能采集
vllm/                    Mil 定制推理运行时
csrc/                    原生 C++/CUDA 扩展
artifact/
  README.md              实验与复现指南
  mil_artifact/          实验组织、验证、资源构建与绘图
  scripts/               命令行入口脚本
  configs/               实验、方法与硬件配置
  inputs/                固定请求集与模型配置
  expected/              源码完整性签名（不含实验结果）
  manifests/             资源目录与源码分发清单
  env/                   环境定义与安装说明
  tests/                 正确性与可用性测试
  docs/                  实验与实现文档
```

阅读源码时，可先看[核心调度指南](artifact/docs/core_scheduler_guide.zh-CN.md)，了解计划搜索、
成本预测和运行时协调；再看[定制 vLLM 运行路径指南](artifact/docs/vllm_runtime_guide.zh-CN.md)，
了解请求执行、权重卸载、KV 缓存管理与 worker 生命周期。

<a id="modified-runtime"></a>

## 定制运行时

**Mil 需要使用本仓库中的定制 vLLM 源码。** CPU 调度使用本地 Python 源码；
GPU 执行还需要对应的原生扩展。安装步骤见 [Mil 环境说明](artifact/env/README.zh-CN.md)。

<a id="poster"></a>

## 海报

<p align="center">
  <a href="https://github.com/puddingfjz/Mil/blob/main/assets/poster/Mil_VLDB_A1_Poster.pdf">
    <img
      src="https://raw.githubusercontent.com/puddingfjz/Mil/main/assets/poster/Mil_VLDB_A1_Poster.png"
      alt="Mil VLDB 海报：面向多 LLM 应用的成本引导最小完工时间调度"
      width="900"
    >
  </a>
</p>

点击海报可打开完整分辨率的 PDF。

<a id="upstream-attribution-and-license"></a>

## 上游归属与许可证

Mil 基于 [vLLM](https://github.com/vllm-project/vllm) 构建，保留上游版权及许可证声明。
仓库许可证条款见 [LICENSE](LICENSE)。
