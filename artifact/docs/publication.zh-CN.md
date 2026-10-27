# 源码分发

[English](publication.md)

`codex/release` 发布分支包含源码、运行输入、测试和实验说明，
不包含历史实验结果。

## 打包与验证

在文档指定的 Python 环境中，从仓库根目录执行：

```bash
python -m artifact.mil_artifact.package --output /tmp/mil-release-01
cd /tmp/mil-release-01/source
bash artifact/scripts/validate.sh
```

新输出目录包含 `source/`、`mil-source.tar.gz` 及记录文件/压缩包校验值的
`manifest.json`。`artifact/manifests/source_distribution.yaml` 是显式文件清单；
打包不递归复制工作区，也不访问备份目录。

源码包生成 `artifact/source_snapshot.json`，用于无 Git 环境下的预检和运行身份核对，
不包含自身哈希。源码变更或缺失会使验证失败。运行器将此记录复制到运行输入，
并在收集和验证阶段传递其哈希。

## 内容与资源获取

保留成本表、输出长度分布、数字化请求集、已知输出长度输入和模型配置。
不包含旧日志、历史结果索引、CPU 调度基线、性能表、绘图数值或结果图片。
单元测试中的合成样例是人工数据，不是报告的实测值；源码完整性签名保留。

私有数据、权重、凭据、Git 元数据、备份及生成结果不在源码包中。
原始数据、模型权重和新 profiling 测量需遵循各自条款获取，见
[模型/数据](data_and_models.zh-CN.md) 和 [资源生成](resource_generation.zh-CN.md)。
Saturn* 需要 Gurobi 授权，不会自动替换求解器。

发布分支从仅含源码与输入的根提交开始；其他分支或另行保留的 Git 对象不属于该发布版。
获取不带旧结果的工作副本时应明确选择 `codex/release`，仓库默认分支是独立设置。

## 验证范围

`cpu_preparation.yaml` 和 `release.yaml` 描述源码/输入及功能检查，
不认证历史性能。绘图必须显式提供用户自己的测量。
隔离验收应在全新 Conda 环境中使用独立源码包，禁用用户级包且不预装官方 vLLM。
见 [CPU 快速开始](cpu_quickstart.zh-CN.md)
和 [GPU 安装](../env/README.zh-CN.md#gpu-runtime)。
