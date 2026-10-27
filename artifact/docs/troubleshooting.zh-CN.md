<a id="troubleshooting"></a>

# 故障排查

[English](troubleshooting.md)

<a id="doctor-reports-unresolved-paths"></a>

## `doctor` 报告路径无法解析

设置 `MIL_DATA_ROOT`，必要时设置 `MIL_MODEL_ROOT`。
不要将机器本地路径写入 Python 源码或提交到版本控制。

<a id="gpu-name-or-count-does-not-match"></a>

## GPU 型号或数量不匹配

使用与主机对应的硬件配置；默认遇到不匹配即停止。
`--allow-profile-mismatch` 仅将 GPU 数量、型号、显存不匹配降为警告，并将运行标为不可比较。
NVIDIA 驱动缺失或故障仍是错误，不能用该标志绕过。

若 `nvidia-smi` 在受限环境中失败，应检查设备访问权限，并与主机上的输出对比。
该环境中的设备可见性问题本身不证明主机驱动故障。

<a id="a-gpu-is-locked"></a>

## GPU 被锁定

使用同一结果根目录的另一个 artifact 运行占有了所请求的设备。
可通过 `--gpu-lock-timeout SECONDS` 等待，或按集群协调规则选择不重叠的硬件配置／结果根目录。
运行器退出时由内核释放锁。不要删除其他活动运行的锁，也不要终止无关 GPU 进程。

<a id="doctor-reports-a-disk-space-error"></a>

## `doctor` 报告磁盘空间不足

检查器按硬件配置中的暂定存储策略核对文件系统容量和空闲字节数。
清单缺失资源计入仓库、数据或模型目录需求，额外预留 5% 下载／暂存开销。
结果空间按展开任务数和每任务 400 MB 估计，依据为 1,762 份历史日志及 2.05 安全系数。
可将 `MIL_RESULTS_ROOT` 放在更大的文件系统或先准备已验证资源。
存储容量应适配所选任务矩阵；该结果估计是暂定值，大规模运行需监测实际用量。

<a id="doctor-reports-a-gurobi-license-failure"></a>

## `doctor` 报告 Gurobi 授权失败

包含 Saturn* 的配置需要固定版本的 `gurobipy`／PuLP 及有效 Gurobi 授权。
按机构的 Gurobi 方案配置许可证，再对目标 profile 运行 doctor。
不要切换到 CBC：历史调用器显式使用 Gurobi，替换会改变基线含义。
不含 Saturn* 的配置不执行该授权检查。

本地安装需要时，可用 `GRB_LICENSE_FILE` 选择有效许可证。
凭据不要进入日志或追踪文件；仅 Mil 的 CPU 调度不需要 Gurobi。

<a id="docker-is-unavailable"></a>

## Docker 不可用

使用 `artifact/env/README.md` 的原生 Conda 流程。Docker 是可选项，原生安装和验证
命令无需 Docker 守护进程。

<a id="a-method-is-blocked"></a>

## 方法被阻止

方法映射不完整或 dispatcher 缺少对应方法。不要在同一论文标签下替换其他方法；
应先更新 `paper_mapping.md`、实现或恢复精确变体，并添加回归用例。

<a id="a-run-was-interrupted"></a>

## 运行中断

使用 `--resume <run-directory>`。成功任务目录保留，失败任务产生新尝试记录并保留旧日志。
若 Git 提交、artifact 输入摘要、展开任务矩阵、profile、Python 可执行文件或硬件不匹配策略
不同，恢复会拒绝执行。每个任务位于独立进程组，SIGINT／SIGTERM 清理只影响该任务。
不可变输入变化时应启动新运行，避免混用来源。

<a id="a-task-exceeds-its-wall-clock-timeout"></a>

## 任务超过墙钟时间限制

运行器记录 `timed_out: true` 和退出码 124，仅终止当前任务进程组。
修复原因，或用更大的正数 `--task-timeout-seconds` 开始新运行；恢复禁止修改限时，
以免同一运行混合失败策略。profile 默认 `task_timeout_seconds: null`，
只有显式请求时才会自动超时。

<a id="a-zero-exit-task-is-reported-invalid"></a>

## 退出码为零的任务被标为无效

历史进程可能在打印内部失败后仍返回零。收集器因此要求执行、搜索、准备和总计时字段，
并拒绝 traceback、断言和 CUDA 显存不足诊断。
检查不可变尝试日志，修复原因，再用 `--resume <run-directory>` 重新解析和尝试该任务。
不要编辑日志或以零替换缺失值。

<a id="repeated-runs-differ-despite-the-same-seed"></a>

## 相同种子的重复运行结果不同

检查 `metadata.json` 的 `seed_provenance` 中 `experiment_seed`、`workload_ratio_seed`
和 `legacy_test_id`，三者用途不同。统一实验种子控制进程随机数和采样，未强制启用可能改变
性能含义的 GPU 确定性算法，因此不保证 GPU 结果逐位一致。
记录计时离散程度，并按 `configs/figures.yaml` 指定方式聚合。

<a id="legacy-files-appear-to-be-changing"></a>

## 原有文件似乎被改动

停止运行。artifact 应在暂存工作目录中执行。检查 `metadata.json` 的 `working_directory`，
它必须位于所选结果根目录内，而非 `benchmarks/`。
