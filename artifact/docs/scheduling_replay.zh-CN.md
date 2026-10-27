<a id="cpu-scheduling-comparison"></a>

# CPU 调度比较

[English](scheduling_replay.md)

常规验证使用[正常输入 CPU 调度](cpu_quickstart.zh-CN.md)，输入为模型集合、应用图和数字化请求集，
检查真实调度输出。可选对比需显式提供外部基线，发布包不附带基线。

```bash
python -m artifact.mil_artifact.compare_schedules \
  --requests artifact/inputs/frozen/ensemble.json \
  --result /path/to/cpu-output/result.json \
  --reference /path/to/your/cpu-reference.json \
  --output /tmp/mil-comparison-01
```

该比较不启动搜索，核对方法、输入、资源、参数、有序计划、可用阶段成本和预测总成本。
LightRule 未提供的初始成本保持 null，不要求不同机器的搜索墙钟时间相等。

诊断回放某份原始日志时，可使用 `artifact.mil_artifact.scheduling_replay`，重建独立
LLM-Blender 图和日志输入长度，再启动实际 CPU 搜索。它需要该日志及文档指定的原生依赖，
不加载权重或启动 GPU 推理，与正常输入验证分开。

比较多份原记录时，`compare_schedules --sources` 使用数字化来源索引和本地日志，按全文哈希、
参数、模型顺序及 CPU 阶段完整性筛选，比较全部合格候选。后续失败不否定已完成的 CPU 阶段；
结果相同或修改时间更新均不能证明源码提交或最终论文样本归属。
缺失原始日志会明确报告，不用合成测量替代。
