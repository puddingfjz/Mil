# Parallel utilities

[简体中文](README.zh-CN.md)

The files in this folder are ported from [Megatron-LM](https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core). We only keep the codes that are used in inference.

This directory contains parallel-state management, communication operations,
custom all-reduce and supporting utilities. See the
[runtime guide](../../../artifact/docs/vllm_runtime_guide.md) for their role in
Mil's execution and process-group lifecycle.
