# Mil paper artifact

[简体中文](README.zh-CN.md)

Mil schedules applications composed of multiple LLMs. This repository contains
its scheduling and runtime implementation, experiment configurations, numeric
inputs, resource-generation tools and plotting tools.

## Getting started

- [Mil artifact guide](artifact/README.md): experiments, validation and plotting.
- [CPU quickstart](artifact/docs/cpu_quickstart.md): GPU-free correctness and usability checks.

The release includes CPU and GPU execution paths. Default validation and
plotting do not require GPUs. GPU build, inference and performance validation
are outside the scope of the bundled CPU checks.

## Code structure

The paper artifact includes both the core implementation and the experiment
tools. The `artifact/` directory provides the reproducibility interface; the
scheduling algorithms and modified inference runtime live outside it.

```text
benchmarks/              Plan search, cost simulation, runtime coordination and profiling
vllm/                    Mil-modified vLLM inference runtime
csrc/                    Native C++/CUDA extensions
artifact/
  README.md              Experiment and reproduction guide
  mil_artifact/          Experiment orchestration, validation, resource building and plotting
  scripts/               Command-line entry scripts
  configs/               Experiment, method and hardware configurations
  inputs/                Frozen request sets and model configurations
  expected/              Source-integrity signatures (no experiment results)
  manifests/             Resource catalogs and source distribution manifest
  env/                   Environment specifications and installation instructions
  tests/                 Correctness and usability tests
  docs/                  Experiment and implementation documentation
```

For a guided source walkthrough, start with the
[core scheduler guide](artifact/docs/core_scheduler_guide.md), covering plan
search, cost prediction and runtime coordination. Then read the
[modified vLLM runtime guide](artifact/docs/vllm_runtime_guide.md) for request
execution, weight offloading, KV cache management and worker lifecycle.

## Modified runtime

**Mil requires the modified vLLM sources in this repository.** CPU scheduling
uses the local Python sources; GPU execution also requires the corresponding
native extensions. Follow the [Mil environment instructions](artifact/env/README.md).

## Upstream attribution and license

Mil builds on [vLLM](https://github.com/vllm-project/vllm). Upstream copyright
and license notices are retained. See [LICENSE](LICENSE) for the repository
license terms.
