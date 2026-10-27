# Mil paper artifact

[简体中文](README.zh-CN.md)

Mil schedules applications composed of multiple LLMs. This repository contains
its scheduling and runtime implementation, experiment configurations, numeric
inputs, resource-generation tools and plotting tools.

## Paper and poster

**Mil: Cost-guided Minimum Makespan Scheduling for Applications of Multiple LLMs**

[Paper](https://github.com/puddingfjz/Mil/blob/main/technical_report/technical_report.pdf)
· [DOI](https://doi.org/10.14778/3819518.3819522)
· [Poster PDF](https://github.com/puddingfjz/Mil/blob/main/assets/poster/Mil_VLDB_A1_Poster.pdf)

Publication materials are hosted on the `main` branch and are not bundled with
the source-only release. A [poster preview](#poster) is available below.

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

## Poster

<p align="center">
  <a href="https://github.com/puddingfjz/Mil/blob/main/assets/poster/Mil_VLDB_A1_Poster.pdf">
    <img
      src="https://raw.githubusercontent.com/puddingfjz/Mil/main/assets/poster/Mil_VLDB_A1_Poster.png"
      alt="Mil VLDB poster: Cost-guided Minimum Makespan Scheduling for Applications of Multiple LLMs"
      width="900"
    >
  </a>
</p>

Click the poster to open the full-resolution PDF.

## Upstream attribution and license

Mil builds on [vLLM](https://github.com/vllm-project/vllm). Upstream copyright
and license notices are retained. See [LICENSE](LICENSE) for the repository
license terms.
