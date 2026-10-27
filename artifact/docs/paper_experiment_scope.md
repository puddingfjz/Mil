<a id="artifact-实验范围"></a>

# Artifact experiment scope

[简体中文](paper_experiment_scope.zh-CN.md)

This scope defines experiments and parameters; configuration coverage does not
mean all performance experiments have been reproduced.

Paper: **Mil: Cost-guided Minimum Makespan Scheduling for Applications of Multiple
LLMs**, using the 14-page PDF with SHA-256
`7c554ef32ece500046e4999f898ca40904f3b3b7af489540e8cfd0f3c6a7c1d0`.
Page numbers below are PDF page positions. Paper configurations follow the
published settings; historical scripts/logs provide implementation and provenance context.

The catalog covers all paper experiment items and provides extensions such as
512-token Ensembling and eight-variant ablation. Standard and extended settings
are identified and aggregated separately.

<a id="标准范围与扩展"></a>

### Standard scope and extensions

- Figures 6–8 correspond to 28 unique workloads, matching the Round2 main script.
- `minRandom-differentSampleSize` with `SAMPLE_SIZE=10/12/14/16/18/20/22` is
  optional. Paper §5 samples at most N ready LLMs for MIN-R; N=8 here, matching
  standard `sample_size=8`.
- Earlier 5000-request noise scans, `noise_level=1e-6`, 1200 documents and
  15000/20000-request stress tests are outside the standard paper scope.
- With at least N models, MIN-R runs each point three times and reports mean
  and min/max. Paper configurations assign three new seeds. Historical
  `run_i=4/5/6` labels repetitions, not seeds or final samples. CPU acceptance
  checks configuration/statistics; actual MIN-R repetitions run separately.
- Single, batch, Oracle and paper tasks have no default outer wall-clock limit.
  Saturn's five-minute per-MILP limit is separate. Explicitly time-limited
  diagnostic runs are analyzed separately from paper-standard settings.

<a id="1-正式实验输入"></a>

## 1. Normal experiment inputs

The workload interface has three inputs:

| Input | Required specification |
| --- | --- |
| Model set | Models, revisions and tokenizers; weight locations come from environment configuration |
| Application type | Ensembling, Routing, Chain Summary or Mixed; dependency graph, prompt templates, routes, chunking and evaluation count |
| Request set | Dataset revision, selected samples/order, model assignments, replication count, maximum or specified output lengths |

Hardware, candidate plans, cost tables and output-length distributions are
supporting runtime/scheduler resources. Definition 1 (PDF page 3) uses
`(N, M, I, P, ρ, E)`: GPU count, models, initial lengths, execution plans,
processing functions and dependencies. Workload inputs and resources jointly
provide these fields.

Historical logs support result comparison. `scheduling_replay --source-log`
extracts lengths to isolate input differences and check initial schedules.
Normal scheduling constructs lengths/dependencies from the three inputs without
historical logs. CPU replay validates initial plans and predicted costs, not
actual inference time or runtime-adjustment speedups by itself.

<a id="2-共用实验设置"></a>

## 2. Shared settings

Source: §5, PDF pages 7–8.

- Host: 8 × A100 80GB, with the paper describing NVLink between each GPU pair;
  2 × 24-core CPUs, 1510 GB RAM, Ubuntu and vLLM.
- Nine main methods: Mil, MAX, MIN, MIN-R, Saturn*, LightRule, LightRule*,
  HybridFlow and HybridFlow*.
- Saturn* has a five-minute limit per MILP solve, not per experiment.
- Metrics: inference time, extra time for schedule generation, and their sum,
  end-to-end time. Figures 6–8 show both inference and end-to-end ratios to Mil.
- With at least N models (N GPUs), MIN-R runs each point three times, reports
  inference/end-to-end means and end-to-end min/max error bars. The paper does
  not impose one repetition count or median rule on every other method.

<a id="模型集合与请求来源"></a>

### Model sets and request sources

| Application | Model set | Requests / construction |
| --- | --- | --- |
| Ensembling | vicuna-13b-v1.5, oasst-sft-4-pythia-12b-epoch-3.5, alpaca-13b, baize-v2-13b, koala-13B-HF, dolly-v2-12b, mpt-7b-chat, chatglm3-6b, stablelm-tuned-alpha-7b | MixInstruct; nine independent models process the same requests |
| Routing | Llama-2-70b-chat-hf, Mixtral-8x7B-Instruct-v0.1, WizardLM-13B-V1.2, CodeLlama-34b-Instruct-hf, Mistral-7B-Instruct-v0.2 | ROUTERBENCH; each request is assigned to one model |
| Chain Summary | vicuna-13b-v1.5 for summarization, Llama-2-70b-chat-hf for evaluation | BOOOOKSCORE and BookSum; 2048-token chunks, sequential summary followed by evaluation |
| Mixed | Chain Summary plus Ensembling | Both applications scheduled together with separate request counts |

Table 1 Routing counts are 408, 1267, 2068, 456 and 2657 for the five models,
totaling 6856. Replication r gives 6856 × r requests. Legacy CLI `reqnum=10000`
does not imply 10000 actual requests: `_get_router_bench_data` uses prepared
per-model lists and the replication count.

<a id="3-主结果figures-68"></a>

## 3. Main results: Figures 6–8

Source: §5.1, settings on PDF page 8 and figures on page 9. Each row is an
independent parameter scan, not a Cartesian product of scan dimensions.

| Item | Experiment | Variable | Fixed settings |
| --- | --- | --- | --- |
| Figure 6(a) | Ensembling request scale | 1000 / 2000 / 5000 / 10000 requests | 9 models, max output 256 |
| Figure 6(b) | Routing, unknown output lengths | 1 / 2 / 3 / 4 copies | 5 models, max output 4096 |
| Figure 6(c) | Routing, known output lengths | 1 / 2 / 3 / 4 copies | Force exact dataset-response token counts |
| Figure 6(d) | Chain Summary document scale | 100 / 200 / 300 / 400 / 500 documents | 4 evaluations, max output 900 |
| Figure 6(e) | Chain Summary evaluations | 2 / 4 / 8 evaluations | 300 documents, max output 900 |
| Figure 6(f) | Chain Summary output lengths | max output 100 / 500 / 900 | 400 documents, 4 evaluations |
| Figure 7 | Mixed workload composition | 100 / 200 / 300 / 400 / 500 summary documents | 5000 Ensembling requests; 4 evaluations; summary/Ensembling max output 900/256 |
| Figure 8 | Mixed application scalability | 2 / 4 / 8 applications | Base: 200 summary documents + 5000 Ensembling requests, replicated 1 / 2 / 4 times; otherwise Figure 7 settings |

Figure 8's x-axis is application count, with eight GPUs fixed.
`model_replicate_num=1/2/4` means 2/4/8 applications, not models. At larger scales,
some baselines fail to finish hours of search, fail MILP or cannot fit the models
simultaneously. Cross markers represent failures, not zero-second results or omitted points.

There are 31 x-axis coordinates. Removing two overlaps among Chain Summary
scans and the Figure 8 two-application overlap with Figure 7 leaves 28 unique
workloads. Nine methods yield 252 nominal configurations, excluding repetitions,
component experiments and failed tasks; this is not the paper's actual run count.

<a id="figure-9调度执行时间线"></a>

### Figure 9: observed scheduling timelines

Source: PDF page 10. Routing with unknown lengths and one copy shows GPU usage
and model loading/execution/preemption/adjustment. Eight panels cover nine methods,
with MIN and MIN-R sharing a panel. These are another output of the same main
experiment, not a new request set. Reproduction requires observed events and
GPU assignments; predicted initial plans do not provide actual execution times.

<a id="512搜索效率"></a>

### §5.1.2: search efficiency

Source: PDF page 11, a text analysis rather than a numbered figure. Extract
search time, initialization/estimation overhead, enumerated candidates and
cost-estimation counts from the same experiments. Analyze scalability overhead separately. CPU resources affect wall time, so replay on fewer CPUs
need not exactly match the paper host's seconds.

<a id="4-组件分析tables-24-及正文"></a>

## 4. Component analysis: Tables 2–4 and text

Source: §5.2, PDF pages 11–12.

| Item | Comparison | Settings | Outputs |
| --- | --- | --- | --- |
| Table 2 | Mil with/without preemption | Ensembling 10000; Routing unknown lengths, 4 copies; Mixed (500, 5000), 2 apps; otherwise main settings | inference + extra, end-to-end ratio to preemption-enabled control |
| Table 3 | Output-length sampling distribution error | Ensembling 1000 / 2000, max output 4096; Exact, 0, 1e-5, 1e-4, 1e-3, 1e-2, Rand | Inference ratio to normal Mil; noise settings and seeds |
| §5.2 estimation error | Predicted versus measured inference | Four §5.1 applications; known-length Routing separately | Mean/max relative errors per application and per-point predictions/measurements |
| §5.2 runtime adjustment | Mil with/without runtime adjustment | All §5.1 experiments, including Mixed and application scalability | Per-point inference/end-to-end times and speedups |
| Table 4 | Greedy Mil versus Oracle | Three-model Ensembling: baize-v2-13b, koala-13B-HF, dolly-v2-12b; 1000 / 2000 requests, max output 256 | Inference time, estimation error, search time and estimation/enumeration counts |

Table 2 omits Chain Summary because the paper reports identical schedules with
and without preemption. The historical eight-variant Mil/MIN `draw_ablation.py`
chart is a different experiment and does not replace Table 2.

Table 3 adds uniform `[-a, a]` noise to output-length probabilities. Rand samples
uniformly over valid lengths; Exact uses exact lengths. These seven columns
are not the `randomCost`/no-preemption variants. Table 3 uses a 4096-token limit,
not Figure 6(a)'s 256.

Define denominators, absolute-value treatment and aggregate samples for estimation errors.

Table 4 Oracle enumerates schedules using processing functions and known
output lengths; it does not execute every plan on GPUs. The paper reports OOM
above three models, so the default nine-model Ensembling set is unsuitable.


<a id="5-processing-function-的支撑实验figures-34"></a>

## 5. Processing-function supporting experiments: Figures 3–4

Source: §3, PDF pages 4–5. Configuration/analysis entries cover the experiments
below. Acquire the required inputs explicitly; no supporting experiment results
are bundled. See [coverage limits](paper_coverage.md).

| Item | Objective | Data and models |
| --- | --- | --- |
| Figures 3(a), 3(b) | Output-length eCDF stability across input-length groups and request categories | vicuna-13b-v1.5; No Robots, 10000 requests, 10 categories |
| Figure 3(c) and JSD text | Cross-dataset output-length distributions | Eight datasets; figure uses vicuna-13b-v1.5, text also tests chatglm3-6b and WizardLM-13B-V1.2 |
| Figure 3(d) | Per-generation-iteration compute latency versus FLOPs, by batch size | Llama 7B, one GPU, vLLM profiling |
| §3 input preparation / output sampling | Latency fits for the other two iteration components | Same profiling procedure, described in text without a separate figure |
| Figures 4(a), 4(b) | Running requests per iteration in actual versus simulated inference | vicuna-13b-v1.5, ShareGPT, 1000 requests |
| §3 complete cost example | Predicted/measured time with and without model loading | Same as Figure 4; record inference and loading-inclusive time separately |

Figure 3(c) uses No Robots, dolly-15k, Natural Questions, TLDR, GSM8K, HumanEval,
mbpp and ShareGPT. The paper states 10000 samples per dataset; replication/
sampling for smaller datasets must be established from experiment code, not
assumed.

Table 1 contains input statistics. Figures 1, 2 and 5 are problem, dependency
or architecture illustrations, not independent performance tasks.

<a id="6-配置与资源入口"></a>

## 6. Configuration and resource entries

`configs/paper_experiments.yaml` maps these items to 72 settings and 323 expanded
tasks, with methods specified in `configs/methods.yaml`. The 512-token
Ensembling and other extensions remain separate from paper-standard settings.

Acquisition uses `benchmarks/construct_cost_model.py`, `benchmark_throughput.py`
and `comp_model_size.py`, requiring real models and GPUs; default CPU acceptance
does not start them. `artifact/mil_artifact/resource_build.py` uses explicit
configuration/existing logs to build tables and distributions into a new directory,
not source files. Existing numeric resources are included; the sampler is
`benchmarks/output_length_sampler.py`. See [resource generation](resource_generation.md).

Configuration coverage, CPU functionality, historical candidates and measured
GPU performance are distinct; see [paper mapping](paper_mapping.md) and
[coverage limits](paper_coverage.md).
