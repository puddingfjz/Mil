<a id="论文到-artifact-的映射"></a>

# Paper-to-artifact mapping

[简体中文](paper_mapping.zh-CN.md)

See [experiment scope](paper_experiment_scope.md) for the paper requirements.
`configs/paper_experiments.yaml` is the machine-readable catalog; `paper_suite`
checks 20 items and expands 72 unique settings into 323 scheduling tasks.
Repeated chart coordinates reuse tasks. Full-combination and 512-token settings
are provided as extensions.

| Paper item | Stable ID / entry |
| --- | --- |
| Figure 6(a), Ensembling 1k/2k/5k/10k, 256 tokens | `figure_6a` / `paper_results` |
| Figure 6(b/c), Routing unknown/known output lengths, 1–4 copies | `figure_6b`, `figure_6c` / `paper_results` |
| Figure 6(d–f), independent summary document-count, evaluation-count and output-length scans | `figure_6d`, `figure_6e`, `figure_6f` / `paper_results` |
| Figure 7, Mixed | `figure_7` / `paper_results` |
| Figure 8, 2/4/8 applications with 8 logical GPUs | `figure_8` / `paper_results` |
| Figure 9, observed Routing timeline with unknown output lengths and one copy | `figure_9` / `paper_results --timelines` |
| Table 2, preemption | `table_2` / Mil comparisons for three workloads |
| Table 3, Exact / 0 / 1e-5 / 1e-4 / 1e-3 / 1e-2 / Rand | `table_3` / 1k/2k requests, 4096 tokens |
| Table 4, three-model Oracle | `table_4` / Baize, Koala, Dolly; 1k/2k, 256 tokens |
| §5.2 runtime adjustment | `runtime_adjustment` / all §5.1 settings with adjustment-disabled controls |
| §5.1.2 search efficiency, §5.2 prediction error | `search_efficiency`, `estimation_error` / `analyses.json` |
| Figure 3(a/b), input-length groups and request categories | `figure_3ab` / `supporting_analyses` |
| Figure 3(c), distributions/JSD for three models and eight datasets | `figure_3c_jsd` / `supporting_analyses` |
| Figure 3(d), FLOPs/iteration latency, preparation and sampling costs | `figure_3d` / `supporting_analyses` |
| Figure 4, actual and simulated traces | `figure_4` / `supporting_analyses` |
| Table 1, Routing input counts | `table_1` / statistics from normal request sets |

Figures 1, 2 and 5 are illustrations, not additional performance tasks.
Complete configurations do not imply completed experiments. No experimental results
are bundled; see [validation scope](paper_coverage.md).

<a id="方法对应"></a>

## Method mappings

`configs/methods.yaml` records dispatchers, switches and evidence. Main comparisons
include Mil, MAX, MIN, MIN-R, Saturn*, LightRule, LightRule*, HybridFlow and HybridFlow*.

Mil for Ensembling, Routing and Mixed uses `greedy_saturn_LPT`. Final Chain
Summary plotting sources select `greedy_saturn`, implemented by
`get_candidate_plan_groups_greedy_best_exec_plan_first`. The LPT variant calls
`get_candidate_plan_groups_greedy_best_exec_plan_first_LPT` and considers models
by topology level and longest completion time under minimum-GPU plans.
These are registered as `mil` and `mil_chain_summary`, with separate statistics.
Optional CPU regression comparisons require explicit external references; no historical result baseline is bundled.

Historical commit 9011731 distinguishes old/new HybridFlow implementations.
`hybridflow_model_parallel_old` retains the older variant; the newer one loads
models in a colocated set individually and uses CLI name `hybridflow_model_parallel`.
Logs lack commits, so Round2 labels are only generational clues, not proof of
paper membership. Results remain separate.

LightRule uses expected output lengths (`noise_level=-2`); LightRule* uses the
original cost model. Missing initial-plan costs remain null, not zero.
Saturn* uses PuLP/Gurobi with its 300-second solve limit and no automatic CBC
fallback. Users provide valid Gurobi authorization; default CPU acceptance does not start it.

<a id="对照与统计协议"></a>

## Controls and statistical protocol

- Table 2 disables preemption with `greedy_saturn_LPT_no_preemption`. The old
  `randomCost` also disables preemption and is not Table 3 Rand.
- Table 3 Exact uses known output lengths. Rand samples uniformly over the legal
  interval on unknown-length paths; other noise settings perturb output-length probabilities.
- Table 4 selects builder subset `[3:6]`. Oracle has no outer timeout by default;
  explicit diagnostic timeouts and their status are recorded.
- The paper requires three MIN-R runs for at least eight models, reporting means
  and min/max. New configurations specify seeds 0/1/2. Record actual seeds, repetition labels and sample counts.
- Other methods use explicit repetition settings. Aggregate only matching parameters/methods; do not select a post-hoc minimum.
- `search_time_seconds` includes preparation; `total = inference + search`.
  See [timing conventions](timing_convention.md).

<a id="输入模拟与实测的边界"></a>

## Inputs, simulation and measurements

Normal inputs are model set, application and request set. `cpu_schedule prepare`
calls original builders on raw data; `cpu_schedule run` reads independent numeric
request sets and invokes the real scheduler/cost simulator. Log replay is a
separate diagnostic entry, `scheduling_replay`.

CPU output describes initial model/GPU assignment, stage order and predicted
cost. Original functions plus synthetic completion events check runtime logic;
actual speedups require real measurements. Figure 9 uses observed stage
events only. Missing paired actual/simulated Figure 4 traces are reported rather
than replaced with predicted plans.

The distribution includes sources, running inputs, tests and commands, not logs or results.
Resource manifests define acquisition and licensing. BooookScore texts, weights and license secrets are excluded.
