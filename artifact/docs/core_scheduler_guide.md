<a id="核心调度源码阅读指南"></a>

# Core scheduler source guide

[简体中文](core_scheduler_guide.zh-CN.md)

The three core modules handle plan search, cost simulation and runtime coordination.
Candidate search, initial GPU placement and runtime plan selection use distinct
ordering rules, described below.

For weight transfers, KV cache and worker lifecycle, see the
[modified vLLM runtime guide](vllm_runtime_guide.md).

<a id="建议阅读顺序"></a>

## Suggested reading order

| Module | Entry points | Responsibility |
| --- | --- | --- |
| `search_exec_plans.py` | `get_best_model_schedule`, `_get_best_model_schedule`, `MyModelSystem.get_candidate_plan_groups_dispatch` | Select cost resources, construct model state, generate method-specific candidates and search stage sequences |
| `fake_scheduling.py` | `fake_FCFS_schedule`, `fake_FCFS_schedule_vertical_fuse`, `estimate_cost_from_predicted_logs` | Simulate progress using request lengths, arrivals and cache limits; estimate iteration latency and completion time from cost tables |
| `schedule_multi_model.py` | `search_best_scheduling`, `prepare_exec_plan_states`, `initialize_SHARED_CONTECT`, `get_the_next_round_exec_plan_schedule` | Convert search results into runtime state, assign GPU IDs, coordinate model processes and stage transitions |

`schedule_multi_model.py` also constructs application request graphs and length
callbacks. `get_schedule_setting` distinguishes dataset and synthetic inputs.
`search_best_scheduling` builds callbacks, invokes search and converts results to
`MyExecPlanState`. The full entry `main_with_preemption` then organizes model processes.

These modules depend on Ray, vLLM and related components. Use
`artifact/scripts/validate.sh` for GPU-free functional checks; CPU search commands
are documented in the [CPU guide](cpu_quickstart.md).

<a id="搜索模块中的数据对象"></a>

## Search data objects

- `MyModelInfor` and `MyFusedModelInfor`: models, dependencies and request progress.
- `MyExecPlan` and `MyVerticalFusedExecPlan`: resources, predicted cost and stage progress of one plan.
- `MyExecPlanGroup`: plans within a stage and their dependency arrivals.
- `MyExecPlanGroupSeq`: stage-group sequence, durations and saved model state.
- `MyModelSystem`: system state and method-specific candidate generators.

Methods dispatch through `get_candidate_plan_groups_dispatch`.
`hybridflow_model_parallel_old` maps to `get_candidate_plan_groups_HybridFlow_old`.
Paper-method mappings are defined in `artifact/configs/methods.yaml` and dispatch tests.

`get_one_stage_exec_plans_sorted` selects a method-specific ordering path, with
early returns for some methods. `_get_best_model_schedule` handles cumulative
predicted cost, state deduplication/pruning and stage expansion. Internal search
count rules are separate from outer wall-clock limits; CPU scheduling has no
outer time limit by default.

<a id="成本预测不是实测运行时间"></a>

## Predicted cost versus measured runtime

`fake_FCFS_schedule` selects simulation paths based on arrivals, builds
prefill/decode iteration records and uses cost tables to estimate cumulative
latency, request completion and throughput. Vertical fusion has a separate
simulation entry. Request IDs, length arrays and progress ranges correspond
by index; updates and sorting must preserve that correspondence.

The fast path in `get_info_at_stop_time` uses the supplied `stop_iter_i` to
compute the completed range. `update_fake_FCFS_schedule_metadata` updates
progress after a stage. These times are model predictions, not GPU measurements.

<a id="三种排序和放置场景"></a>

## Three ordering and placement contexts

| Context | Entry | Behavior |
| --- | --- | --- |
| Search candidates | `MyModelSystem` methods and `get_one_stage_exec_plans_sorted` | Each method defines candidate construction, costing and ordering; not uniformly ordered by GPU count |
| Initial GPU placement | `initialize_SHARED_CONTECT` | Sort first-stage plans descending by a tuple of `num_worker`, whether total GPUs are a multiple of the interconnect-unit size, and `dp_size`; then assign GPU IDs |
| Runtime plans awaiting launch | `_try_to_load_exec_plans` and two root selectors | Respect candidate dependencies; usually prioritize dependency roots with larger `num_worker * dp_size`, while plain LightRule preserves input root order |

A runtime dependency root is defined relative to the current candidate list:
its upstream model IDs do not intersect candidate model IDs. The selector
chooses a launchable plan from these roots. `LightRule_our_cost_model` uses
GPU priority; plain LightRule preserves input order.

`_get_the_first_plan_state_when_sorted_by_topology` returns the first root in
input order. The GPU-priority selector chooses roots by descending GPU count.
Both behaviors have CPU functional tests.

`no_runtime_adjustment` uses stage-barrier advancement and decides when to
consider the next stage from still-running plans. It calls `_try_to_load_exec_plans`
to allocate GPUs; `get_the_next_round_exec_plan_schedule` handles interconnect-aware
placement adjustments.
