# Mil implementation

[简体中文](README.zh-CN.md)

`search_exec_plans.py` searches execution plans, `fake_scheduling.py` predicts
their costs, and `schedule_multi_model.py` coordinates placement and runtime
adjustment. `my_llm_infer_worker_multiprocessing.py` hosts inference workers.
Registered method selectors include compatibility variants with `old` in their
names. See `artifact/configs/methods.yaml` for their dispatch mappings.

Use the [artifact guide](../artifact/README.md) for parameterized experiments,
CPU validation, resource generation and plotting. Do not import GPU runtime
modules as a way to perform a CPU-only check.
The search, simulation and Saturn solver modules are libraries; invoke the
artifact entry points to run configured experiments.

`construct_cost_model.py`, `benchmark_throughput.py` and `comp_model_size.py`
provide GPU measurement primitives. They require an explicitly prepared model
and GPU environment. The artifact resource planner generates collection commands
without executing them. Numeric cost/distribution modules are fixed inputs,
not proof that their original measurements were rerun.
