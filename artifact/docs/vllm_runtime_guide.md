<a id="vllm-运行路径"></a>

# vLLM runtime paths

[简体中文](vllm_runtime_guide.zh-CN.md)

`benchmarks/schedule_multi_model.py` produces multi-model plans and passes them
to vLLM. These GPU execution paths require build and inference validation
separate from CPU scheduling tests.

<a id="请求与执行"></a>

## Requests and execution

`LLM._run_engine` in `vllm/entrypoints/llm.py` receives subsequent inputs from
the communicator and returns outputs periodically or on completion. An empty
running queue may still be waiting for upstream model outputs.
`vllm/core/multimodel_scheduler.py` manages cross-model requests and shared state.

`LLMEngine.step` calls `Scheduler.schedule`. Workers process cache operations,
then call `ModelRunner.execute_model` to prepare inputs, execute models and sample.
The engine updates request state. This consumes the selected plan rather than
searching the entire multi-model schedule again.

<a id="初始化和重新调度"></a>

## Initialization and rescheduling

`USE_VLLM == 'True'` or `SOFT_RESCHEDULE == 'False'` selects normal initialization.
Otherwise an actor pool is reused or created. Actor reuse does not imply reuse
of the Worker, model tensors or KV cache: internal workers, model loading and
schedulers still undergo reinitialization.

`_init_cache_no_profile` uses cache-capacity data for fixed model/parallelism
settings and scales it by the GPU memory ratio; it is not a general measurement
for arbitrary models or hardware. `no_profile` and `fake_profile` still involve
CUDA memory queries, cache initialization or model warmup, and are not CPU-only entries.

<a id="权重和-kv-缓存"></a>

## Weights and KV cache

Normal Llama mode uses `load_weights_ori`; offloading mode uses
`preprocess_for_offloading_weights_fast`. Cache GPU selection combines
`TOT_ORDERED_GPUS` and `CUDA_VISIBLE_DEVICES`. Weight residency depends on the
actual interval, degree and cache-block allocation; an environment-variable
value is not directly the number of resident layers.

`weight_cache_cpu` can contain views of GPU-allocated tensors. `forward_ours`
applies residency changes, waits for loading events, computes, records events
and starts subsequent transfers. Address updates, CUDA stream waits and event
ordering in `load_layer_params` are interdependent. Native code in
`csrc/cache_kernels.cu` performs asynchronous device-to-device copies on the current stream.

Increasing resident layers dynamically changes logical block tables, available
KV regions, reference counts, tensor views and physical movement together.
See `core/scheduler.py`, `core/block_manager.py` and `worker/cache_engine.py`.
Actor, cache and process-group cleanup is implemented in the engine, workers
and `model_executor/parallel_utils/`.
