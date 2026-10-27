# Add a model

[简体中文](add_a_new_model.zh-CN.md)

Obtain the model weights and tokenizer under their own access and license terms.
Add an explicit revision and model configuration to the artifact model manifest;
do not include credentials or machine-specific cache paths in source control.

Collect model size, FLOPs coefficients and initialization cost using the functions
in `comp_model_size.py` on the intended GPU hardware. Collect per-iteration latency
with `construct_cost_model.py` and output lengths with `benchmark_throughput.py`.
These are GPU measurements and are not executed by CPU validation.

Declare the resulting logs and model metadata in a resource configuration, then
use the [parameterized builder](../artifact/docs/resource_generation.md) to produce
new standalone cost and distribution modules. Update initialization metadata
before rebuilding latency resources. The builder does not append to source files.
Select new resources explicitly and run CPU checks before starting GPU experiments.
