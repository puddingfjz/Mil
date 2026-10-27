# CPU scheduling comparison

[简体中文](scheduling_replay.zh-CN.md)

Use [normal-input CPU scheduling](cpu_quickstart.md) for regular validation.
It consumes the model set, application graph and numeric request set, then
checks the actual scheduler output. Optional comparison requires an externally supplied baseline; none is bundled.

```bash
python -m artifact.mil_artifact.compare_schedules \
  --requests artifact/inputs/frozen/ensemble.json \
  --result /path/to/cpu-output/result.json \
  --reference /path/to/your/cpu-reference.json \
  --output /tmp/mil-comparison-01
```

This comparison does not run a search. It checks method, inputs, resources,
parameters, ordered plans, available stage costs and predicted total cost.
LightRule's unavailable initial cost remains null. Search wall time is not a
cross-machine equality requirement.

For a diagnostic replay of a specific original log, the optional
`artifact.mil_artifact.scheduling_replay` entry reconstructs the independent
LLM-Blender graph and logged input lengths, then starts a real CPU search.
It needs that log and the documented native dependencies. It does not load
weights or start GPU inference. This is separate from normal-input verification.

For comparison with multiple original records, `compare_schedules --sources`
uses a numeric source index and locally available logs. Full source hashes,
parameters, model order and CPU-stage completeness determine eligibility;
all eligible candidates are compared. A later failure does not erase a completed
CPU stage, and neither a matching result nor a newer mtime certifies source
commit identity or final-paper membership. Missing original logs are reported,
not replaced with synthetic measurements.
