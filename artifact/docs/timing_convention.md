<a id="计时边界与参考结果"></a>

# Timing boundaries and reference results

[简体中文](timing_convention.zh-CN.md)

Artifact timing follows the logging boundaries in
`benchmarks/search_exec_plans.py:get_best_model_schedule`.

The code records `time1` before model construction, dependency correction and
fusion; `time_before_search` after fusion; and `time2` after search. It logs:

- `Total search time = time2 - time1`, including initialization/preparation and search.
- `Total time for preparation before search = time_before_search - time1`, a subset of the above.
- `total running time`, the subsequent execution phase with its cold-start/loading boundaries.

Thus `extra = Total search time` and `total = inference + extra`.
For search-only analysis, use `search_after_preparation = search - preparation`;
do not add it to inclusive search again. Input tokenization precedes this
timing interval and is not part of the logged GPU inference time.

Keep each run's original timing components and source identity together; do not combine timings across runs.
No reference measurements are bundled. Field names alone cannot establish timing-boundary equivalence for external logs without source versions.
