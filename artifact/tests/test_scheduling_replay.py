from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

from artifact.mil_artifact.scheduling_replay import compare, literal_prefix, parse_plan, read_inputs


class SchedulingReplayTest(unittest.TestCase):
    def test_extracts_inputs_without_copying_runtime_output_dictionaries(self):
        text = """test_case: general
outlen_known: False
specify_outlen: False
reqnum: 1
search_method_baseline: greedy_saturn_LPT
running in _get_llm_blender_data
out_lens_dict: {0: [99]}
real model_paths: ['model/example']
real in_edge_dict_with_dummy_inp_nodes: {0: [-1]}
len(inp_lens) of model 0: [12]
Best group seq: [[([0], (1, 0.9, 2, 0, 1))]] [3.5] 3.5 2.0
"""
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.log"
            source.write_text(text)
            record = read_inputs(source)
        self.assertNotIn("out_lens_dict", record["parameters"])
        self.assertEqual(record["input_lengths"], {0: [12]})
        self.assertEqual(record["expected"]["predicted_seconds"], 3.5)

    def test_literal_parser_handles_tuples_and_quoted_brackets(self):
        value, rest = literal_prefix("[('name[0]', (1, 2))] trailing")
        self.assertEqual(value, [('name[0]', (1, 2))])
        self.assertEqual(rest, "trailing")

    def test_parses_plan_and_preserves_stage_order(self):
        plan = parse_plan("[[([0], (1, 0.9, 2, 0, 2))]] [3.5] 3.5 20.0")
        self.assertEqual(plan["predicted_seconds"], 3.5)
        self.assertTrue(compare(plan, plan)["passed"])
        changed = {**plan, "plans": [[([1], (1, 0.9, 2, 0, 2))]]}
        self.assertFalse(compare(plan, changed)["passed"])

    def test_rejects_equal_total_with_different_stage_times(self):
        plan = {"plans": [["a"], ["b"]], "stage_times": [1.0, 2.0], "predicted_seconds": 3.0}
        changed = {**plan, "stage_times": [2.0, 1.0]}
        self.assertFalse(compare(plan, changed)["passed"])

    def test_rejects_different_predicted_time(self):
        plan = {"plans": [["a"]], "predicted_seconds": 3.0}
        self.assertFalse(compare(plan, {**plan, "predicted_seconds": 4.0})["passed"])

    def test_absent_stage_timings_are_not_claimed_as_verified(self):
        plan = {"plans": [["a"]], "predicted_seconds": 3.0}
        result = compare(plan, plan)
        self.assertTrue(result["passed"])
        self.assertIsNone(result["stage_times_match"])
