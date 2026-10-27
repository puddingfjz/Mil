from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from artifact.mil_artifact.config import artifact_root, expand_environment, load_yaml
from artifact.mil_artifact.hashing import sha256_file
from artifact.mil_artifact.legacy_results import audit_manifest, export_reference_run
from artifact.mil_artifact.schema import read_json, read_jsonl, validate_metric_record


class LegacyReferenceAuditTest(unittest.TestCase):
    def test_recomputes_checksum_and_statistics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            logs = []
            for index, runtime in enumerate((10.0, 12.0, 14.0), 1):
                log = root / f"run-{index}.log"
                log.write_text(
                    f"Best group seq: [] 9.0 0\n"
                    f"Total search time: 2.0\n"
                    f"Total time for preparation before search: 1.0\n"
                    f"total running time: {runtime}s\n",
                    encoding="utf-8",
                )
                logs.append(log)
            manifest = root / "references.yaml"
            manifest.write_text(
                yaml.safe_dump(
                    {
                        "schema_version": 1,
                        "references": [
                            {
                                "id": "test",
                                "legacy_method": "greedy_saturn_LPT",
                                "sample_count": 3,
                                "sources": [
                                    {"path": str(log), "checksum_sha256": sha256_file(log)}
                                    for log in logs
                                ],
                                "metrics": {
                                    "real_running_time_seconds": {
                                        "median": 12.0,
                                        "min": 10.0,
                                        "max": 14.0,
                                    },
                                    "total_time_seconds": {
                                        "median": 14.0,
                                        "min": 12.0,
                                        "max": 16.0,
                                    },
                                },
                            }
                        ],
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
            report = audit_manifest(manifest)

        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["issue_count"], 0)



if __name__ == '__main__':
    unittest.main()
