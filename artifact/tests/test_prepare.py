from __future__ import annotations

import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import yaml

from artifact.mil_artifact.hashing import sha256_file
from artifact.mil_artifact.config import ArtifactConfigError
from artifact.mil_artifact.prepare import (
    _download_public_dataset,
    main as prepare_main,
    prepare_datasets,
    prepare_models,
)


def _dataset_manifest(path: Path, destination: str, source: Path) -> Path:
    manifest = path / "datasets.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "datasets": [
                    {
                        "id": "sample",
                        "path": destination,
                        "checksum_kind": "file",
                        "size_bytes": source.stat().st_size,
                        "checksum_sha256": sha256_file(source),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return manifest


class DataPreparationTest(unittest.TestCase):
    def test_reuses_verified_existing_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            existing = root / "existing.bin"
            existing.write_bytes(b"artifact-data")
            manifest = _dataset_manifest(root, "${TEST_DATA_ROOT}/existing.bin", existing)
            reports = prepare_datasets(
                manifest,
                {},
                environment={"TEST_DATA_ROOT": str(root)},
            )

        self.assertEqual(reports[0]["status"], "ok")
        self.assertEqual(reports[0]["action"], "reused")

    def test_links_only_a_checksum_matching_explicit_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.bin"
            source.write_bytes(b"artifact-data")
            destination = root / "prepared" / "target.bin"
            manifest = _dataset_manifest(root, str(destination), source)
            reports = prepare_datasets(
                manifest,
                {"sample": source},
                strategy="link",
            )

            self.assertTrue(destination.is_symlink())
            self.assertEqual(destination.read_bytes(), source.read_bytes())
        self.assertEqual(reports[0]["status"], "ok")
        self.assertEqual(reports[0]["action"], "link")

    def test_refuses_to_overwrite_an_existing_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.bin"
            source.write_bytes(b"expected")
            destination = root / "target.bin"
            destination.write_bytes(b"do-not-overwrite")
            manifest = _dataset_manifest(root, str(destination), source)
            reports = prepare_datasets(
                manifest,
                {"sample": source},
                strategy="copy",
            )
            retained = destination.read_bytes()

        self.assertEqual(reports[0]["status"], "error")
        self.assertEqual(retained, b"do-not-overwrite")

    @mock.patch("artifact.mil_artifact.prepare._download_public_dataset")
    def test_downloads_only_then_links_a_checksum_matching_source(self, download) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "downloaded.bin"
            source.write_bytes(b"artifact-data")
            destination = root / "prepared" / "target.bin"
            manifest = _dataset_manifest(root, str(destination), source)
            raw = yaml.safe_load(manifest.read_text(encoding="utf-8"))
            raw["datasets"][0]["acquisition"] = {
                "automatic_download": "allowed",
                "provider": "huggingface_hub",
                "repository_id": "org/data",
                "revision": "revision-1",
                "filename": "downloaded.bin",
            }
            manifest.write_text(yaml.safe_dump(raw), encoding="utf-8")
            download.return_value = source
            reports = prepare_datasets(
                manifest,
                {},
                strategy="link",
                download_public=True,
            )

            self.assertTrue(destination.is_symlink())
            self.assertEqual(destination.read_bytes(), source.read_bytes())
        self.assertEqual(reports[0]["status"], "ok")
        self.assertEqual(reports[0]["action"], "download-link")

    def test_public_download_fails_closed_without_manifest_approval(self) -> None:
        with self.assertRaisesRegex(ArtifactConfigError, "not approved"):
            _download_public_dataset({"id": "restricted"})

    @mock.patch("artifact.mil_artifact.prepare._print_reports")
    @mock.patch("artifact.mil_artifact.prepare.prepare_datasets", return_value=[])
    def test_cli_public_download_defaults_to_link_staging(self, prepare, _print) -> None:
        exit_code = prepare_main(["data", "--download-public"])

        self.assertEqual(exit_code, 0)
        self.assertEqual(prepare.call_args.args[2], "link")
        self.assertTrue(prepare.call_args.kwargs["download_public"])


class ModelPreparationTest(unittest.TestCase):
    def test_reuses_exact_pinned_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            snapshot = root / "models--org--model" / "snapshots" / "revision-1"
            snapshot.mkdir(parents=True)
            (snapshot / "config.json").write_text("{}\n", encoding="utf-8")
            (snapshot / "model.safetensors").write_bytes(b"weights")
            manifest = root / "models.yaml"
            manifest.write_text(
                yaml.safe_dump(
                    {
                        "schema_version": 1,
                        "models": [
                            {
                                "id": "org/model",
                                "revision": "revision-1",
                                "integrity": {
                                    "file_count": 2,
                                    "size_bytes": 10,
                                    "weight_file_count": 1,
                                    "weight_size_bytes": 7,
                                    "config_sha256": sha256_file(snapshot / "config.json"),
                                    "weight_layout_sha256": (
                                        "3167b968b9a588f081f422c55c153f684441345b6"
                                        "c55842a4cb5de5a4df0ecc2"
                                    ),
                                },
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with mock.patch.dict("os.environ", {"MIL_MODEL_ROOT": str(root)}, clear=False):
                reports = prepare_models(manifest)

        self.assertEqual(reports[0]["status"], "ok")
        self.assertEqual(reports[0]["action"], "reused")
        self.assertEqual(reports[0]["file_count"], 2)
        self.assertEqual(reports[0]["weight_file_count"], 1)

    def test_rejects_snapshot_with_manifest_integrity_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            snapshot = root / "models--org--model" / "snapshots" / "revision-1"
            snapshot.mkdir(parents=True)
            (snapshot / "config.json").write_text("{}\n", encoding="utf-8")
            (snapshot / "model.safetensors").write_bytes(b"weights")
            manifest = root / "models.yaml"
            manifest.write_text(
                yaml.safe_dump(
                    {
                        "schema_version": 1,
                        "models": [
                            {
                                "id": "org/model",
                                "revision": "revision-1",
                                "integrity": {"weight_size_bytes": 8},
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with mock.patch.dict("os.environ", {"MIL_MODEL_ROOT": str(root)}, clear=False):
                reports = prepare_models(manifest)

        self.assertEqual(reports[0]["status"], "error")
        self.assertIn("weight_size_bytes mismatch", reports[0]["detail"])

    def test_rejects_changed_license_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            snapshot = root / "models--org--model" / "snapshots" / "revision-1"
            snapshot.mkdir(parents=True)
            (snapshot / "config.json").write_text("{}\n", encoding="utf-8")
            (snapshot / "model.safetensors").write_bytes(b"weights")
            (snapshot / "README.md").write_text("terms\n", encoding="utf-8")
            manifest = root / "models.yaml"
            manifest.write_text(
                yaml.safe_dump(
                    {
                        "schema_version": 1,
                        "models": [
                            {
                                "id": "org/model",
                                "revision": "revision-1",
                                "license": {
                                    "evidence": {"README.md": "0" * 64}
                                },
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with mock.patch.dict("os.environ", {"MIL_MODEL_ROOT": str(root)}, clear=False):
                reports = prepare_models(manifest)

        self.assertEqual(reports[0]["status"], "error")
        self.assertIn("license evidence mismatch", reports[0]["detail"])

    def test_rejects_snapshot_without_model_weights(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            snapshot = root / "models--org--model" / "snapshots" / "revision-1"
            snapshot.mkdir(parents=True)
            (snapshot / "config.json").write_text("{}\n", encoding="utf-8")
            manifest = root / "models.yaml"
            manifest.write_text(
                yaml.safe_dump(
                    {
                        "schema_version": 1,
                        "models": [{"id": "org/model", "revision": "revision-1"}],
                    }
                ),
                encoding="utf-8",
            )
            with mock.patch.dict("os.environ", {"MIL_MODEL_ROOT": str(root)}, clear=False):
                reports = prepare_models(manifest)

        self.assertEqual(reports[0]["status"], "error")
        self.assertIn("model weight files missing", reports[0]["detail"])

    def test_model_download_requires_explicit_terms_acknowledgement(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "models.yaml"
            manifest.write_text(
                yaml.safe_dump(
                    {
                        "schema_version": 1,
                        "models": [
                            {
                                "id": "org/model",
                                "revision": "revision-1",
                                "license": {"status": "confirmed"},
                                "access_requirement": "upstream_terms",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with mock.patch.dict("os.environ", {"MIL_MODEL_ROOT": str(root)}, clear=False):
                reports = prepare_models(manifest, download=True)

        self.assertEqual(reports[0]["status"], "error")
        self.assertIn("--acknowledge-access-terms", reports[0]["detail"])

    def test_model_download_refuses_unresolved_license_after_acknowledgement(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "models.yaml"
            manifest.write_text(
                yaml.safe_dump(
                    {
                        "schema_version": 1,
                        "models": [
                            {
                                "id": "org/model",
                                "revision": "revision-1",
                                "license": {"status": "unresolved"},
                                "access_requirement": "manual_legal_review",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with mock.patch.dict("os.environ", {"MIL_MODEL_ROOT": str(root)}, clear=False):
                reports = prepare_models(
                    manifest,
                    download=True,
                    access_terms_acknowledged=True,
                )

        self.assertEqual(reports[0]["status"], "error")
        self.assertIn("license is unresolved", reports[0]["detail"])

    def test_model_download_uses_exact_revision_after_acknowledgement(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "models.yaml"
            manifest.write_text(
                yaml.safe_dump(
                    {
                        "schema_version": 1,
                        "models": [
                            {
                                "id": "org/model",
                                "revision": "revision-1",
                                "license": {"status": "confirmed"},
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            download = mock.Mock()

            def create_snapshot(**_kwargs: object) -> str:
                snapshot = root / "models--org--model" / "snapshots" / "revision-1"
                snapshot.mkdir(parents=True)
                (snapshot / "config.json").write_text("{}\n", encoding="utf-8")
                (snapshot / "model.safetensors").write_bytes(b"weights")
                return str(snapshot)

            download.side_effect = create_snapshot
            hub = types.ModuleType("huggingface_hub")
            hub.snapshot_download = download
            with mock.patch.dict("os.environ", {"MIL_MODEL_ROOT": str(root)}, clear=False):
                with mock.patch.dict("sys.modules", {"huggingface_hub": hub}):
                    reports = prepare_models(
                        manifest,
                        download=True,
                        access_terms_acknowledged=True,
                    )

        self.assertEqual(reports[0]["status"], "ok")
        self.assertEqual(reports[0]["action"], "downloaded")
        self.assertEqual(download.call_args.kwargs["revision"], "revision-1")


if __name__ == "__main__":
    unittest.main()
