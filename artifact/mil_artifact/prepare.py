"""Idempotently prepare checksum-pinned Artifact datasets and model snapshots."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from .config import (
    ArtifactConfigError,
    artifact_root,
    expand_environment,
    has_unresolved_environment,
    load_yaml,
)
from .hashing import hash_manifest_path, sha256_file


def _source_assignments(expressions: Sequence[str]) -> dict[str, Path]:
    sources: dict[str, Path] = {}
    for expression in expressions:
        identifier, separator, raw_path = expression.partition("=")
        if not separator or not identifier or not raw_path:
            raise ArtifactConfigError(
                f"source must use DATASET_ID=PATH syntax, got {expression!r}"
            )
        if identifier in sources:
            raise ArtifactConfigError(f"duplicate source for dataset {identifier!r}")
        sources[identifier] = Path(raw_path).expanduser().resolve()
    return sources


def _path_report(
    identifier: str,
    path: Path,
    entry: Mapping[str, Any],
    verify_checksums: bool,
    action: str,
) -> dict[str, Any]:
    if not path.exists():
        return {
            "id": identifier,
            "status": "error",
            "action": action,
            "path": os.fspath(path),
            "detail": "missing",
        }
    kind = str(entry.get("checksum_kind", "file"))
    expected = entry.get("checksum_sha256")
    if not expected:
        return {
            "id": identifier,
            "status": "warning",
            "action": action,
            "path": os.fspath(path),
            "detail": "checksum is not frozen",
        }
    if not verify_checksums:
        return {
            "id": identifier,
            "status": "warning",
            "action": action,
            "path": os.fspath(path),
            "detail": "checksum verification skipped",
        }
    actual, file_count, size_bytes = hash_manifest_path(path, kind)
    matches = (
        actual == expected
        and entry.get("file_count") in (None, file_count)
        and entry.get("size_bytes") in (None, size_bytes)
    )
    return {
        "id": identifier,
        "status": "ok" if matches else "error",
        "action": action,
        "path": os.fspath(path),
        "checksum_kind": kind,
        "checksum_sha256": actual,
        "size_bytes": size_bytes,
        "file_count": file_count,
        "detail": "verified" if matches else "manifest checksum/count/size mismatch",
    }


def _stage_dataset(source: Path, destination: Path, strategy: str) -> None:
    if not source.exists():
        raise ArtifactConfigError(f"dataset source does not exist: {source}")
    if os.path.lexists(destination):
        raise ArtifactConfigError(f"refusing to overwrite existing dataset target: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if strategy == "link":
        destination.symlink_to(source, target_is_directory=source.is_dir())
        return
    if strategy != "copy":
        raise ArtifactConfigError(f"unsupported dataset preparation strategy: {strategy}")

    with tempfile.TemporaryDirectory(
        prefix=f".{destination.name}.prepare-",
        dir=destination.parent,
    ) as temporary:
        payload = Path(temporary) / "payload"
        if source.is_dir():
            shutil.copytree(source, payload, symlinks=True)
        else:
            shutil.copy2(source, payload)
        payload.replace(destination)


def _download_public_dataset(
    entry: Mapping[str, Any],
    cache_dir: Path | None = None,
) -> Path:
    acquisition = entry.get("acquisition", {})
    if not isinstance(acquisition, dict) or acquisition.get("automatic_download") != "allowed":
        raise ArtifactConfigError(
            f"dataset {entry.get('id')!r} is not approved for automatic download"
        )
    if acquisition.get("provider") != "huggingface_hub":
        raise ArtifactConfigError(
            f"dataset {entry.get('id')!r} has unsupported acquisition provider"
        )
    required = ("repository_id", "revision", "filename")
    missing = [name for name in required if not acquisition.get(name)]
    if missing:
        raise ArtifactConfigError(
            f"dataset {entry.get('id')!r} acquisition is missing: {', '.join(missing)}"
        )
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ArtifactConfigError(
            "huggingface_hub is required for --download-public"
        ) from exc
    try:
        downloaded = hf_hub_download(
            repo_id=str(acquisition["repository_id"]),
            repo_type=str(acquisition.get("repo_type", "dataset")),
            revision=str(acquisition["revision"]),
            filename=str(acquisition["filename"]),
            cache_dir=os.fspath(cache_dir) if cache_dir else None,
        )
    except Exception as exc:
        raise ArtifactConfigError(
            f"public download failed for dataset {entry.get('id')!r}: {exc}"
        ) from exc
    return Path(downloaded).resolve()


def prepare_datasets(
    manifest_path: Path,
    source_assignments: Mapping[str, Path],
    strategy: str = "check",
    selected_ids: Sequence[str] = (),
    verify_checksums: bool = True,
    environment: Mapping[str, str] | None = None,
    download_public: bool = False,
    download_cache: Path | None = None,
) -> list[dict[str, Any]]:
    raw_manifest = load_yaml(manifest_path)
    manifest = expand_environment(raw_manifest, environment)
    selected = set(selected_ids)
    known = {str(entry.get("id")) for entry in manifest.get("datasets", [])}
    unknown = sorted((selected | set(source_assignments)) - known)
    if unknown:
        raise ArtifactConfigError("unknown dataset id(s): " + ", ".join(unknown))

    reports: list[dict[str, Any]] = []
    for entry in manifest.get("datasets", []):
        identifier = str(entry.get("id"))
        if selected and identifier not in selected:
            continue
        raw_path = str(entry.get("path", ""))
        if has_unresolved_environment(raw_path):
            reports.append(
                {
                    "id": identifier,
                    "status": "error",
                    "action": "none",
                    "path": raw_path,
                    "detail": "unresolved environment variable",
                }
            )
            continue
        destination = Path(raw_path).expanduser().resolve()
        existing = _path_report(
            identifier,
            destination,
            entry,
            verify_checksums,
            "reused",
        )
        if existing["status"] != "error" or destination.exists():
            reports.append(existing)
            continue

        source = source_assignments.get(identifier)
        downloaded_source = False
        if source is None and download_public:
            try:
                source = _download_public_dataset(entry, download_cache)
                downloaded_source = True
            except (ArtifactConfigError, OSError, ValueError) as exc:
                reports.append(
                    {
                        "id": identifier,
                        "status": "error",
                        "action": "download",
                        "path": os.fspath(destination),
                        "detail": str(exc),
                    }
                )
                continue
        if source is None or strategy == "check":
            reports.append(existing)
            continue
        source_report = _path_report(
            identifier,
            source,
            entry,
            True,
            "source-verified",
        )
        if source_report["status"] != "ok":
            source_report["detail"] = "source rejected: " + source_report["detail"]
            reports.append(source_report)
            continue
        try:
            _stage_dataset(source, destination, strategy)
        except (ArtifactConfigError, OSError) as exc:
            reports.append(
                {
                    "id": identifier,
                    "status": "error",
                    "action": strategy,
                    "path": os.fspath(destination),
                    "detail": str(exc),
                }
            )
            continue
        reports.append(
            _path_report(
                identifier,
                destination,
                entry,
                verify_checksums,
                f"download-{strategy}" if downloaded_source else strategy,
            )
        )
    return reports


def _model_cache_root() -> Path:
    configured = os.environ.get("MIL_MODEL_ROOT") or os.environ.get("HF_HUB_CACHE")
    if configured:
        return Path(configured).expanduser().resolve()
    if os.environ.get("HF_HOME"):
        return (Path(os.environ["HF_HOME"]).expanduser() / "hub").resolve()
    return (Path.home() / ".cache" / "huggingface" / "hub").resolve()


def _model_snapshot(root: Path, identifier: str, revision: str) -> Path:
    return root / ("models--" + identifier.replace("/", "--")) / "snapshots" / revision


def model_snapshot_health(
    snapshot: Path,
    expected_integrity: Mapping[str, Any] | None = None,
    expected_evidence: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Return structural and lightweight integrity health for one HF snapshot.

    Weight payloads are deliberately not re-read: revision pinning identifies
    the upstream content, while the sorted path/size digest detects missing or
    truncated local shards without hashing hundreds of gigabytes per preflight.
    """

    broken = [path for path in snapshot.rglob("*") if path.is_symlink() and not path.exists()]
    files = sorted(
        (path for path in snapshot.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(snapshot).as_posix(),
    )
    problems: list[str] = []
    config = snapshot / "config.json"
    config_sha256: str | None = None
    if not config.is_file():
        problems.append("config.json missing")
    else:
        try:
            parsed_config = json.loads(config.read_text(encoding="utf-8"))
            if not isinstance(parsed_config, dict):
                problems.append("config.json is not a JSON object")
            config_sha256 = sha256_file(config)
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            problems.append(f"config.json is unreadable or invalid: {exc}")
    weight_files = [
        path
        for path in files
        if path.suffix in {".bin", ".pt", ".pth", ".safetensors"}
    ]
    if not weight_files:
        problems.append("model weight files missing")
    if broken:
        problems.append(f"snapshot has {len(broken)} broken link(s)")

    weight_layout = hashlib.sha256()
    for path in weight_files:
        relative = path.relative_to(snapshot).as_posix()
        weight_layout.update(f"{path.stat().st_size}  {relative}\n".encode("utf-8"))
    observed = {
        "file_count": len(files),
        "size_bytes": sum(path.stat().st_size for path in files),
        "weight_file_count": len(weight_files),
        "weight_size_bytes": sum(path.stat().st_size for path in weight_files),
        "config_sha256": config_sha256,
        "weight_layout_sha256": weight_layout.hexdigest(),
    }
    for name, expected in (expected_integrity or {}).items():
        if name not in observed:
            continue
        actual = observed[name]
        if expected != actual:
            problems.append(f"{name} mismatch: expected {expected}, got {actual}")

    for relative, expected in (expected_evidence or {}).items():
        evidence_path = Path(relative)
        if evidence_path.is_absolute() or ".." in evidence_path.parts:
            problems.append(f"unsafe evidence path: {relative}")
            continue
        resolved = snapshot / evidence_path
        if not resolved.is_file():
            problems.append(f"license evidence missing: {relative}")
            continue
        actual = sha256_file(resolved)
        if actual != expected:
            problems.append(
                f"license evidence mismatch for {relative}: expected {expected}, got {actual}"
            )

    return {
        "ok": not problems,
        **observed,
        "license_evidence_file_count": len(expected_evidence or {}),
        "broken_links": len(broken),
        "detail": "verified" if not problems else "; ".join(problems),
    }


def prepare_models(
    manifest_path: Path,
    download: bool = False,
    selected_ids: Sequence[str] = (),
    access_terms_acknowledged: bool = False,
) -> list[dict[str, Any]]:
    manifest = load_yaml(manifest_path)
    selected = set(selected_ids)
    known = {str(entry.get("id")) for entry in manifest.get("models", [])}
    unknown = sorted(selected - known)
    if unknown:
        raise ArtifactConfigError("unknown model id(s): " + ", ".join(unknown))
    root = _model_cache_root()
    reports: list[dict[str, Any]] = []

    for entry in manifest.get("models", []):
        identifier = str(entry.get("id"))
        if selected and identifier not in selected:
            continue
        revision = entry.get("revision")
        if not isinstance(revision, str) or not revision:
            reports.append(
                {
                    "id": identifier,
                    "status": "error",
                    "action": "none",
                    "detail": "revision is not frozen",
                }
            )
            continue
        snapshot = _model_snapshot(root, identifier, revision)
        action = "reused"
        if not snapshot.is_dir() and download:
            access_requirement = str(entry.get("access_requirement", "unspecified"))
            license_status = str(entry.get("license", {}).get("status", "unresolved"))
            if not access_terms_acknowledged:
                reports.append(
                    {
                        "id": identifier,
                        "revision": revision,
                        "status": "error",
                        "action": "download",
                        "detail": (
                            "download requires --acknowledge-access-terms; "
                            f"manifest requirement={access_requirement}"
                        ),
                    }
                )
                continue
            if license_status == "unresolved":
                reports.append(
                    {
                        "id": identifier,
                        "revision": revision,
                        "status": "error",
                        "action": "download",
                        "detail": (
                            "automatic download disabled because model license is unresolved; "
                            f"manifest requirement={access_requirement}"
                        ),
                    }
                )
                continue
            try:
                from huggingface_hub import snapshot_download

                snapshot_download(
                    repo_id=identifier,
                    revision=revision,
                    cache_dir=os.fspath(root),
                )
                action = "downloaded"
            except Exception as exc:
                reports.append(
                    {
                        "id": identifier,
                        "revision": revision,
                        "status": "error",
                        "action": "download",
                        "detail": str(exc),
                    }
                )
                continue
        if not snapshot.is_dir():
            reports.append(
                {
                    "id": identifier,
                    "revision": revision,
                    "status": "error",
                    "action": "none",
                    "path": os.fspath(snapshot),
                    "detail": "snapshot missing; pass --download after accepting access terms",
                }
            )
            continue
        health = model_snapshot_health(
            snapshot,
            entry.get("integrity"),
            entry.get("license", {}).get("evidence"),
        )
        reports.append(
            {
                "id": identifier,
                "revision": revision,
                "status": "ok" if health["ok"] else "error",
                "action": action,
                "path": os.fspath(snapshot),
                **{name: value for name, value in health.items() if name != "ok"},
            }
        )
    return reports


def _print_reports(reports: Sequence[Mapping[str, Any]], as_json: bool) -> None:
    if as_json:
        print(json.dumps(list(reports), indent=2, sort_keys=True))
        return
    for report in reports:
        print(
            f"[{str(report['status']).upper():7}] {report['id']}: "
            f"action={report.get('action')} {report.get('detail')}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="resource", required=True)

    data = subparsers.add_parser("data", help="verify or stage prepared datasets")
    data.add_argument("--manifest", type=Path, default=artifact_root() / "manifests/datasets.yaml")
    data.add_argument("--dataset", action="append", default=[], help="dataset id; repeatable")
    data.add_argument(
        "--source",
        action="append",
        default=[],
        metavar="DATASET_ID=PATH",
        help="explicit prepared source for a missing dataset; repeatable",
    )
    data.add_argument("--strategy", choices=["check", "link", "copy"], default="check")
    data.add_argument(
        "--download-public",
        action="store_true",
        help="fetch only manifest-approved exact public objects; defaults to link staging",
    )
    data.add_argument(
        "--download-cache",
        type=Path,
        help="optional Hugging Face cache for --download-public",
    )
    data.add_argument("--skip-checksums", action="store_true")
    data.add_argument("--json", action="store_true")

    models = subparsers.add_parser("models", help="verify or download pinned model snapshots")
    models.add_argument("--manifest", type=Path, default=artifact_root() / "manifests/models.yaml")
    models.add_argument("--model", action="append", default=[], help="model id; repeatable")
    models.add_argument("--download", action="store_true")
    models.add_argument(
        "--acknowledge-access-terms",
        action="store_true",
        help="confirm that the evaluator has reviewed and satisfied each selected model's upstream terms",
    )
    models.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.resource == "data":
            strategy = "link" if args.download_public and args.strategy == "check" else args.strategy
            reports = prepare_datasets(
                args.manifest.expanduser().resolve(),
                _source_assignments(args.source),
                strategy,
                args.dataset,
                not args.skip_checksums,
                download_public=args.download_public,
                download_cache=(
                    args.download_cache.expanduser().resolve()
                    if args.download_cache
                    else None
                ),
            )
        else:
            reports = prepare_models(
                args.manifest.expanduser().resolve(),
                args.download,
                args.model,
                args.acknowledge_access_terms,
            )
    except (ArtifactConfigError, OSError, ValueError) as exc:
        print(f"resource preparation error: {exc}", file=sys.stderr)
        return 2
    _print_reports(reports, args.json)
    return 1 if any(report["status"] == "error" for report in reports) else 0


if __name__ == "__main__":
    raise SystemExit(main())
