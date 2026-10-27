"""Content hashing shared by manifest generation and verification."""

from __future__ import annotations

import hashlib
from pathlib import Path


CHUNK_SIZE = 8 * 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(path: Path) -> tuple[str, int, int]:
    """Return tree-v1 digest, regular-file count, and total byte count.

    tree-v1 hashes sorted UTF-8 lines of the form
    ``<content sha256>  <relative POSIX path>\\n``. Directory timestamps and
    absolute locations intentionally do not affect the digest.
    """

    files = sorted(candidate for candidate in path.rglob("*") if candidate.is_file())
    digest = hashlib.sha256()
    total_bytes = 0
    for candidate in files:
        total_bytes += candidate.stat().st_size
        relative = candidate.relative_to(path).as_posix()
        digest.update(f"{sha256_file(candidate)}  {relative}\n".encode("utf-8"))
    return digest.hexdigest(), len(files), total_bytes


def hash_manifest_path(path: Path, kind: str) -> tuple[str, int | None, int]:
    if kind == "file":
        return sha256_file(path), None, path.stat().st_size
    if kind == "tree-v1":
        return sha256_tree(path)
    raise ValueError(f"unsupported checksum kind: {kind}")
