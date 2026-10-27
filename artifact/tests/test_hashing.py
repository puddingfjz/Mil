from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from artifact.mil_artifact.hashing import hash_manifest_path, sha256_file, sha256_tree


class ContentHashingTest(unittest.TestCase):
    def test_file_hash_reports_digest_and_size(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "payload.bin"
            path.write_bytes(b"mil-artifact\n")
            digest, file_count, size = hash_manifest_path(path, "file")
        self.assertEqual(digest, hashlib.sha256(b"mil-artifact\n").hexdigest())
        self.assertIsNone(file_count)
        self.assertEqual(size, 13)

    def test_tree_v1_is_path_and_content_stable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "nested").mkdir()
            (root / "z.txt").write_text("z", encoding="utf-8")
            (root / "nested" / "a.txt").write_text("a", encoding="utf-8")

            expected_lines = (
                f"{sha256_file(root / 'nested' / 'a.txt')}  nested/a.txt\n"
                f"{sha256_file(root / 'z.txt')}  z.txt\n"
            )
            digest, count, size = sha256_tree(root)

        self.assertEqual(digest, hashlib.sha256(expected_lines.encode("utf-8")).hexdigest())
        self.assertEqual(count, 2)
        self.assertEqual(size, 2)

    def test_unknown_hash_kind_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "unsupported checksum kind"):
                hash_manifest_path(Path(temporary), "unknown")


if __name__ == "__main__":
    unittest.main()
