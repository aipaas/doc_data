import hashlib
import pickle
import tempfile
import unittest
from pathlib import Path

import modelscope_download_audit as audit


def manifest_item(
    path: str,
    content: bytes,
    revision: str = "abcdef123456",
    is_lfs: bool = True,
) -> dict[str, object]:
    return {
        "Path": path,
        "Type": "blob",
        "Revision": revision,
        "IsLFS": is_lfs,
        "Size": len(content),
        "Sha256": hashlib.sha256(content).hexdigest(),
    }


class DownloadAuditTests(unittest.TestCase):
    def test_complete_repository_matches_manifest_and_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "owner" / "dataset"
            target.mkdir(parents=True)
            content = b"complete-content"
            (target / "data.bin").write_bytes(content)
            with (target / ".msc").open("wb") as handle:
                pickle.dump(
                    [{"Path": "data.bin", "Revision": "abcdef123456"}],
                    handle,
                )

            result = audit.audit_repository(
                Path(temporary),
                "owner/dataset",
                target,
                [manifest_item("data.bin", content)],
                hash_max_bytes=1024,
            )

            self.assertEqual(result.status, "complete")
            self.assertEqual(result.remote_files, 1)
            self.assertEqual(result.metadata_files, 1)

    def test_missing_and_stale_files_are_incomplete(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "owner" / "dataset"
            target.mkdir(parents=True)
            content = b"present"
            (target / "present.bin").write_bytes(content)
            with (target / ".msc").open("wb") as handle:
                pickle.dump(
                    [{"Path": "present.bin", "Revision": "old-revision"}],
                    handle,
                )
            manifest = [
                manifest_item("present.bin", content),
                manifest_item("missing.bin", b"missing"),
            ]

            result = audit.audit_repository(
                Path(temporary),
                "owner/dataset",
                target,
                manifest,
                hash_max_bytes=1024,
            )

            self.assertEqual(result.status, "incomplete")
            self.assertEqual(result.missing, ["missing.bin"])
            self.assertEqual(result.revision_mismatch, ["present.bin"])

    def test_non_lfs_crlf_is_hash_normalized(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "owner" / "dataset"
            target.mkdir(parents=True)
            remote_content = b"line-one\nline-two\n"
            (target / "README.md").write_bytes(remote_content.replace(b"\n", b"\r\n"))
            with (target / ".msc").open("wb") as handle:
                pickle.dump(
                    [{"Path": "README.md", "Revision": "abcdef123456"}],
                    handle,
                )

            result = audit.audit_repository(
                Path(temporary),
                "owner/dataset",
                target,
                [manifest_item("README.md", remote_content, is_lfs=False)],
                hash_max_bytes=1024,
            )

            self.assertEqual(result.status, "complete")
            self.assertEqual(result.size_mismatch, [])

    def test_non_lfs_binary_hash_uses_raw_bytes_first(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "owner" / "dataset"
            target.mkdir(parents=True)
            content = b"binary\r\ncontent\x00\xff"
            (target / "image.bin").write_bytes(content)
            with (target / ".msc").open("wb") as handle:
                pickle.dump(
                    [{"Path": "image.bin", "Revision": "abcdef123456"}],
                    handle,
                )

            result = audit.audit_repository(
                Path(temporary),
                "owner/dataset",
                target,
                [manifest_item("image.bin", content, is_lfs=False)],
                hash_max_bytes=1024,
            )

            self.assertEqual(result.status, "complete")
            self.assertEqual(result.hash_mismatch, [])


if __name__ == "__main__":
    unittest.main()
