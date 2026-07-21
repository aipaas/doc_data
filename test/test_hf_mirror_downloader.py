import hashlib
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from unittest.mock import patch

import httpx
from huggingface_hub.hf_api import RepoFile

from hf_mirror_downloader import (
    InsufficientDiskSpace,
    RemoteFileSpec,
    Sha256Manifest,
    default_local_dir,
    direct_incomplete_path,
    direct_mirror_download,
    download_dataset,
    ensure_file,
    file_spec_from_repo_file,
    force_mirror_api_url,
    list_repo_tree_from_mirror,
    verify_local_file,
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_blob_sha1(data: bytes) -> str:
    hasher = hashlib.sha1()
    hasher.update(f"blob {len(data)}\0".encode("ascii"))
    hasher.update(data)
    return hasher.hexdigest()


class DownloaderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory(
            dir=Path(__file__).resolve().parent
        )
        self.local_dir = Path(self.temporary_directory.name)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def manifest(self, commit: str = "a" * 40) -> Sha256Manifest:
        manifest, warning = Sha256Manifest.load(
            self.local_dir, "owner/dataset", commit
        )
        self.assertIsNone(warning)
        return manifest

    def test_explicit_windows_data_root_builds_repository_directory(self) -> None:
        destination = default_local_dir(
            "kensho/PubTables-v2", Path(r"E:\data\doc")
        )

        self.assertEqual(
            destination,
            Path(r"E:\data\doc\huggingface\kensho\PubTables-v2"),
        )

    def test_environment_data_root_builds_repository_directory(self) -> None:
        with patch.dict("os.environ", {"HF_MIRROR_DATA_ROOT": r"E:\data\doc"}):
            destination = default_local_dir("juliozhao/DocSynth300K")

        self.assertEqual(
            destination,
            Path(r"E:\data\doc\huggingface\juliozhao\DocSynth300K"),
        )

    def test_official_pagination_link_is_forced_back_to_mirror(self) -> None:
        official_url = (
            "https://huggingface.co/api/datasets/gvl610/iFLYTAB/tree/abc123"
            "?recursive=true&cursor=next-page"
        )

        rewritten = force_mirror_api_url(official_url)

        self.assertEqual(
            rewritten,
            "https://hf-mirror.com/api/datasets/gvl610/iFLYTAB/tree/abc123"
            "?recursive=true&cursor=next-page",
        )

    def test_every_tree_page_uses_mirror_even_when_next_link_does_not(self) -> None:
        calls: list[str] = []

        def fake_page_loader(
            url: str,
            params: dict[str, str] | None,
            headers: dict[str, str],
            attempts: int,
        ) -> tuple[list[dict[str, object]], str | None]:
            del params, headers, attempts
            calls.append(url)
            if len(calls) == 1:
                return (
                    [{"type": "file", "path": "first.txt", "size": 1, "oid": "a" * 40}],
                    "https://huggingface.co/api/datasets/gvl610/iFLYTAB/tree/abc123"
                    "?cursor=second-page",
                )
            return (
                [{"type": "file", "path": "second.txt", "size": 2, "oid": "b" * 40}],
                None,
            )

        files = list_repo_tree_from_mirror(
            repo_id="gvl610/iFLYTAB",
            commit="abc123",
            token=None,
            attempts=3,
            page_loader=fake_page_loader,
        )

        self.assertEqual([file.path for file in files], ["first.txt", "second.txt"])
        self.assertEqual(len(calls), 2)
        self.assertTrue(all(url.startswith("https://hf-mirror.com/api/") for url in calls))

    def test_lfs_file_uses_remote_sha256(self) -> None:
        data = b"verified lfs payload"
        target = self.local_dir / "data.bin"
        target.write_bytes(data)
        spec = RemoteFileSpec("data.bin", len(data), sha256(data), None)

        result = verify_local_file(target, spec, show_progress=False)

        self.assertTrue(result.ok)
        self.assertEqual(result.sha256, sha256(data))

    def test_same_size_corruption_fails_sha256(self) -> None:
        expected = b"abcdef"
        target = self.local_dir / "data.bin"
        target.write_bytes(b"abcdeg")
        spec = RemoteFileSpec("data.bin", len(expected), sha256(expected), None)

        result = verify_local_file(target, spec, show_progress=False)

        self.assertFalse(result.ok)
        self.assertIn("SHA-256 mismatch", result.reason)

    def test_regular_git_file_gets_sha256_and_git_verification(self) -> None:
        data = b"regular git file\n"
        target = self.local_dir / "README.md"
        target.write_bytes(data)
        spec = RemoteFileSpec("README.md", len(data), None, git_blob_sha1(data))

        result = verify_local_file(
            target,
            spec,
            manifest_sha256=sha256(data),
            show_progress=False,
        )

        self.assertTrue(result.ok)
        self.assertTrue(result.manifest_sha256_matches)

    def test_same_size_corruption_fails_git_blob_verification(self) -> None:
        expected = b"regular git file\n"
        target = self.local_dir / "README.md"
        target.write_bytes(b"regular git filf\n")
        spec = RemoteFileSpec(
            "README.md", len(expected), None, git_blob_sha1(expected)
        )

        result = verify_local_file(target, spec, show_progress=False)

        self.assertFalse(result.ok)
        self.assertIn("Git blob hash mismatch", result.reason)

    def test_size_mismatch_fails_before_hashing(self) -> None:
        target = self.local_dir / "data.bin"
        target.write_bytes(b"short")
        spec = RemoteFileSpec("data.bin", 100, sha256(b"x" * 100), None)

        result = verify_local_file(target, spec, show_progress=False)

        self.assertFalse(result.ok)
        self.assertIn("size mismatch", result.reason)

    def test_repo_metadata_selects_lfs_sha256_or_git_blob(self) -> None:
        lfs = RepoFile(
            path="large.bin",
            size=4,
            oid="b" * 40,
            lfs={"size": 4, "oid": "c" * 64, "pointerSize": 128},
        )
        regular = RepoFile(path="README.md", size=2, oid="d" * 40)

        self.assertEqual(file_spec_from_repo_file(lfs).remote_sha256, "c" * 64)
        self.assertEqual(file_spec_from_repo_file(regular).git_blob_sha1, "d" * 40)

    def test_manifest_round_trip(self) -> None:
        data = b"manifest"
        spec = RemoteFileSpec("data.bin", len(data), sha256(data), None)
        manifest = self.manifest()
        manifest.record(spec, sha256(data))
        manifest.save()

        loaded, warning = Sha256Manifest.load(
            self.local_dir, "owner/dataset", "a" * 40
        )

        self.assertIsNone(warning)
        self.assertEqual(loaded.expected_sha256(spec), sha256(data))

    def test_manifest_parallel_updates_keep_every_file(self) -> None:
        manifest = self.manifest()
        specs = [
            RemoteFileSpec(f"file-{index}.bin", 0, sha256(b""), None)
            for index in range(20)
        ]

        def record(spec: RemoteFileSpec) -> None:
            manifest.record(spec, sha256(b""))
            manifest.save()

        with ThreadPoolExecutor(max_workers=4) as executor:
            list(executor.map(record, specs))

        loaded, warning = Sha256Manifest.load(
            self.local_dir, "owner/dataset", "a" * 40
        )
        self.assertIsNone(warning)
        self.assertEqual(len(loaded.files), len(specs))

    def test_default_two_workers_process_files_concurrently(self) -> None:
        specs = [
            RemoteFileSpec(f"file-{index}.bin", 0, sha256(b""), None)
            for index in range(4)
        ]
        state_lock = Lock()
        active = 0
        peak_active = 0

        def fake_ensure_file(**kwargs: object) -> bool:
            nonlocal active, peak_active
            del kwargs
            with state_lock:
                active += 1
                peak_active = max(peak_active, active)
            time.sleep(0.05)
            with state_lock:
                active -= 1
            return True

        with (
            patch(
                "hf_mirror_downloader.fetch_snapshot",
                return_value=("a" * 40, specs),
            ),
            patch("hf_mirror_downloader.ensure_file", side_effect=fake_ensure_file),
        ):
            result = download_dataset(
                "owner/dataset",
                local_dir=self.local_dir,
                attempts=1,
                show_hash_progress=False,
            )

        self.assertEqual(result, 0)
        self.assertEqual(peak_active, 2)

    def test_valid_existing_file_skips_download(self) -> None:
        data = b"already complete"
        spec = RemoteFileSpec("data.bin", len(data), sha256(data), None)
        (self.local_dir / spec.path).write_bytes(data)
        manifest = self.manifest()

        def unexpected_download(**kwargs: object) -> str:
            self.fail(f"download should not be called: {kwargs}")

        result = ensure_file(
            repo_id="owner/dataset",
            commit="a" * 40,
            local_dir=self.local_dir,
            spec=spec,
            manifest=manifest,
            token=None,
            attempts=2,
            show_hash_progress=False,
            download_impl=unexpected_download,
            sleeper=lambda _: None,
        )

        self.assertTrue(result)

    def test_corrupt_existing_file_is_deleted_and_cleanly_downloaded(self) -> None:
        good = b"correct bytes"
        target = self.local_dir / "data.bin"
        target.write_bytes(b"incorrect byt")
        spec = RemoteFileSpec("data.bin", len(good), sha256(good), None)
        manifest = self.manifest()
        force_flags: list[bool] = []

        def fake_download(**kwargs: object) -> str:
            force_flags.append(bool(kwargs["force_download"]))
            self.assertFalse(target.exists())
            target.write_bytes(good)
            return str(target)

        result = ensure_file(
            repo_id="owner/dataset",
            commit="a" * 40,
            local_dir=self.local_dir,
            spec=spec,
            manifest=manifest,
            token=None,
            attempts=2,
            show_hash_progress=False,
            download_impl=fake_download,
            sleeper=lambda _: None,
        )

        self.assertTrue(result)
        self.assertEqual(force_flags, [True])
        self.assertEqual(target.read_bytes(), good)

    def test_network_retry_preserves_incomplete_download(self) -> None:
        data = b"resumed bytes"
        target = self.local_dir / "data.bin"
        spec = RemoteFileSpec("data.bin", len(data), sha256(data), None)
        manifest = self.manifest()
        force_flags: list[bool] = []

        def fake_download(**kwargs: object) -> str:
            force_flags.append(bool(kwargs["force_download"]))
            if len(force_flags) == 1:
                raise OSError("temporary network failure")
            target.write_bytes(data)
            return str(target)

        result = ensure_file(
            repo_id="owner/dataset",
            commit="a" * 40,
            local_dir=self.local_dir,
            spec=spec,
            manifest=manifest,
            token=None,
            attempts=2,
            show_hash_progress=False,
            download_impl=fake_download,
            sleeper=lambda _: None,
        )

        self.assertTrue(result)
        self.assertEqual(force_flags, [False, False])

    def test_xet_size_limit_falls_back_to_direct_mirror_download(self) -> None:
        data = b"large xet payload"
        target = self.local_dir / "large.bin"
        spec = RemoteFileSpec("large.bin", len(data), sha256(data), None)
        manifest = self.manifest()

        def unavailable_regular_download(**kwargs: object) -> str:
            del kwargs
            raise ValueError(
                "The file is too large to be downloaded using the regular download "
                "method. Install `hf_xet` with `pip install hf_xet` for xet-powered "
                "downloads."
            )

        def direct_download(**kwargs: object) -> Path:
            self.assertEqual(kwargs["repo_id"], "owner/dataset")
            target.write_bytes(data)
            return target

        with patch(
            "hf_mirror_downloader.direct_mirror_download",
            side_effect=direct_download,
        ) as direct:
            result = ensure_file(
                repo_id="owner/dataset",
                commit="a" * 40,
                local_dir=self.local_dir,
                spec=spec,
                manifest=manifest,
                token=None,
                attempts=2,
                show_hash_progress=False,
                download_impl=unavailable_regular_download,
                sleeper=lambda _: None,
            )

        self.assertTrue(result)
        self.assertEqual(direct.call_count, 1)
        self.assertEqual(target.read_bytes(), data)

    def test_direct_mirror_download_resumes_with_range(self) -> None:
        data = b"abcdef"
        spec = RemoteFileSpec("nested/data.bin", len(data), sha256(data), None)
        target = self.local_dir / "nested" / "data.bin"
        partial = direct_incomplete_path(target)
        partial.parent.mkdir(parents=True)
        partial.write_bytes(data[:3])
        requested_hosts: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requested_hosts.append(request.url.host)
            self.assertEqual(request.headers["range"], "bytes=3-")
            if request.url.host == "hf-mirror.com":
                return httpx.Response(
                    302,
                    headers={
                        "location": "https://cas-bridge.xethub.hf.co/object?signature=x"
                    },
                )
            self.assertEqual(request.url.host, "cas-bridge.xethub.hf.co")
            return httpx.Response(
                206,
                headers={
                    "content-range": "bytes 3-5/6",
                    "content-length": "3",
                },
                content=data[3:],
            )

        transport = httpx.MockTransport(handler)
        with httpx.Client(transport=transport) as client:
            result = direct_mirror_download(
                repo_id="owner/dataset",
                commit="a" * 40,
                local_dir=self.local_dir,
                spec=spec,
                token="secret-token",
                force_download=False,
                show_progress=False,
                stop_event=None,
                client=client,
            )

        self.assertEqual(result, target)
        self.assertEqual(target.read_bytes(), data)
        self.assertFalse(partial.exists())
        self.assertEqual(
            requested_hosts, ["hf-mirror.com", "cas-bridge.xethub.hf.co"]
        )

    def test_direct_mirror_download_rejects_official_host_redirect(self) -> None:
        data = b"x"
        spec = RemoteFileSpec("data.bin", len(data), sha256(data), None)
        requested_hosts: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requested_hosts.append(request.url.host)
            if request.url.host != "hf-mirror.com":
                self.fail(f"unexpected request to {request.url.host}")
            return httpx.Response(
                302,
                headers={
                    "location": "https://huggingface.co/owner/dataset/resolve/main/data.bin"
                },
            )

        transport = httpx.MockTransport(handler)
        with httpx.Client(transport=transport) as client:
            with self.assertRaisesRegex(RuntimeError, "disallowed host"):
                direct_mirror_download(
                    repo_id="owner/dataset",
                    commit="a" * 40,
                    local_dir=self.local_dir,
                    spec=spec,
                    token=None,
                    force_download=False,
                    show_progress=False,
                    stop_event=None,
                    client=client,
                )

        self.assertEqual(requested_hosts, ["hf-mirror.com"])

    def test_direct_mirror_download_checks_disk_space_before_request(self) -> None:
        data = b"payload"
        spec = RemoteFileSpec("data.bin", len(data), sha256(data), None)

        class DiskUsage:
            free = 0

        with patch("hf_mirror_downloader.shutil.disk_usage", return_value=DiskUsage()):
            with self.assertRaisesRegex(InsufficientDiskSpace, "insufficient disk space"):
                direct_mirror_download(
                    repo_id="owner/dataset",
                    commit="a" * 40,
                    local_dir=self.local_dir,
                    spec=spec,
                    token=None,
                    force_download=False,
                    show_progress=False,
                    stop_event=None,
                )

        self.assertFalse(direct_incomplete_path(self.local_dir / "data.bin").exists())

    def test_bad_download_is_deleted_then_force_downloaded(self) -> None:
        good = b"good payload"
        bad = b"baad payload"
        target = self.local_dir / "data.bin"
        spec = RemoteFileSpec("data.bin", len(good), sha256(good), None)
        manifest = self.manifest()
        force_flags: list[bool] = []

        def fake_download(**kwargs: object) -> str:
            force_flags.append(bool(kwargs["force_download"]))
            target.write_bytes(bad if len(force_flags) == 1 else good)
            return str(target)

        result = ensure_file(
            repo_id="owner/dataset",
            commit="a" * 40,
            local_dir=self.local_dir,
            spec=spec,
            manifest=manifest,
            token=None,
            attempts=2,
            show_hash_progress=False,
            download_impl=fake_download,
            sleeper=lambda _: None,
        )

        self.assertTrue(result)
        self.assertEqual(force_flags, [False, True])
        self.assertEqual(target.read_bytes(), good)


if __name__ == "__main__":
    unittest.main()
