import base64
import hashlib
import tempfile
import unittest
import os
import zlib
from pathlib import Path
from unittest.mock import patch

import requests

import aistudio_batch_downloader as downloader


class FakeResponse:
    def __init__(self, status_code=200, headers=None, chunks=()):
        self.status_code = status_code
        self.headers = requests.structures.CaseInsensitiveDict(headers or {})
        self._chunks = list(chunks)

    def raise_for_status(self):
        if self.status_code >= 400:
            response = requests.Response()
            response.status_code = self.status_code
            raise requests.HTTPError(response=response)

    def iter_content(self, chunk_size):
        del chunk_size
        yield from self._chunks

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


class FakeFileSession:
    def __init__(self, full_content, partial_size=0):
        self.full_content = full_content
        self.partial_size = partial_size
        self.get_headers = []

    def head(self, *args, **kwargs):
        del args, kwargs
        md5 = base64.b64encode(hashlib.md5(self.full_content).digest()).decode()
        return FakeResponse(
            headers={
                "Content-Length": str(len(self.full_content)),
                "Content-MD5": md5,
                "x-bce-content-crc32": str(zlib.crc32(self.full_content) & 0xFFFFFFFF),
            }
        )

    def get(self, *args, **kwargs):
        del args
        headers = kwargs["headers"]
        self.get_headers.append(dict(headers))
        range_value = headers.get("Range")
        if range_value:
            start_value, end_value = range_value.removeprefix("bytes=").split("-", 1)
            offset = int(start_value)
            end = int(end_value) if end_value else len(self.full_content) - 1
            return FakeResponse(
                status_code=206,
                headers={
                    "Content-Range": (
                        f"bytes {offset}-{end}/"
                        f"{len(self.full_content)}"
                    )
                },
                chunks=[self.full_content[offset : end + 1]],
            )
        return FakeResponse(status_code=200, chunks=[self.full_content])


class FakeLegacyClient:
    def __init__(self, content, partial_size=0):
        self.session = FakeFileSession(content, partial_size)
        self.timeout = 1.0
        self.retry_delay = 0.0

    def fetch_file_url(self, dataset_id, file_id):
        self.last_request = (dataset_id, file_id)
        return "https://example.invalid/signed-file"


class DatasetEntryTests(unittest.TestCase):
    def test_repo_id_is_built_from_catalog_fields(self):
        entry = downloader.DatasetEntry.from_api(
            {
                "datasetId": 10,
                "datasetName": "sample",
                "datasetType": 2,
                "gitLogin": "owner",
                "repoName": "repo",
                "itemVersion": 2,
            },
            page=3,
        )
        self.assertEqual(entry.repo_id, "owner/repo")
        self.assertEqual(entry.page, 3)

    def test_legacy_entry_has_no_repo_id(self):
        entry = downloader.DatasetEntry.from_api(
            {"datasetId": 11, "datasetName": "old", "datasetType": 2}, page=1
        )
        self.assertIsNone(entry.repo_id)

    def test_windows_component_is_sanitized(self):
        self.assertEqual(downloader.sanitize_component('a<b>:c"d', "x"), "a_b__c_d")
        self.assertEqual(downloader.sanitize_component("CON", "x"), "_CON")
        self.assertEqual(downloader.sanitize_component("...", "fallback"), "fallback")


class CatalogTests(unittest.TestCase):
    def test_fetches_server_reported_pages_and_deduplicates(self):
        client = downloader.AiStudioClient(None, timeout=1, retries=0, retry_delay=0)
        pages = {
            1: {
                "page": 1,
                "pageSize": 2,
                "totalPage": 2,
                "totalCount": 3,
                "data": [
                    {
                        "datasetId": 1,
                        "datasetName": "one",
                        "datasetType": 2,
                        "gitLogin": "a",
                        "repoName": "one",
                    },
                    {"datasetId": 2, "datasetName": "two", "datasetType": 2},
                ],
            },
            2: {
                "page": 2,
                "pageSize": 2,
                "totalPage": 2,
                "totalCount": 3,
                "data": [
                    {"datasetId": 2, "datasetName": "two", "datasetType": 2},
                    {
                        "datasetId": 3,
                        "datasetName": "three",
                        "datasetType": 2,
                        "gitLogin": "b",
                        "repoName": "three",
                    },
                ],
            },
        }

        def fake_request(method, url, **kwargs):
            self.assertEqual((method, url), ("POST", downloader.LIST_URL))
            self.assertEqual(kwargs["json"]["tags"], [26])
            return pages[kwargs["json"]["p"]]

        try:
            with patch.object(client, "request_json", side_effect=fake_request):
                catalog = client.fetch_catalog(26, 1, 2, None, 3)
        finally:
            client.close()
        self.assertEqual([entry.dataset_id for entry in catalog.entries], [1, 2, 3])
        self.assertEqual(catalog.repo_count, 2)
        self.assertEqual(catalog.legacy_count, 1)

    def test_reconciliation_reverses_pages_and_finds_moving_entry(self):
        client = downloader.AiStudioClient(None, timeout=1, retries=0, retry_delay=0)
        calls = []
        current_pass = 0

        def item(dataset_id):
            return {
                "datasetId": dataset_id,
                "datasetName": f"dataset-{dataset_id}",
                "datasetType": 2,
            }

        def fake_request(method, url, **kwargs):
            nonlocal current_pass
            self.assertEqual((method, url), ("POST", downloader.LIST_URL))
            page = kwargs["json"]["p"]
            if page == 1:
                current_pass += 1
            calls.append(page)
            first_pass = {1: [1, 2], 2: [2, 3], 3: [4]}
            second_pass = {1: [1, 2], 2: [2, 3], 3: [4, 5]}
            ids = (first_pass if current_pass == 1 else second_pass)[page]
            return {
                "totalPage": 3,
                "totalCount": 5,
                "data": [item(dataset_id) for dataset_id in ids],
            }

        try:
            with patch.object(client, "request_json", side_effect=fake_request):
                catalog = client.fetch_catalog(26, 1, 2, None, 3)
        finally:
            client.close()
        self.assertEqual(calls, [1, 2, 3, 1, 3, 2])
        self.assertEqual({entry.dataset_id for entry in catalog.entries}, {1, 2, 3, 4, 5})

    def test_previous_catalog_is_merged_for_same_query(self):
        current = downloader.Catalog(
            entries=(downloader.DatasetEntry(1, "one", 2, "", "", 1, 1),),
            total_count=2,
            total_pages=1,
            fetched_pages=1,
            passes=1,
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "catalog.json"
            path.write_text(
                '{"query":{"task_id":26,"order_type":1},"total_count":2,'
                '"total_pages":1,"datasets":[{"dataset_id":2,"name":"two",'
                '"dataset_type":2,"git_login":"","repo_name":"",'
                '"item_version":1,"page":1}]}',
                encoding="utf-8",
            )
            merged = downloader.merge_saved_catalog(path, current, 26, 1)
        self.assertEqual([entry.dataset_id for entry in merged.entries], [1, 2])

    def test_dataset_id_selection_preserves_requested_order(self):
        entries = [
            downloader.DatasetEntry(1, "one", 2, "", "", 1, 1),
            downloader.DatasetEntry(2, "two", 2, "", "", 1, 1),
            downloader.DatasetEntry(3, "three", 2, "", "", 1, 1),
        ]
        selected = downloader.select_entries_by_ids(entries, (3, 1))
        self.assertEqual([entry.dataset_id for entry in selected], [3, 1])

    def test_dataset_id_selection_rejects_missing_ids(self):
        entries = [downloader.DatasetEntry(1, "one", 2, "", "", 1, 1)]
        with self.assertRaisesRegex(downloader.BatchDownloadError, "2"):
            downloader.select_entries_by_ids(entries, (1, 2))

    def test_dataset_id_argument_deduplicates_without_reordering(self):
        args = downloader.parse_args(["--dataset-ids", "3, 1,3"])
        self.assertEqual(args.dataset_ids, (3, 1))

    def test_force_legacy_api_argument(self):
        args = downloader.parse_args(
            ["--force-legacy-api", "--legacy-workers", "8"]
        )
        self.assertTrue(args.force_legacy_api)
        self.assertEqual(args.legacy_workers, 8)


class CommandTests(unittest.TestCase):
    def test_official_cli_command_shape(self):
        command = downloader.build_cli_command(
            ["aistudio"], "owner/repo", Path("D:/datasets/owner/repo"), 3
        )
        self.assertEqual(
            command,
            [
                "aistudio",
                "download",
                "--dataset",
                "owner/repo",
                "--local_dir",
                str(Path("D:/datasets/owner/repo")),
                "--max-workers",
                "3",
            ],
        )

    def test_private_markers_cover_sdk_not_found(self):
        self.assertTrue(
            downloader.contains_marker(
                "NotExistError: repo not found", downloader.PRIVATE_MARKERS
            )
        )

    def test_dotenv_loads_token_without_overriding_environment(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / ".env"
            path.write_text(
                "# comment\nAISTUDIO_ACCESS_TOKEN=from-file\nOTHER_VALUE='quoted'\n",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"AISTUDIO_ACCESS_TOKEN": "from-process"}, clear=False):
                os.environ.pop("OTHER_VALUE", None)
                self.assertTrue(downloader.load_dotenv(path))
                self.assertEqual(os.environ["AISTUDIO_ACCESS_TOKEN"], "from-process")
                self.assertEqual(os.environ["OTHER_VALUE"], "quoted")

    def test_sdk_command_uses_local_resilience_wrapper(self):
        command = downloader.find_aistudio_command()
        self.assertEqual(command[0], downloader.sys.executable)
        self.assertIn(downloader.SDK_HELPER_COMMAND, command)

    def test_resumed_sdk_file_gets_sha256_digest(self):
        content = b"resumed content"
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "file.bin"

            def completed_resume(*args, **kwargs):
                del args, kwargs
                target.write_bytes(content)
                return None

            digest = downloader.finish_resumed_sdk_http_download(
                completed_resume,
                "https://example.invalid/file",
                temporary,
                "file.bin",
                len(content),
            )
            self.assertEqual(digest, hashlib.sha256(content).hexdigest())

    def test_incomplete_sdk_resume_is_an_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "file.bin"

            def incomplete_resume(*args, **kwargs):
                del args, kwargs
                target.write_bytes(b"short")
                return None

            with self.assertRaises(downloader.IntegrityError):
                downloader.finish_resumed_sdk_http_download(
                    incomplete_resume,
                    "https://example.invalid/file",
                    temporary,
                    "file.bin",
                    100,
                )


class StateTests(unittest.TestCase):
    def test_completed_repo_is_skipped_when_target_exists(self):
        entry = downloader.DatasetEntry(1, "one", 2, "owner", "repo", 2, 1)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "owner" / "repo"
            target.mkdir(parents=True)
            state = downloader.StateStore(root / "state.json")
            state.record(entry, "completed", kind="repo", target=str(target))
            loaded = downloader.StateStore(root / "state.json")
            result = downloader.should_skip_previous(
                entry, target, loaded.get(entry), False, False
            )
            self.assertEqual(result, "already_completed")

    def test_legacy_completion_rechecks_size_and_md5(self):
        content = b"complete legacy file"
        md5 = hashlib.md5(content).hexdigest()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "file.bin").write_bytes(content)
            state_item = {
                "files": [
                    {"filename": "file.bin", "size": len(content), "md5": md5}
                ]
            }
            self.assertTrue(downloader.legacy_state_is_complete(root, state_item))
            (root / "file.bin").write_bytes(b"corrupt")
            self.assertFalse(downloader.legacy_state_is_complete(root, state_item))

    def test_repo_auth_failure_is_retried_on_next_run(self):
        entry = downloader.DatasetEntry(1, "one", 2, "owner", "repo", 2, 1)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "owner" / "repo"
            state = downloader.StateStore(root / "state.json")
            with patch.object(
                downloader,
                "run_streaming_command",
                return_value=(1, "Unauthorized: access token expired"),
            ):
                result = downloader.download_repo_dataset(
                    entry, target, state, ["aistudio"], 2, 0, 0
                )
            self.assertEqual(result, "skipped_auth_required")
            previous = state.get(entry)
            self.assertEqual(previous["status"], "skipped_auth_required")
            self.assertIsNone(
                downloader.should_skip_previous(entry, target, previous, True, False)
            )

    def test_repo_range_error_returns_without_retries(self):
        entry = downloader.DatasetEntry(1, "one", 2, "owner", "repo", 2, 1)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "owner" / "repo"
            state = downloader.StateStore(root / "state.json")
            with patch.object(
                downloader,
                "run_streaming_command",
                return_value=(1, "RequestError: download.fail:416"),
            ) as run_command:
                result = downloader.download_repo_dataset(
                    entry, target, state, ["aistudio"], 3, 5, 0
                )
            self.assertEqual(result, "failed")
            run_command.assert_called_once()

    def test_failed_repo_download_falls_back_to_file_api(self):
        entry = downloader.DatasetEntry(1, "one", 2, "owner", "repo", 2, 1)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "owner" / "repo"
            state = downloader.StateStore(root / "state.json")
            client = object()
            with (
                patch.object(
                    downloader, "download_repo_dataset", return_value="failed"
                ) as repo_download,
                patch.object(
                    downloader, "download_legacy_dataset", return_value="completed"
                ) as legacy_download,
            ):
                result = downloader.download_repo_with_legacy_fallback(
                    entry,
                    target,
                    state,
                    ["aistudio"],
                    3,
                    2,
                    1.0,
                    client,
                    "token",
                )
            self.assertEqual(result, "completed")
            repo_download.assert_called_once()
            legacy_download.assert_called_once_with(entry, target, state, client, 2, 1)

    def test_force_legacy_api_bypasses_repo_download(self):
        entry = downloader.DatasetEntry(1, "one", 2, "owner", "repo", 2, 1)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "owner" / "repo"
            state = downloader.StateStore(root / "state.json")
            client = object()
            with (
                patch.object(downloader, "download_repo_dataset") as repo_download,
                patch.object(
                    downloader, "download_legacy_dataset", return_value="completed"
                ) as legacy_download,
            ):
                result = downloader.download_repo_with_legacy_fallback(
                    entry,
                    target,
                    state,
                    ["aistudio"],
                    3,
                    2,
                    1.0,
                    client,
                    "token",
                    force_legacy_api=True,
                )
            self.assertEqual(result, "completed")
            repo_download.assert_not_called()
            legacy_download.assert_called_once_with(entry, target, state, client, 2, 1)


class LegacyDownloadTests(unittest.TestCase):
    def test_signed_url_refresh_does_not_consume_failure_retries(self):
        content = b"abcdefghij"
        client = FakeLegacyClient(content)
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary)
            with (
                patch.object(downloader, "LEGACY_RANGE_CHUNK_SIZE", 2),
                patch.object(downloader, "LEGACY_SIGNED_URL_MAX_AGE", 0),
            ):
                downloader.download_legacy_file(
                    client,
                    dataset_id=100,
                    file_info={
                        "fileId": 198,
                        "fileOriginName": "sample.bin",
                        "fileSize": len(content),
                    },
                    target_dir=target,
                    file_retries=0,
                    legacy_workers=2,
                )
            self.assertEqual((target / "sample.bin").read_bytes(), content)
            self.assertGreater(len(client.session.get_headers), 2)

    def test_parallel_ranges_resume_and_append_in_order(self):
        content = b"abcdefghij"
        client = FakeLegacyClient(content)
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary)
            (target / "sample.bin.part").write_bytes(content[:2])
            with patch.object(downloader, "LEGACY_RANGE_CHUNK_SIZE", 2):
                downloader.download_legacy_file(
                    client,
                    dataset_id=100,
                    file_info={
                        "fileId": 199,
                        "fileOriginName": "sample.bin",
                        "fileSize": len(content),
                    },
                    target_dir=target,
                    file_retries=0,
                    legacy_workers=3,
                )
            self.assertEqual((target / "sample.bin").read_bytes(), content)
            ranges = sorted(
                headers["Range"]
                for headers in client.session.get_headers
                if headers["Range"] != "bytes=0-0"
            )
            self.assertEqual(
                ranges,
                ["bytes=2-3", "bytes=4-5", "bytes=6-7", "bytes=8-9"],
            )

    def test_range_response_must_match_requested_offsets(self):
        response = FakeResponse(
            status_code=206,
            headers={"Content-Range": "bytes 1-2/4"},
            chunks=[b"ab"],
        )

        class MismatchedSession:
            def get(self, *args, **kwargs):
                del args, kwargs
                return response

        with self.assertRaisesRegex(downloader.IntegrityError, "Unexpected range"):
            downloader.fetch_legacy_range(
                MismatchedSession(),
                "https://example.invalid/file",
                0,
                1,
                4,
                1.0,
            )

    def test_resumes_part_file_and_checks_remote_md5(self):
        content = b"abcdef"
        client = FakeLegacyClient(content)
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary)
            (target / "sample.bin.part").write_bytes(content[:3])
            result = downloader.download_legacy_file(
                client,
                dataset_id=100,
                file_info={
                    "fileId": 200,
                    "fileOriginName": "sample.bin",
                    "fileSize": len(content),
                },
                target_dir=target,
                file_retries=0,
            )
            self.assertEqual((target / "sample.bin").read_bytes(), content)
            self.assertFalse((target / "sample.bin.part").exists())
            self.assertEqual(client.session.get_headers[0]["Range"], "bytes=3-")
            self.assertEqual(result["md5"], hashlib.md5(content).hexdigest())
            self.assertEqual(result["crc32"], zlib.crc32(content) & 0xFFFFFFFF)

    def test_invalid_final_file_is_preserved_and_redownloaded(self):
        content = b"correct"
        client = FakeLegacyClient(content)
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary)
            (target / "sample.bin").write_bytes(b"bad")
            downloader.download_legacy_file(
                client,
                dataset_id=100,
                file_info={
                    "fileId": 201,
                    "fileOriginName": "sample.bin",
                    "fileSize": len(content),
                },
                target_dir=target,
                file_retries=0,
            )
            self.assertEqual((target / "sample.bin").read_bytes(), content)
            invalid_files = list(target.glob("sample.bin.invalid-*"))
            self.assertEqual(len(invalid_files), 1)
            self.assertEqual(invalid_files[0].read_bytes(), b"bad")


if __name__ == "__main__":
    unittest.main()
