#!/usr/bin/env python3
"""Batch download public AI Studio datasets from a dataset overview query."""

from __future__ import annotations

import argparse
import base64
import binascii
import concurrent.futures
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import zlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import requests


LIST_URL = "https://aistudio.baidu.com/studio/dataset/v2/publiclist"
DETAIL_URL = "https://aistudio.baidu.com/studio/dataset/detail"
FILE_URL = (
    "https://aistudio.baidu.com/llm/files/datasets/"
    "{dataset_id}/file/{file_id}/download"
)
SOURCE_URL = "https://aistudio.baidu.com/datasetoverview?orderType=1&task=26"
PUBLIC_DATASET_TYPE = 2
STATE_VERSION = 1
DEFAULT_PAGE_SIZE = 20
DEFAULT_TASK_ID = 26
DEFAULT_ORDER_TYPE = 1
CHUNK_SIZE = 1024 * 1024
LEGACY_RANGE_CHUNK_SIZE = 16 * 1024 * 1024
LEGACY_PROGRESS_BYTES = 1024 * 1024 * 1024
LEGACY_SIGNED_URL_MAX_AGE = 45.0
SDK_HELPER_COMMAND = "__aistudio_sdk_download"

AUTH_MARKERS = (
    "need login",
    "access token",
    "unauthorized",
    "forbidden",
    "permission denied",
    "no permission",
    "\u65e0\u6743\u9650",
    "\u9700\u8981\u767b\u5f55",
    "\u8bf7\u5148\u767b\u5f55",
)
PRIVATE_MARKERS = AUTH_MARKERS + (
    "private",
    "repo not found",
    "notexisterror",
    "404",
    "\u79c1\u6709",
    "\u4e0d\u5b58\u5728",
)
FILE_API_FALLBACK_MARKERS = ("download.fail:416",)


class BatchDownloadError(RuntimeError):
    """Base error for expected batch downloader failures."""


class ApiError(BatchDownloadError):
    """AI Studio returned an invalid response or application error."""


class AuthenticationRequired(ApiError):
    """A request requires an AI Studio access token."""


class PrivateOrUnavailable(ApiError):
    """A dataset is private, deleted, or otherwise unavailable."""


class IntegrityError(BatchDownloadError):
    """A downloaded file does not match its remote metadata."""


class SignedUrlRefreshRequired(BatchDownloadError):
    """The current signed file URL should be replaced before it expires."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log(level: str, message: str) -> None:
    print(f"[{level}] {message}", flush=True)


def contains_marker(text: str, markers: Iterable[str]) -> bool:
    lowered = text.casefold()
    return any(marker.casefold() in lowered for marker in markers)


def sanitize_component(value: str, fallback: str, max_length: int = 100) -> str:
    value = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", str(value)).strip(" .")
    if not value:
        value = fallback
    if value.upper() in {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{i}" for i in range(1, 10)),
        *(f"LPT{i}" for i in range(1, 10)),
    }:
        value = f"_{value}"
    return value[:max_length].rstrip(" .") or fallback


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise BatchDownloadError(f"Cannot read state file {path}: {exc}") from exc


@dataclass(frozen=True)
class DatasetEntry:
    dataset_id: int
    name: str
    dataset_type: int
    git_login: str
    repo_name: str
    item_version: int
    page: int

    @property
    def key(self) -> str:
        return str(self.dataset_id)

    @property
    def repo_id(self) -> str | None:
        if self.git_login and self.repo_name:
            return f"{self.git_login}/{self.repo_name}"
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "name": self.name,
            "dataset_type": self.dataset_type,
            "git_login": self.git_login,
            "repo_name": self.repo_name,
            "repo_id": self.repo_id,
            "item_version": self.item_version,
            "page": self.page,
        }

    @classmethod
    def from_api(cls, item: dict[str, Any], page: int) -> "DatasetEntry":
        dataset_id = item.get("datasetId")
        if dataset_id is None:
            raise ApiError(f"Page {page} contains an item without datasetId")
        return cls(
            dataset_id=int(dataset_id),
            name=str(item.get("datasetName") or f"dataset-{dataset_id}"),
            dataset_type=int(item.get("datasetType") or 0),
            git_login=str(item.get("gitLogin") or "").strip(),
            repo_name=str(item.get("repoName") or "").strip(),
            item_version=int(item.get("itemVersion") or 1),
            page=page,
        )


@dataclass(frozen=True)
class Catalog:
    entries: tuple[DatasetEntry, ...]
    total_count: int
    total_pages: int
    fetched_pages: int
    passes: int

    @property
    def repo_count(self) -> int:
        return sum(entry.repo_id is not None for entry in self.entries)

    @property
    def legacy_count(self) -> int:
        return sum(entry.repo_id is None for entry in self.entries)


class StateStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.data = load_json(
            path,
            {
                "version": STATE_VERSION,
                "source_url": SOURCE_URL,
                "updated_at": utc_now(),
                "items": {},
            },
        )
        if self.data.get("version") != STATE_VERSION:
            raise BatchDownloadError(
                f"Unsupported state version in {path}: {self.data.get('version')}"
            )
        if not isinstance(self.data.get("items"), dict):
            raise BatchDownloadError(f"Invalid items object in state file {path}")

    def get(self, entry: DatasetEntry) -> dict[str, Any]:
        item = self.data["items"].get(entry.key, {})
        return item if isinstance(item, dict) else {}

    def record(self, entry: DatasetEntry, status: str, **values: Any) -> None:
        previous = self.get(entry)
        self.data["items"][entry.key] = {
            **previous,
            "dataset_id": entry.dataset_id,
            "dataset_name": entry.name,
            "repo_id": entry.repo_id,
            "status": status,
            "updated_at": utc_now(),
            **values,
        }
        self.data["updated_at"] = utc_now()
        atomic_write_json(self.path, self.data)


class AiStudioClient:
    def __init__(
        self,
        token: str | None,
        timeout: float,
        retries: int,
        retry_delay: float,
    ) -> None:
        self.token = token
        self.timeout = timeout
        self.retries = retries
        self.retry_delay = retry_delay
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "aistudio-batch-downloader/1.0", "Accept": "application/json"}
        )

    def close(self) -> None:
        self.session.close()

    def _auth_headers(self) -> dict[str, str]:
        return {"Authorization": self.token} if self.token else {}

    def request_json(
        self,
        method: str,
        url: str,
        *,
        authenticated: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        headers = dict(kwargs.pop("headers", {}))
        if authenticated:
            headers.update(self._auth_headers())

        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                response = self.session.request(
                    method,
                    url,
                    headers=headers,
                    timeout=self.timeout,
                    **kwargs,
                )
                if response.status_code in (401, 403):
                    raise AuthenticationRequired(
                        f"HTTP {response.status_code} for {url}"
                    )
                if response.status_code == 404:
                    raise PrivateOrUnavailable(f"HTTP 404 for {url}")
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, dict):
                    raise ApiError(f"Expected a JSON object from {url}")
                error_code = payload.get("errorCode", 0)
                if str(error_code) != "0":
                    message = str(payload.get("errorMsg") or f"errorCode={error_code}")
                    if contains_marker(message, AUTH_MARKERS):
                        raise AuthenticationRequired(message)
                    if contains_marker(message, PRIVATE_MARKERS):
                        raise PrivateOrUnavailable(message)
                    raise ApiError(f"AI Studio API error from {url}: {message}")
                result = payload.get("result", payload)
                if not isinstance(result, dict):
                    raise ApiError(f"Expected a result object from {url}")
                return result
            except (AuthenticationRequired, PrivateOrUnavailable, ApiError):
                raise
            except (requests.RequestException, ValueError) as exc:
                last_error = exc
                if attempt >= self.retries:
                    break
                delay = self.retry_delay * (2**attempt)
                log("WARN", f"Request failed ({attempt + 1}/{self.retries + 1}): {exc}")
                time.sleep(delay)
        raise ApiError(f"Request failed for {url}: {last_error}") from last_error

    def fetch_catalog(
        self,
        task_id: int,
        order_type: int,
        page_size: int,
        max_pages: int | None,
        catalog_passes: int,
    ) -> Catalog:
        def fetch_page(page: int) -> dict[str, Any]:
            result = self.request_json(
                "POST",
                LIST_URL,
                json={
                    "tags": [task_id],
                    "orderType": order_type,
                    "kw": "",
                    "pageSize": page_size,
                    "p": page,
                    "topic": 0,
                },
            )
            if not isinstance(result.get("data"), list):
                raise ApiError(f"Page {page} has no data list")
            return result

        entries_by_id: dict[int, DatasetEntry] = {}
        total_pages = 1
        total_count = 0
        fetch_pages = 1
        completed_passes = 0

        for catalog_pass in range(1, catalog_passes + 1):
            first = fetch_page(1)
            pass_total_pages = int(first.get("totalPage") or 1)
            pass_total_count = int(first.get("totalCount") or len(first["data"]))
            total_pages = max(total_pages, pass_total_pages)
            total_count = max(total_count, pass_total_count)
            fetch_pages = pass_total_pages
            if max_pages is not None:
                fetch_pages = min(pass_total_pages, max_pages)

            pages: list[tuple[int, dict[str, Any]]] = [(1, first)]
            page_numbers = list(range(2, fetch_pages + 1))
            if catalog_pass % 2 == 0:
                page_numbers.reverse()
            for page in page_numbers:
                log(
                    "INFO",
                    f"Fetching catalog page {page}/{fetch_pages} "
                    f"(pass {catalog_pass}/{catalog_passes})...",
                )
                pages.append((page, fetch_page(page)))

            for page, result in pages:
                for item in result["data"]:
                    if not isinstance(item, dict):
                        continue
                    entry = DatasetEntry.from_api(item, page)
                    entries_by_id.setdefault(entry.dataset_id, entry)
            completed_passes = catalog_pass

            if max_pages is not None or len(entries_by_id) >= total_count:
                break
            if catalog_pass < catalog_passes:
                log(
                    "WARN",
                    f"Catalog moved while paging: expected {total_count}, got "
                    f"{len(entries_by_id)} unique items. Starting reconciliation pass "
                    f"{catalog_pass + 1}/{catalog_passes}.",
                )

        entries = list(entries_by_id.values())
        if max_pages is None and len(entries) < total_count:
            log(
                "WARN",
                f"Catalog is still short after {completed_passes} passes: expected "
                f"{total_count}, got {len(entries)} unique items.",
            )
        return Catalog(
            tuple(entries), total_count, total_pages, fetch_pages, completed_passes
        )

    def fetch_legacy_detail(self, dataset_id: int) -> dict[str, Any]:
        return self.request_json(
            "POST", DETAIL_URL, data={"datasetId": dataset_id}
        )

    def fetch_file_url(self, dataset_id: int, file_id: int) -> str:
        if not self.token:
            raise AuthenticationRequired(
                "Legacy dataset downloads require AISTUDIO_ACCESS_TOKEN"
            )
        result = self.request_json(
            "GET",
            FILE_URL.format(dataset_id=dataset_id, file_id=file_id),
            authenticated=True,
        )
        file_url = result.get("fileUrl")
        if not isinstance(file_url, str) or not file_url.startswith(("http://", "https://")):
            raise ApiError(f"No fileUrl returned for dataset {dataset_id}, file {file_id}")
        return file_url


def load_access_token() -> str | None:
    token = os.getenv("AISTUDIO_ACCESS_TOKEN", "").strip()
    if token:
        return token

    roots: list[Path] = []
    cache_home = os.getenv("AISTUDIO_CACHE_HOME")
    if cache_home:
        roots.append(Path(cache_home))
    roots.append(Path.home())

    candidates: list[Path] = []
    for root in roots:
        candidates.extend(
            [root / ".cache" / "aistudio" / ".auth" / "token", root / ".aistudio_token"]
        )
    for path in candidates:
        try:
            value = path.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if value:
            return value
    return None


def load_dotenv(path: Path) -> bool:
    """Load simple KEY=VALUE entries without overriding process environment."""
    if not path.is_file():
        return False
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except OSError as exc:
        raise BatchDownloadError(f"Cannot read env file {path}: {exc}") from exc
    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            raise BatchDownloadError(
                f"Invalid .env entry at {path}:{line_number}; expected KEY=VALUE"
            )
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise BatchDownloadError(f"Invalid .env key at {path}:{line_number}")
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        os.environ.setdefault(key, value)
    return True


def find_aistudio_command() -> list[str]:
    # Run the official SDK through our small compatibility wrapper. AI Studio SDK
    # 0.3.8 otherwise leaves a successfully resumed small file in ._tmp and exits 0.
    return [sys.executable, "-B", str(Path(__file__).resolve()), SDK_HELPER_COMMAND]


def build_cli_command(
    command_prefix: Sequence[str], repo_id: str, target: Path, max_workers: int
) -> list[str]:
    return [
        *command_prefix,
        "download",
        "--dataset",
        repo_id,
        "--local_dir",
        str(target),
        "--max-workers",
        str(max_workers),
    ]


def run_streaming_command(command: Sequence[str]) -> tuple[int, str]:
    log("CMD", subprocess.list2cmdline(list(command)))
    process = subprocess.Popen(
        list(command),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        bufsize=1,
    )
    output_tail: list[str] = []
    try:
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            output_tail.append(line)
            if len(output_tail) > 300:
                del output_tail[:100]
        return process.wait(), "".join(output_tail)
    except KeyboardInterrupt:
        if process.poll() is None:
            try:
                process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        raise
    finally:
        if process.stdout is not None:
            process.stdout.close()


def hash_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def finish_resumed_sdk_http_download(
    original: Any,
    url: str,
    local_dir: str,
    file_name: str,
    file_size: int,
    headers: dict[str, str] | None = None,
    disable_tqdm: bool = False,
) -> str:
    """Turn the SDK's None-after-resume result into a verified SHA-256 digest."""
    digest = original(
        url,
        local_dir,
        file_name,
        file_size,
        headers=headers,
        disable_tqdm=disable_tqdm,
    )
    if digest:
        return str(digest)

    partial_path = Path(local_dir) / file_name
    actual_size = partial_path.stat().st_size if partial_path.is_file() else -1
    if actual_size != file_size:
        raise IntegrityError(
            f"SDK partial file is incomplete for {file_name}: "
            f"expected {file_size}, got {actual_size}"
        )
    return hash_sha256(partial_path)


def resilient_sdk_download_main(argv: Sequence[str]) -> int:
    """Invoke the official SDK with strict resume and SHA-256 behavior."""
    parser = argparse.ArgumentParser(prog="aistudio download")
    parser.add_argument("action", choices=["download"])
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--local_dir", required=True)
    parser.add_argument("--max-workers", type=int, default=3)
    args = parser.parse_args(argv)

    # This variable is read when aistudio_sdk.utils.caching is imported.
    os.environ["AISTUDIO_ENABLE_DEFAULT_HASH_VALIDATION"] = "true"
    try:
        import importlib

        sdk_file_download = importlib.import_module("aistudio_sdk.file_download")
        sdk_snapshot_download = importlib.import_module("aistudio_sdk.snapshot_download")

        original_http_get = sdk_file_download.http_get_model_file

        def fixed_http_get(
            url: str,
            local_dir: str,
            file_name: str,
            file_size: int,
            headers: dict[str, str] | None = None,
            disable_tqdm: bool = False,
        ) -> str:
            return finish_resumed_sdk_http_download(
                original_http_get,
                url,
                local_dir,
                file_name,
                file_size,
                headers,
                disable_tqdm,
            )

        sdk_file_download.http_get_model_file = fixed_http_get
        original_download_file = sdk_file_download.download_file

        def strict_download_file(*download_args: Any, **download_kwargs: Any) -> str:
            result = original_download_file(*download_args, **download_kwargs)
            if result is None:
                metadata = download_args[1] if len(download_args) > 1 else {}
                filename = metadata.get("path", "unknown") if isinstance(metadata, dict) else "unknown"
                raise RuntimeError(f"AI Studio SDK did not complete file: {filename}")
            return str(result)

        sdk_file_download.download_file = strict_download_file
        sdk_snapshot_download.download_file = strict_download_file
        sdk_snapshot_download.snapshot_download(
            repo_id=args.dataset,
            repo_type="dataset",
            local_dir=args.local_dir,
            max_workers=args.max_workers,
            token=os.getenv("AISTUDIO_ACCESS_TOKEN") or None,
        )
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        log("SDK-ERROR", f"{type(exc).__name__}: {exc}")
        return 1


def repo_target(output_dir: Path, entry: DatasetEntry) -> Path:
    assert entry.repo_id is not None
    return output_dir / sanitize_component(entry.git_login, "owner") / sanitize_component(
        entry.repo_name, f"dataset-{entry.dataset_id}"
    )


def legacy_target(output_dir: Path, entry: DatasetEntry) -> Path:
    name = sanitize_component(entry.name, "dataset", max_length=80)
    return output_dir / "legacy" / f"{entry.dataset_id}_{name}"


def decode_content_md5(value: str | None) -> str | None:
    if not value:
        return None
    try:
        raw = base64.b64decode(value.strip(), validate=True)
    except (ValueError, binascii.Error):
        return None
    return raw.hex() if len(raw) == 16 else None


def remote_md5(headers: requests.structures.CaseInsensitiveDict[str]) -> str | None:
    content_md5 = decode_content_md5(headers.get("Content-MD5"))
    if content_md5:
        return content_md5
    etag = (headers.get("ETag") or "").strip().strip('"')
    return etag.lower() if re.fullmatch(r"[0-9a-fA-F]{32}", etag) else None


def remote_crc32(headers: requests.structures.CaseInsensitiveDict[str]) -> int | None:
    value = headers.get("x-bce-content-crc32")
    if not value:
        return None
    try:
        checksum = int(value.strip(), 10)
    except ValueError:
        return None
    return checksum if 0 <= checksum <= 0xFFFFFFFF else None


def hash_md5(path: Path) -> str:
    try:
        digest = hashlib.md5(usedforsecurity=False)
    except TypeError:
        digest = hashlib.md5()
    with path.open("rb") as handle:
        while chunk := handle.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def hash_crc32(path: Path) -> int:
    checksum = 0
    with path.open("rb") as handle:
        while chunk := handle.read(4 * 1024 * 1024):
            checksum = zlib.crc32(chunk, checksum)
    return checksum & 0xFFFFFFFF


def move_aside(path: Path, reason: str) -> Path:
    suffix = datetime.now().strftime("%Y%m%d-%H%M%S")
    destination = path.with_name(f"{path.name}.{reason}-{suffix}")
    counter = 1
    while destination.exists():
        destination = path.with_name(f"{path.name}.{reason}-{suffix}-{counter}")
        counter += 1
    os.replace(path, destination)
    return destination


def probe_file_url(
    session: requests.Session, file_url: str, timeout: float
) -> tuple[int | None, str | None, int | None]:
    try:
        response = session.head(
            file_url,
            allow_redirects=True,
            timeout=timeout,
            headers={"Accept-Encoding": "identity"},
        )
        response.raise_for_status()
    except requests.RequestException:
        return None, None, None
    size_value = response.headers.get("Content-Length")
    size = int(size_value) if size_value and size_value.isdigit() else None
    expected_md5 = remote_md5(response.headers)
    expected_crc32 = remote_crc32(response.headers)
    if expected_crc32 is not None:
        return size, expected_md5, expected_crc32

    try:
        with session.get(
            file_url,
            headers={"Accept-Encoding": "identity", "Range": "bytes=0-0"},
            stream=True,
            allow_redirects=True,
            timeout=(timeout, max(timeout, 120.0)),
        ) as range_response:
            if range_response.status_code == 206:
                content_range = parse_content_range(
                    range_response.headers.get("Content-Range", "")
                )
                if content_range is not None:
                    range_start, range_end, range_size = content_range
                    if range_start == 0 and range_end == 0:
                        size = range_size
                expected_md5 = expected_md5 or remote_md5(range_response.headers)
                expected_crc32 = remote_crc32(range_response.headers)
    except (requests.RequestException, IntegrityError, ValueError):
        pass
    return size, expected_md5, expected_crc32


def parse_content_range(value: str) -> tuple[int, int, int] | None:
    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", value.strip())
    if not match:
        return None
    start, end, size = (int(part) for part in match.groups())
    if start > end or end >= size:
        return None
    return start, end, size


def fetch_legacy_range(
    session: requests.Session,
    file_url: str,
    start: int,
    end: int,
    expected_size: int,
    timeout: float,
) -> bytes:
    expected_length = end - start + 1
    headers = {
        "Accept-Encoding": "identity",
        "Range": f"bytes={start}-{end}",
    }
    with session.get(
        file_url,
        headers=headers,
        stream=True,
        allow_redirects=True,
        timeout=(timeout, max(timeout, 120.0)),
    ) as response:
        response.raise_for_status()
        content_range = parse_content_range(response.headers.get("Content-Range", ""))
        if response.status_code != 206 or content_range != (start, end, expected_size):
            raise IntegrityError(
                f"Unexpected range response for bytes {start}-{end}: "
                f"status={response.status_code}, "
                f"Content-Range={response.headers.get('Content-Range', '')!r}"
            )
        content = bytearray()
        for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
            if not chunk:
                continue
            content.extend(chunk)
            if len(content) > expected_length:
                raise IntegrityError(
                    f"Range bytes {start}-{end} exceeded {expected_length} bytes"
                )
        if len(content) != expected_length:
            raise IntegrityError(
                f"Range bytes {start}-{end} returned {len(content)} of "
                f"{expected_length} bytes"
            )
        return bytes(content)


def append_parallel_legacy_ranges(
    session: requests.Session,
    file_url: str,
    partial_path: Path,
    offset: int,
    expected_size: int,
    timeout: float,
    workers: int,
    filename: str,
) -> None:
    started_at = time.monotonic()
    starting_offset = offset
    next_progress = min(expected_size, offset + LEGACY_PROGRESS_BYTES)
    with (
        concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor,
        partial_path.open("ab") as handle,
    ):
        while offset < expected_size:
            ranges: list[tuple[int, int]] = []
            next_start = offset
            for _ in range(workers):
                if next_start >= expected_size:
                    break
                end = min(next_start + LEGACY_RANGE_CHUNK_SIZE - 1, expected_size - 1)
                ranges.append((next_start, end))
                next_start = end + 1

            futures = [
                executor.submit(
                    fetch_legacy_range,
                    session,
                    file_url,
                    start,
                    end,
                    expected_size,
                    timeout,
                )
                for start, end in ranges
            ]
            try:
                chunks = [future.result() for future in futures]
            except Exception:
                for future in futures:
                    future.cancel()
                raise

            for chunk in chunks:
                handle.write(chunk)
            handle.flush()
            offset = ranges[-1][1] + 1

            if offset >= next_progress or offset == expected_size:
                elapsed = max(time.monotonic() - started_at, 0.001)
                rate = (offset - starting_offset) / elapsed / (1024 * 1024)
                log(
                    "INFO",
                    f"{filename}: {offset}/{expected_size} bytes "
                    f"({rate:.1f} MiB/s this run)",
                )
                next_progress = min(expected_size, offset + LEGACY_PROGRESS_BYTES)
            if (
                offset < expected_size
                and time.monotonic() - started_at >= LEGACY_SIGNED_URL_MAX_AGE
            ):
                os.fsync(handle.fileno())
                raise SignedUrlRefreshRequired
        os.fsync(handle.fileno())


def verify_file(
    path: Path,
    expected_size: int,
    expected_md5: str | None,
    expected_crc32: int | None = None,
) -> bool:
    if not path.is_file() or path.stat().st_size != expected_size:
        return False
    if expected_md5 and hash_md5(path).casefold() != expected_md5.casefold():
        return False
    if expected_crc32 is not None and hash_crc32(path) != expected_crc32:
        return False
    return True


def download_legacy_file(
    client: AiStudioClient,
    dataset_id: int,
    file_info: dict[str, Any],
    target_dir: Path,
    file_retries: int,
    legacy_workers: int = 1,
) -> dict[str, Any]:
    file_id = int(file_info["fileId"])
    expected_size = int(file_info.get("fileSize") or 0)
    if expected_size < 0:
        raise ApiError(f"Invalid file size for legacy file {file_id}")
    raw_name = str(
        file_info.get("_localName")
        or file_info.get("fileOriginName")
        or file_info.get("fileName")
        or f"file-{file_id}"
    )
    filename = sanitize_component(Path(raw_name).name, f"file-{file_id}", 180)
    target_dir.mkdir(parents=True, exist_ok=True)
    final_path = target_dir / filename
    partial_path = target_dir / f"{filename}.part"

    last_error: Exception | None = None
    consecutive_failures = 0
    while consecutive_failures <= file_retries:
        attempt_offset = partial_path.stat().st_size if partial_path.exists() else 0
        try:
            file_url = client.fetch_file_url(dataset_id, file_id)
            probed_size, expected_md5, expected_crc32 = probe_file_url(
                client.session, file_url, client.timeout
            )
            if probed_size is not None and expected_size and probed_size != expected_size:
                raise IntegrityError(
                    f"Remote size changed for {filename}: metadata={expected_size}, "
                    f"server={probed_size}"
                )
            if not expected_size and probed_size is not None:
                expected_size = probed_size

            if final_path.exists():
                if verify_file(
                    final_path, expected_size, expected_md5, expected_crc32
                ):
                    log("OK", f"Legacy file already valid: {final_path}")
                    return {
                        "file_id": file_id,
                        "filename": filename,
                        "size": expected_size,
                        "md5": expected_md5,
                        "crc32": expected_crc32,
                    }
                moved = move_aside(final_path, "invalid")
                log("WARN", f"Moved invalid file aside: {moved}")

            if partial_path.exists() and partial_path.stat().st_size > expected_size:
                moved = move_aside(partial_path, "invalid")
                log("WARN", f"Moved oversized partial file aside: {moved}")

            offset = partial_path.stat().st_size if partial_path.exists() else 0
            log(
                "INFO",
                f"Downloading legacy file {filename} from byte {offset} "
                f"with {legacy_workers} worker(s) "
                f"(failure streak {consecutive_failures}/{file_retries + 1})...",
            )
            if legacy_workers > 1 and expected_size > 0:
                if offset < expected_size:
                    append_parallel_legacy_ranges(
                        client.session,
                        file_url,
                        partial_path,
                        offset,
                        expected_size,
                        client.timeout,
                        legacy_workers,
                        filename,
                    )
                elif not partial_path.exists():
                    partial_path.touch()
            else:
                headers = {"Accept-Encoding": "identity"}
                if offset:
                    headers["Range"] = f"bytes={offset}-"
                with client.session.get(
                    file_url,
                    headers=headers,
                    stream=True,
                    allow_redirects=True,
                    timeout=(client.timeout, max(client.timeout, 120.0)),
                ) as response:
                    if response.status_code == 416 and offset == expected_size:
                        pass
                    else:
                        response.raise_for_status()
                        append = offset > 0 and response.status_code == 206
                        if append:
                            content_range = response.headers.get("Content-Range", "")
                            if not content_range.startswith(f"bytes {offset}-"):
                                raise IntegrityError(
                                    f"Unexpected Content-Range for {filename}: "
                                    f"{content_range!r}"
                                )
                        if offset > 0 and not append:
                            log(
                                "WARN",
                                f"Server ignored Range for {filename}; restarting file.",
                            )
                            offset = 0
                        mode = "ab" if append else "wb"
                        with partial_path.open(mode) as handle:
                            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                                if chunk:
                                    handle.write(chunk)
                            handle.flush()
                            os.fsync(handle.fileno())

            actual_size = partial_path.stat().st_size if partial_path.exists() else 0
            if actual_size != expected_size:
                raise IntegrityError(
                    f"Size mismatch for {filename}: expected {expected_size}, got {actual_size}"
                )
            if expected_md5 and hash_md5(partial_path).casefold() != expected_md5.casefold():
                moved = move_aside(partial_path, "invalid")
                raise IntegrityError(f"MD5 mismatch for {filename}; moved to {moved}")
            if expected_crc32 is not None and hash_crc32(partial_path) != expected_crc32:
                moved = move_aside(partial_path, "invalid")
                raise IntegrityError(f"CRC32 mismatch for {filename}; moved to {moved}")
            os.replace(partial_path, final_path)
            log("OK", f"Legacy file complete: {final_path}")
            return {
                "file_id": file_id,
                "filename": filename,
                "size": expected_size,
                "md5": expected_md5,
                "crc32": expected_crc32,
            }
        except SignedUrlRefreshRequired:
            log("INFO", f"Refreshing signed URL for {filename}...")
            continue
        except (AuthenticationRequired, PrivateOrUnavailable):
            raise
        except (requests.RequestException, ApiError, IntegrityError, OSError) as exc:
            last_error = exc
            current_offset = (
                partial_path.stat().st_size if partial_path.exists() else 0
            )
            status_code = getattr(getattr(exc, "response", None), "status_code", None)
            if status_code in (401, 403) and current_offset > attempt_offset:
                log("INFO", f"Signed URL expired for {filename}; refreshing it...")
                continue
            if consecutive_failures >= file_retries:
                break
            delay = client.retry_delay * (2**consecutive_failures)
            consecutive_failures += 1
            log("WARN", f"Legacy file {file_id} failed: {exc}; retrying in {delay:g}s")
            time.sleep(delay)
    raise BatchDownloadError(
        f"Legacy file {file_id} failed after {file_retries + 1} attempts: {last_error}"
    ) from last_error


def legacy_state_is_complete(target: Path, state_item: dict[str, Any]) -> bool:
    files = state_item.get("files")
    if not isinstance(files, list) or not files:
        return False
    for item in files:
        if not isinstance(item, dict):
            return False
        filename = item.get("filename")
        size = item.get("size")
        if not isinstance(filename, str) or not isinstance(size, int):
            return False
        if not verify_file(
            target / filename, size, item.get("md5"), item.get("crc32")
        ):
            return False
    return True


def download_repo_dataset(
    entry: DatasetEntry,
    target: Path,
    state: StateStore,
    command_prefix: Sequence[str],
    sdk_workers: int,
    retries: int,
    retry_delay: float,
) -> str:
    assert entry.repo_id is not None
    target.mkdir(parents=True, exist_ok=True)
    state.record(entry, "running", kind="repo", target=str(target))
    command = build_cli_command(command_prefix, entry.repo_id, target, sdk_workers)
    last_output = ""
    for attempt in range(retries + 1):
        try:
            return_code, last_output = run_streaming_command(command)
        except KeyboardInterrupt:
            state.record(
                entry, "interrupted", kind="repo", target=str(target), attempt=attempt + 1
            )
            raise
        if return_code == 0:
            state.record(
                entry,
                "completed",
                kind="repo",
                target=str(target),
                attempt=attempt + 1,
            )
            return "completed"
        if contains_marker(last_output, AUTH_MARKERS):
            state.record(
                entry,
                "skipped_auth_required",
                kind="repo",
                target=str(target),
                return_code=return_code,
                error=last_output[-4000:],
            )
            return "skipped_auth_required"
        if contains_marker(last_output, PRIVATE_MARKERS):
            state.record(
                entry,
                "skipped_private_or_unavailable",
                kind="repo",
                target=str(target),
                return_code=return_code,
                error=last_output[-4000:],
            )
            return "skipped_private_or_unavailable"
        if contains_marker(last_output, FILE_API_FALLBACK_MARKERS):
            state.record(
                entry,
                "failed",
                kind="repo",
                target=str(target),
                return_code=return_code,
                error=last_output[-4000:],
            )
            log("WARN", "Git media endpoint rejected the byte range; switching API.")
            return "failed"
        if attempt < retries:
            delay = retry_delay * (2**attempt)
            log("WARN", f"CLI failed with exit code {return_code}; retrying in {delay:g}s")
            time.sleep(delay)
            continue
        state.record(
            entry,
            "failed",
            kind="repo",
            target=str(target),
            return_code=return_code,
            error=last_output[-4000:],
        )
        return "failed"
    return "failed"


def download_legacy_dataset(
    entry: DatasetEntry,
    target: Path,
    state: StateStore,
    client: AiStudioClient,
    file_retries: int,
    legacy_workers: int = 1,
) -> str:
    state.record(entry, "running", kind="legacy", target=str(target))
    try:
        detail = client.fetch_legacy_detail(entry.dataset_id)
        if int(detail.get("datasetType") or entry.dataset_type) != PUBLIC_DATASET_TYPE:
            state.record(
                entry, "skipped_private", kind="legacy", target=str(target)
            )
            return "skipped_private"
        file_list = detail.get("fileList")
        if not isinstance(file_list, list):
            raise ApiError(f"Legacy dataset {entry.dataset_id} has no fileList")
        if not file_list:
            state.record(
                entry, "completed", kind="legacy", target=str(target), files=[]
            )
            return "completed"

        name_counts: dict[str, int] = {}
        for file_info in file_list:
            if not isinstance(file_info, dict):
                continue
            file_id = int(file_info.get("fileId") or 0)
            raw_name = str(
                file_info.get("fileOriginName")
                or file_info.get("fileName")
                or f"file-{file_id}"
            )
            name = sanitize_component(Path(raw_name).name, f"file-{file_id}", 180)
            name_counts[name.casefold()] = name_counts.get(name.casefold(), 0) + 1
            file_info["_localName"] = name
        for file_info in file_list:
            if not isinstance(file_info, dict):
                continue
            name = str(file_info.get("_localName") or "")
            if name_counts.get(name.casefold(), 0) > 1:
                file_info["_localName"] = f"{int(file_info['fileId'])}_{name}"

        completed_files: list[dict[str, Any]] = []
        for index, file_info in enumerate(file_list, start=1):
            if not isinstance(file_info, dict) or file_info.get("fileId") is None:
                raise ApiError(f"Invalid file metadata in dataset {entry.dataset_id}")
            log(
                "INFO",
                f"Legacy dataset {entry.dataset_id}: file {index}/{len(file_list)}",
            )
            completed_files.append(
                download_legacy_file(
                    client,
                    entry.dataset_id,
                    file_info,
                    target,
                    file_retries,
                    legacy_workers,
                )
            )
            state.record(
                entry,
                "running",
                kind="legacy",
                target=str(target),
                files=completed_files,
            )
        state.record(
            entry,
            "completed",
            kind="legacy",
            target=str(target),
            files=completed_files,
        )
        return "completed"
    except AuthenticationRequired as exc:
        state.record(
            entry,
            "skipped_auth_required",
            kind="legacy",
            target=str(target),
            error=str(exc),
        )
        return "skipped_auth_required"
    except PrivateOrUnavailable as exc:
        state.record(
            entry,
            "skipped_private_or_unavailable",
            kind="legacy",
            target=str(target),
            error=str(exc),
        )
        return "skipped_private_or_unavailable"
    except KeyboardInterrupt:
        state.record(entry, "interrupted", kind="legacy", target=str(target))
        raise
    except Exception as exc:
        state.record(
            entry,
            "failed",
            kind="legacy",
            target=str(target),
            error=str(exc),
        )
        log("ERROR", f"Legacy dataset {entry.dataset_id} failed: {exc}")
        return "failed"


def download_repo_with_legacy_fallback(
    entry: DatasetEntry,
    target: Path,
    state: StateStore,
    command_prefix: Sequence[str],
    sdk_workers: int,
    retries: int,
    retry_delay: float,
    client: AiStudioClient,
    token: str | None,
    force_legacy_api: bool = False,
    legacy_workers: int = 1,
) -> str:
    if force_legacy_api:
        log(
            "INFO",
            f"Using the dataset file API for Git dataset {entry.dataset_id}.",
        )
        return download_legacy_dataset(
            entry, target, state, client, retries, legacy_workers
        )

    result = download_repo_dataset(
        entry,
        target,
        state,
        command_prefix,
        sdk_workers,
        retries,
        retry_delay,
    )
    if result != "failed" or not token:
        return result

    log(
        "WARN",
        f"Git download failed for dataset {entry.dataset_id}; "
        "trying the dataset file API.",
    )
    return download_legacy_dataset(entry, target, state, client, retries, legacy_workers)


def save_catalog(path: Path, catalog: Catalog, args: argparse.Namespace) -> None:
    atomic_write_json(
        path,
        {
            "source_url": SOURCE_URL,
            "fetched_at": utc_now(),
            "query": {
                "task_id": args.task_id,
                "order_type": args.order_type,
                "page_size": args.page_size,
            },
            "total_count": catalog.total_count,
            "total_pages": catalog.total_pages,
            "fetched_pages": catalog.fetched_pages,
            "catalog_passes": catalog.passes,
            "unique_count": len(catalog.entries),
            "repo_count": catalog.repo_count,
            "legacy_count": catalog.legacy_count,
            "datasets": [entry.to_dict() for entry in catalog.entries],
        },
    )


def merge_saved_catalog(
    path: Path, catalog: Catalog, task_id: int, order_type: int
) -> Catalog:
    """Keep entries seen on earlier runs when a moving ranking shifts page boundaries."""
    if not path.is_file():
        return catalog
    try:
        saved = load_json(path, {})
    except BatchDownloadError as exc:
        log("WARN", f"Ignoring unreadable previous catalog: {exc}")
        return catalog
    if not isinstance(saved, dict):
        return catalog
    query = saved.get("query")
    if not isinstance(query, dict):
        return catalog
    if int(query.get("task_id") or -1) != task_id or int(
        query.get("order_type") or -1
    ) != order_type:
        return catalog

    entries_by_id = {entry.dataset_id: entry for entry in catalog.entries}
    previous_entries = saved.get("datasets")
    if not isinstance(previous_entries, list):
        return catalog
    before = len(entries_by_id)
    for item in previous_entries:
        if not isinstance(item, dict):
            continue
        try:
            entry = DatasetEntry(
                dataset_id=int(item["dataset_id"]),
                name=str(item.get("name") or f"dataset-{item['dataset_id']}"),
                dataset_type=int(item.get("dataset_type") or 0),
                git_login=str(item.get("git_login") or "").strip(),
                repo_name=str(item.get("repo_name") or "").strip(),
                item_version=int(item.get("item_version") or 1),
                page=int(item.get("page") or 0),
            )
        except (KeyError, TypeError, ValueError):
            continue
        entries_by_id.setdefault(entry.dataset_id, entry)
    added = len(entries_by_id) - before
    if added:
        log("INFO", f"Recovered {added} dataset(s) from the previous catalog.")
    entries = tuple(entries_by_id.values())
    return Catalog(
        entries=entries,
        total_count=max(catalog.total_count, int(saved.get("total_count") or 0), len(entries)),
        total_pages=max(catalog.total_pages, int(saved.get("total_pages") or 0)),
        fetched_pages=catalog.fetched_pages,
        passes=catalog.passes,
    )


def parse_dataset_ids(value: str) -> tuple[int, ...]:
    dataset_ids: list[int] = []
    seen: set[int] = set()
    for raw_id in value.split(","):
        raw_id = raw_id.strip()
        try:
            dataset_id = int(raw_id)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                "--dataset-ids must be a comma-separated list of positive integers"
            ) from exc
        if dataset_id <= 0:
            raise argparse.ArgumentTypeError(
                "--dataset-ids must be a comma-separated list of positive integers"
            )
        if dataset_id not in seen:
            dataset_ids.append(dataset_id)
            seen.add(dataset_id)
    if not dataset_ids:
        raise argparse.ArgumentTypeError("--dataset-ids cannot be empty")
    return tuple(dataset_ids)


def select_entries_by_ids(
    entries: Sequence[DatasetEntry], dataset_ids: Sequence[int]
) -> list[DatasetEntry]:
    entries_by_id = {entry.dataset_id: entry for entry in entries}
    missing = [dataset_id for dataset_id in dataset_ids if dataset_id not in entries_by_id]
    if missing:
        missing_text = ", ".join(str(dataset_id) for dataset_id in missing)
        raise BatchDownloadError(
            f"Requested dataset ID(s) are absent from the catalog: {missing_text}"
        )
    return [entries_by_id[dataset_id] for dataset_id in dataset_ids]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download all public datasets from an AI Studio dataset overview query. "
            "Git-backed datasets use the official aistudio CLI; legacy datasets use "
            "the SDK-compatible dataset file API."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd() / "aistudio_datasets",
        help="Root directory for downloaded datasets (default: ./aistudio_datasets).",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(__file__).resolve().with_name(".env"),
        help="Credential env file (default: .env beside this script).",
    )
    parser.add_argument("--task-id", type=int, default=DEFAULT_TASK_ID)
    parser.add_argument("--order-type", type=int, default=DEFAULT_ORDER_TYPE)
    parser.add_argument("--page-size", type=int, default=DEFAULT_PAGE_SIZE)
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Limit catalog pages; default fetches every server-reported page.",
    )
    parser.add_argument(
        "--catalog-passes",
        type=int,
        default=3,
        help="Full scans used to reconcile moving page order (default: 3).",
    )
    parser.add_argument(
        "--max-datasets",
        type=int,
        default=None,
        help="Limit datasets after enumeration, useful for a trial run.",
    )
    parser.add_argument(
        "--dataset-ids",
        type=parse_dataset_ids,
        default=None,
        metavar="ID,ID,...",
        help=(
            "Only process these dataset IDs, preserving the supplied order; "
            "fails if any ID is absent from the catalog."
        ),
    )
    parser.add_argument(
        "--sdk-workers",
        type=int,
        default=3,
        help="Value passed to aistudio download --max-workers (default: 3).",
    )
    parser.add_argument(
        "--legacy-workers",
        type=int,
        default=1,
        help=(
            "Concurrent byte-range workers for the dataset file API "
            "(default: 1)."
        ),
    )
    parser.add_argument(
        "--retries", type=int, default=2, help="Retries after a failed request/download."
    )
    parser.add_argument("--retry-delay", type=float, default=2.0)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument(
        "--repo-only",
        action="store_true",
        help="Skip legacy numeric dataset IDs and only run aistudio download commands.",
    )
    parser.add_argument(
        "--force-legacy-api",
        action="store_true",
        help=(
            "Use the dataset file API for Git-backed datasets while retaining their "
            "owner/repository target directories."
        ),
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Fetch and save the catalog without downloading files.",
    )
    parser.add_argument(
        "--print-commands",
        action="store_true",
        help="With --list-only, print each Git-backed aistudio command.",
    )
    parser.add_argument(
        "--verify-completed",
        dest="verify_completed",
        action="store_true",
        default=True,
        help="Revalidate completed Git repositories with SHA-256 (default: enabled).",
    )
    parser.add_argument(
        "--skip-verify-completed",
        dest="verify_completed",
        action="store_false",
        help="Trust completed Git repository state without revalidating local files.",
    )
    parser.add_argument(
        "--retry-skipped",
        action="store_true",
        help="Retry entries previously marked private/unavailable.",
    )
    args = parser.parse_args(argv)
    if args.page_size <= 0 or args.page_size > 100:
        parser.error("--page-size must be in 1..100")
    if args.max_pages is not None and args.max_pages <= 0:
        parser.error("--max-pages must be positive")
    if args.catalog_passes <= 0:
        parser.error("--catalog-passes must be positive")
    if args.max_datasets is not None and args.max_datasets <= 0:
        parser.error("--max-datasets must be positive")
    if args.sdk_workers <= 0:
        parser.error("--sdk-workers must be positive")
    if args.legacy_workers <= 0:
        parser.error("--legacy-workers must be positive")
    if args.retries < 0:
        parser.error("--retries cannot be negative")
    if args.retry_delay < 0 or args.timeout <= 0:
        parser.error("--retry-delay cannot be negative and --timeout must be positive")
    return args


def should_skip_previous(
    entry: DatasetEntry,
    target: Path,
    previous: dict[str, Any],
    verify_completed: bool,
    retry_skipped: bool,
) -> str | None:
    status = previous.get("status")
    if status == "completed" and not verify_completed:
        if entry.repo_id and target.exists():
            return "already_completed"
        if not entry.repo_id and legacy_state_is_complete(target, previous):
            return "already_completed"
    if (
        status in {"skipped_private", "skipped_private_or_unavailable"}
        and not retry_skipped
    ):
        return str(status)
    return None


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / ".aistudio-batch-state.json"
    catalog_path = output_dir / ".aistudio-dataset-catalog.json"
    load_dotenv(args.env_file.expanduser().resolve())
    token = load_access_token()
    client = AiStudioClient(token, args.timeout, args.retries, args.retry_delay)

    log("INFO", f"Source: {SOURCE_URL}")
    log("INFO", f"Output: {output_dir}")
    log("INFO", "Fetching dataset catalog...")
    try:
        catalog = client.fetch_catalog(
            args.task_id,
            args.order_type,
            args.page_size,
            args.max_pages,
            args.catalog_passes,
        )
        if args.max_pages is None:
            catalog = merge_saved_catalog(
                catalog_path, catalog, args.task_id, args.order_type
            )
        save_catalog(catalog_path, catalog, args)
        log(
            "INFO",
            f"Catalog: {len(catalog.entries)} unique datasets, "
            f"{catalog.repo_count} Git repos, {catalog.legacy_count} legacy IDs, "
            f"{catalog.fetched_pages}/{catalog.total_pages} pages, "
            f"{catalog.passes} pass(es).",
        )
        log("INFO", f"Catalog saved: {catalog_path}")

        entries = list(catalog.entries)
        if args.dataset_ids is not None:
            entries = select_entries_by_ids(entries, args.dataset_ids)
            log("INFO", f"Dataset ID whitelist selected {len(entries)} dataset(s).")
        if args.max_datasets is not None:
            entries = entries[: args.max_datasets]
        command_prefix = find_aistudio_command()

        if args.list_only:
            if args.print_commands:
                for entry in entries:
                    if entry.repo_id is None:
                        continue
                    command = build_cli_command(
                        command_prefix,
                        entry.repo_id,
                        repo_target(output_dir, entry),
                        args.sdk_workers,
                    )
                    print(subprocess.list2cmdline(command))
            return 0

        state = StateStore(state_path)
        counts: dict[str, int] = {}
        for index, entry in enumerate(entries, start=1):
            if entry.dataset_type != PUBLIC_DATASET_TYPE:
                state.record(entry, "skipped_private", target="")
                result = "skipped_private"
                counts[result] = counts.get(result, 0) + 1
                continue

            target = (
                repo_target(output_dir, entry)
                if entry.repo_id
                else legacy_target(output_dir, entry)
            )
            previous = state.get(entry)
            skip = should_skip_previous(
                entry,
                target,
                previous,
                args.verify_completed,
                args.retry_skipped,
            )
            if skip:
                log("SKIP", f"[{index}/{len(entries)}] {entry.name}: {skip}")
                counts[skip] = counts.get(skip, 0) + 1
                continue

            if entry.repo_id:
                log(
                    "INFO",
                    f"[{index}/{len(entries)}] Git dataset {entry.repo_id} -> {target}",
                )
                result = download_repo_with_legacy_fallback(
                    entry,
                    target,
                    state,
                    command_prefix,
                    args.sdk_workers,
                    args.retries,
                    args.retry_delay,
                    client,
                    token,
                    args.force_legacy_api,
                    args.legacy_workers,
                )
            elif args.repo_only:
                state.record(
                    entry,
                    "skipped_legacy",
                    kind="legacy",
                    target=str(target),
                    error="No gitLogin/repoName is exposed by the catalog",
                )
                result = "skipped_legacy"
            elif not token:
                state.record(
                    entry,
                    "skipped_auth_required",
                    kind="legacy",
                    target=str(target),
                    error="Set AISTUDIO_ACCESS_TOKEN to download legacy datasets",
                )
                result = "skipped_auth_required"
            else:
                log(
                    "INFO",
                    f"[{index}/{len(entries)}] Legacy dataset {entry.dataset_id} -> {target}",
                )
                result = download_legacy_dataset(
                    entry,
                    target,
                    state,
                    client,
                    args.retries,
                    args.legacy_workers,
                )
            counts[result] = counts.get(result, 0) + 1

        summary = ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))
        log("SUMMARY", summary or "No datasets selected")
        failed = counts.get("failed", 0)
        auth_skipped = counts.get("skipped_auth_required", 0)
        if auth_skipped:
            log(
                "WARN",
                f"{auth_skipped} legacy datasets require AISTUDIO_ACCESS_TOKEN; "
                "set it and run the same command again.",
            )
        return 1 if failed else 0
    except KeyboardInterrupt:
        log("WARN", "Interrupted. Run the same command again to continue.")
        return 130
    except BatchDownloadError as exc:
        log("ERROR", str(exc))
        return 1
    finally:
        client.close()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == SDK_HELPER_COMMAND:
        raise SystemExit(resilient_sdk_download_main(sys.argv[2:]))
    raise SystemExit(main())
