from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import time
from collections.abc import Callable, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from threading import Event, Lock, RLock
from typing import Any, TypeVar
from urllib.parse import quote, urljoin, urlsplit, urlunsplit


MIRROR_ENDPOINT = "https://hf-mirror.com"

# These variables must be set before importing huggingface_hub. The SDK Xet
# transport is disabled; oversized Xet objects use the mirror-signed HTTP Range
# fallback implemented below.
os.environ["HF_ENDPOINT"] = MIRROR_ENDPOINT
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")

import httpx  # noqa: E402
from huggingface_hub import HfApi, hf_hub_download  # noqa: E402
from huggingface_hub.hf_api import RepoFile  # noqa: E402
from huggingface_hub.utils import build_hf_headers, hf_raise_for_status  # noqa: E402
from tqdm import tqdm  # noqa: E402


HASH_CHUNK_SIZE = 8 * 1024 * 1024
HASH_PROGRESS_THRESHOLD = 8 * 1024 * 1024
DOWNLOAD_CHUNK_SIZE = 8 * 1024 * 1024
MAX_DOWNLOAD_REDIRECTS = 5
MIN_FREE_SPACE_RESERVE = 5 * 1024 * 1024 * 1024
MANIFEST_NAME = ".hf-mirror-sha256.json"
MANIFEST_SCHEMA_VERSION = 1
DATA_ROOT_ENV = "HF_MIRROR_DATA_ROOT"
_HEX_RE = re.compile(r"^[0-9a-f]+$")
_CONTENT_RANGE_RE = re.compile(r"^bytes (\d+)-(\d+)/(\d+)$", re.IGNORECASE)
_XET_SIZE_LIMIT_MARKER = "too large to be downloaded using the regular download method"
T = TypeVar("T")
_CONSOLE_LOCK = Lock()


class DownloadCancelled(Exception):
    pass


class InsufficientDiskSpace(RuntimeError):
    pass


def console_print(message: str) -> None:
    with _CONSOLE_LOCK:
        print(message, flush=True)


def check_cancelled(stop_event: Event | None) -> None:
    if stop_event is not None and stop_event.is_set():
        raise DownloadCancelled("download cancelled")


def sleep_or_cancel(
    delay: float,
    stop_event: Event | None,
    sleeper: Callable[[float], None],
) -> None:
    if stop_event is None:
        sleeper(delay)
    elif stop_event.wait(delay):
        raise DownloadCancelled("download cancelled")


@dataclass(frozen=True, slots=True)
class RemoteFileSpec:
    path: str
    size: int
    remote_sha256: str | None
    git_blob_sha1: str | None

    @property
    def source_algorithm(self) -> str:
        return "sha256" if self.remote_sha256 is not None else "git-blob-sha1"

    @property
    def source_digest(self) -> str:
        digest = self.remote_sha256 or self.git_blob_sha1
        if digest is None:
            raise ValueError(f"No remote digest is available for {self.path}")
        return digest


@dataclass(frozen=True, slots=True)
class VerificationResult:
    ok: bool
    reason: str
    sha256: str | None = None
    manifest_sha256_matches: bool | None = None


def _normalized_digest(value: str, length: int, label: str) -> str:
    digest = value.strip().lower()
    if len(digest) != length or _HEX_RE.fullmatch(digest) is None:
        raise ValueError(f"Invalid {label}: {value!r}")
    return digest


def file_spec_from_repo_file(repo_file: RepoFile) -> RemoteFileSpec:
    if repo_file.size < 0:
        raise ValueError(f"Invalid remote size for {repo_file.path}: {repo_file.size}")

    if repo_file.lfs is not None:
        if repo_file.lfs.size != repo_file.size:
            raise ValueError(
                f"Inconsistent LFS size for {repo_file.path}: "
                f"tree={repo_file.size}, lfs={repo_file.lfs.size}"
            )
        return RemoteFileSpec(
            path=repo_file.path,
            size=repo_file.size,
            remote_sha256=_normalized_digest(
                repo_file.lfs.sha256, 64, f"LFS SHA-256 for {repo_file.path}"
            ),
            git_blob_sha1=None,
        )

    return RemoteFileSpec(
        path=repo_file.path,
        size=repo_file.size,
        remote_sha256=None,
        git_blob_sha1=_normalized_digest(
            repo_file.blob_id, 40, f"Git blob SHA-1 for {repo_file.path}"
        ),
    )


def local_path_for(local_dir: Path, repo_path: str) -> Path:
    posix_path = PurePosixPath(repo_path)
    if posix_path.is_absolute() or not posix_path.parts:
        raise ValueError(f"Unsafe repository path: {repo_path!r}")
    if any(part in {"", ".", ".."} for part in posix_path.parts):
        raise ValueError(f"Unsafe repository path: {repo_path!r}")
    return local_dir.joinpath(*posix_path.parts)


def verify_local_file(
    path: Path,
    spec: RemoteFileSpec,
    *,
    manifest_sha256: str | None = None,
    show_progress: bool = True,
    stop_event: Event | None = None,
) -> VerificationResult:
    check_cancelled(stop_event)
    if not path.is_file():
        return VerificationResult(False, "file is missing or is not a regular file")

    try:
        actual_size = path.stat().st_size
    except OSError as exc:
        return VerificationResult(False, f"cannot stat file: {exc}")

    if actual_size != spec.size:
        return VerificationResult(
            False, f"size mismatch: local={actual_size}, remote={spec.size}"
        )

    sha256_hasher = hashlib.sha256()
    git_hasher = hashlib.sha1() if spec.git_blob_sha1 is not None else None
    if git_hasher is not None:
        git_hasher.update(f"blob {spec.size}\0".encode("ascii"))

    progress = tqdm(
        total=spec.size,
        desc=f"verify {spec.path}",
        unit="B",
        unit_scale=True,
        dynamic_ncols=True,
        leave=False,
        disable=not show_progress or spec.size < HASH_PROGRESS_THRESHOLD,
    )
    try:
        with path.open("rb") as file_handle:
            while True:
                check_cancelled(stop_event)
                chunk = file_handle.read(HASH_CHUNK_SIZE)
                if not chunk:
                    break
                sha256_hasher.update(chunk)
                if git_hasher is not None:
                    git_hasher.update(chunk)
                progress.update(len(chunk))
    except OSError as exc:
        return VerificationResult(False, f"cannot read file: {exc}")
    finally:
        progress.close()

    actual_sha256 = sha256_hasher.hexdigest()
    if spec.remote_sha256 is not None and actual_sha256 != spec.remote_sha256:
        return VerificationResult(
            False,
            f"SHA-256 mismatch: local={actual_sha256}, remote={spec.remote_sha256}",
            sha256=actual_sha256,
        )

    if git_hasher is not None:
        actual_git_sha1 = git_hasher.hexdigest()
        if actual_git_sha1 != spec.git_blob_sha1:
            return VerificationResult(
                False,
                f"Git blob hash mismatch: local={actual_git_sha1}, "
                f"remote={spec.git_blob_sha1}",
                sha256=actual_sha256,
            )

    manifest_matches = None
    if manifest_sha256 is not None:
        manifest_matches = actual_sha256 == manifest_sha256

    return VerificationResult(
        True,
        "content matches remote metadata",
        sha256=actual_sha256,
        manifest_sha256_matches=manifest_matches,
    )


class Sha256Manifest:
    def __init__(
        self,
        path: Path,
        repo_id: str,
        commit: str,
        files: dict[str, dict[str, Any]] | None = None,
        *,
        dirty: bool = False,
    ) -> None:
        self.path = path
        self.repo_id = repo_id
        self.commit = commit
        self.files = files or {}
        self.dirty = dirty
        self._lock = RLock()

    @classmethod
    def load(
        cls, local_dir: Path, repo_id: str, commit: str
    ) -> tuple["Sha256Manifest", str | None]:
        path = local_dir / MANIFEST_NAME
        if not path.is_file():
            return cls(path, repo_id, commit, dirty=True), None

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            return cls(path, repo_id, commit, dirty=True), f"invalid manifest ignored: {exc}"

        expected_header = (
            payload.get("schema_version") == MANIFEST_SCHEMA_VERSION
            and payload.get("endpoint") == MIRROR_ENDPOINT
            and payload.get("repo_id") == repo_id
            and payload.get("commit") == commit
            and isinstance(payload.get("files"), dict)
        )
        if not expected_header:
            return (
                cls(path, repo_id, commit, dirty=True),
                "manifest belongs to a different repository snapshot; rebuilding it",
            )

        files = {
            key: value
            for key, value in payload["files"].items()
            if isinstance(key, str) and isinstance(value, dict)
        }
        return cls(path, repo_id, commit, files), None

    def expected_sha256(self, spec: RemoteFileSpec) -> str | None:
        with self._lock:
            entry = self.files.get(spec.path)
            if entry is None:
                return None
            if entry.get("size") != spec.size:
                return None
            if entry.get("source_algorithm") != spec.source_algorithm:
                return None
            if entry.get("source_digest") != spec.source_digest:
                return None
            value = entry.get("sha256")
            if not isinstance(value, str):
                return None
            try:
                return _normalized_digest(value, 64, f"manifest SHA-256 for {spec.path}")
            except ValueError:
                return None

    def record(self, spec: RemoteFileSpec, sha256: str) -> None:
        entry = {
            "size": spec.size,
            "sha256": _normalized_digest(sha256, 64, f"SHA-256 for {spec.path}"),
            "source_algorithm": spec.source_algorithm,
            "source_digest": spec.source_digest,
        }
        with self._lock:
            if self.files.get(spec.path) != entry:
                self.files[spec.path] = entry
                self.dirty = True

    def retain(self, paths: set[str]) -> None:
        with self._lock:
            stale_paths = set(self.files) - paths
            if stale_paths:
                for path in stale_paths:
                    del self.files[path]
                self.dirty = True

    def save(self) -> None:
        with self._lock:
            if not self.dirty and self.path.is_file():
                return

            self.path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "schema_version": MANIFEST_SCHEMA_VERSION,
                "endpoint": MIRROR_ENDPOINT,
                "repo_id": self.repo_id,
                "commit": self.commit,
                "files": dict(sorted(self.files.items())),
            }
            temporary_path = self.path.with_name(f"{self.path.name}.{os.getpid()}.tmp")
            try:
                with temporary_path.open("w", encoding="utf-8", newline="\n") as file_handle:
                    json.dump(
                        payload,
                        file_handle,
                        ensure_ascii=True,
                        indent=2,
                        sort_keys=True,
                    )
                    file_handle.write("\n")
                    file_handle.flush()
                    os.fsync(file_handle.fileno())
                temporary_path.replace(self.path)
                self.dirty = False
            finally:
                if temporary_path.exists():
                    temporary_path.unlink()


def retry_call(
    label: str,
    action: Callable[[], T],
    attempts: int,
    *,
    sleeper: Callable[[float], None] = time.sleep,
) -> T:
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return action()
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            last_error = exc
            print(f"[WARN] {label} failed ({attempt}/{attempts}): {exc}")
            if attempt < attempts:
                delay = min(2 ** (attempt - 1), 30)
                print(f"[INFO] Retrying in {delay} seconds...")
                sleeper(delay)

    raise RuntimeError(f"{label} failed after {attempts} attempts") from last_error


def force_mirror_api_url(url: str) -> str:
    parsed = urlsplit(url)
    if parsed.path != "/api" and not parsed.path.startswith("/api/"):
        raise RuntimeError(f"Mirror pagination returned a non-API URL: {url}")
    mirror = urlsplit(MIRROR_ENDPOINT)
    return urlunsplit((mirror.scheme, mirror.netloc, parsed.path, parsed.query, ""))


def fetch_tree_page(
    url: str,
    params: dict[str, str] | None,
    headers: dict[str, str],
    attempts: int,
    *,
    sleeper: Callable[[float], None] = time.sleep,
) -> tuple[list[dict[str, Any]], str | None]:
    mirror_url = force_mirror_api_url(url)

    def request_page() -> tuple[list[dict[str, Any]], str | None]:
        timeout = httpx.Timeout(120.0, connect=30.0)
        with httpx.Client(timeout=timeout, follow_redirects=False) as client:
            response = client.get(mirror_url, params=params, headers=headers)
            if response.is_redirect:
                raise RuntimeError(
                    f"Mirror API redirected outside the requested page: "
                    f"{response.headers.get('location', '<missing location>')}"
                )
            hf_raise_for_status(response)
            payload = response.json()
            if not isinstance(payload, list) or not all(
                isinstance(item, dict) for item in payload
            ):
                raise RuntimeError(f"Unexpected tree response from {mirror_url}")
            next_page = response.links.get("next", {}).get("url")
            return payload, next_page

    return retry_call(
        f"load metadata page from {mirror_url}",
        request_page,
        attempts,
        sleeper=sleeper,
    )


def list_repo_tree_from_mirror(
    repo_id: str,
    commit: str,
    token: str | None,
    attempts: int,
    *,
    page_loader: Callable[
        [str, dict[str, str] | None, dict[str, str], int],
        tuple[list[dict[str, Any]], str | None],
    ] = fetch_tree_page,
) -> list[RepoFile]:
    encoded_repo_id = quote(repo_id, safe="/")
    encoded_commit = quote(commit, safe="")
    current_url = (
        f"{MIRROR_ENDPOINT}/api/datasets/{encoded_repo_id}/tree/{encoded_commit}"
    )
    params: dict[str, str] | None = {
        "recursive": "true",
        "expand": "false",
        "limit": "1000",
    }
    headers = build_hf_headers(
        token=token,
        headers={"Accept": "application/json", "Accept-Encoding": "identity"},
    )
    files: list[RepoFile] = []
    seen_pages: set[str] = set()
    page_number = 0

    while True:
        current_url = force_mirror_api_url(current_url)
        if current_url in seen_pages:
            raise RuntimeError(f"Pagination loop detected at {current_url}")
        seen_pages.add(current_url)

        page_number += 1
        payload, next_page = page_loader(current_url, params, headers, attempts)
        print(f"[INFO] Metadata page {page_number}: {len(payload)} entries")
        for item in payload:
            if item.get("type") == "file":
                files.append(RepoFile(**item))

        if next_page is None:
            return files
        current_url = force_mirror_api_url(next_page)
        params = None


def fetch_snapshot(
    api: HfApi,
    repo_id: str,
    revision: str,
    token: str | None,
    attempts: int,
) -> tuple[str, list[RemoteFileSpec]]:
    info = retry_call(
        "resolve dataset revision",
        lambda: api.dataset_info(repo_id=repo_id, revision=revision),
        attempts,
    )
    commit = info.sha
    if not isinstance(commit, str) or not commit:
        raise RuntimeError(f"Mirror did not return a commit for {repo_id}@{revision}")

    tree = list_repo_tree_from_mirror(
        repo_id=repo_id,
        commit=commit,
        token=token,
        attempts=attempts,
    )
    specs = sorted(
        (file_spec_from_repo_file(item) for item in tree if isinstance(item, RepoFile)),
        key=lambda item: item.path.casefold(),
    )
    if not specs:
        raise RuntimeError(f"No files found in {repo_id}@{commit}")
    if len({spec.path for spec in specs}) != len(specs):
        raise RuntimeError(f"Duplicate file paths returned for {repo_id}@{commit}")
    return commit, specs


def delete_local_file(path: Path) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if path.is_dir() and not path.is_symlink():
        raise IsADirectoryError(f"Refusing to delete directory at file path: {path}")
    try:
        path.unlink()
    except PermissionError:
        path.chmod(stat.S_IWRITE)
        path.unlink()


def mirror_download_url(repo_id: str, commit: str, repo_path: str) -> str:
    encoded_repo_id = quote(repo_id, safe="/")
    encoded_commit = quote(commit, safe="")
    encoded_path = quote(repo_path, safe="/")
    return (
        f"{MIRROR_ENDPOINT}/datasets/{encoded_repo_id}/resolve/"
        f"{encoded_commit}/{encoded_path}"
    )


def direct_incomplete_path(target: Path) -> Path:
    return target.with_name(f"{target.name}.incomplete")


def is_allowed_download_host(host: str | None) -> bool:
    if not host:
        return False
    normalized = host.casefold().rstrip(".")
    mirror_host = (urlsplit(MIRROR_ENDPOINT).hostname or "").casefold()
    return (
        normalized == mirror_host
        or normalized.endswith(f".{mirror_host}")
        or normalized == "xethub.hf.co"
        or normalized.endswith(".xethub.hf.co")
    )


def open_mirror_download_response(
    client: httpx.Client,
    url: str,
    headers: dict[str, str],
    *,
    max_redirects: int = MAX_DOWNLOAD_REDIRECTS,
) -> httpx.Response:
    current_url = url
    current_headers = dict(headers)
    previous_host: str | None = None

    for redirect_count in range(max_redirects + 1):
        parsed = urlsplit(current_url)
        if parsed.scheme != "https" or not is_allowed_download_host(parsed.hostname):
            raise RuntimeError(f"Refusing download host outside mirror storage: {parsed.hostname}")
        if previous_host is not None and parsed.hostname != previous_host:
            current_headers = {
                key: value
                for key, value in current_headers.items()
                if key.casefold() not in {"authorization", "cookie", "proxy-authorization"}
            }

        request = client.build_request("GET", current_url, headers=current_headers)
        response = client.send(request, stream=True, follow_redirects=False)
        if not response.is_redirect:
            return response

        location = response.headers.get("location")
        if not location:
            response.close()
            raise RuntimeError(f"Redirect from {parsed.hostname} has no Location header")
        if redirect_count >= max_redirects:
            response.close()
            raise RuntimeError(f"Too many download redirects for {url}")

        next_url = urljoin(str(response.url), location)
        next_parsed = urlsplit(next_url)
        if next_parsed.scheme != "https" or not is_allowed_download_host(
            next_parsed.hostname
        ):
            response.close()
            raise RuntimeError(
                "Mirror redirected to a disallowed host: "
                f"{next_parsed.hostname or '<missing>'}"
            )
        response.close()
        previous_host = parsed.hostname
        current_url = next_url

    raise RuntimeError(f"Too many download redirects for {url}")


def validate_range_response(
    response: httpx.Response, offset: int, expected_size: int
) -> None:
    if response.status_code == 206:
        content_range = response.headers.get("content-range", "")
        match = _CONTENT_RANGE_RE.fullmatch(content_range.strip())
        if match is None:
            raise RuntimeError(f"Invalid Content-Range: {content_range!r}")
        start, end, total = (int(value) for value in match.groups())
        if start != offset or end < start or total != expected_size or end >= total:
            raise RuntimeError(
                f"Unexpected Content-Range {content_range!r}; expected byte {offset} "
                f"of {expected_size}"
            )
        content_length = response.headers.get("content-length")
        if content_length and content_length.isdigit():
            expected_length = end - start + 1
            if int(content_length) != expected_length:
                raise RuntimeError(
                    f"Content-Length mismatch: header={content_length}, "
                    f"range={expected_length}"
                )
        return

    if response.status_code == 200 and offset == 0:
        content_length = response.headers.get("content-length")
        if content_length and content_length.isdigit() and int(content_length) != expected_size:
            raise RuntimeError(
                f"Content-Length mismatch: header={content_length}, remote={expected_size}"
            )
        return

    if response.status_code == 200:
        raise RuntimeError(
            f"Server ignored Range at byte {offset}; clean retry is required"
        )
    response.raise_for_status()
    raise RuntimeError(f"Unexpected download status HTTP {response.status_code}")


def direct_mirror_download(
    *,
    repo_id: str,
    commit: str,
    local_dir: Path,
    spec: RemoteFileSpec,
    token: str | None,
    force_download: bool,
    show_progress: bool,
    stop_event: Event | None,
    client: httpx.Client | None = None,
) -> Path:
    target = local_path_for(local_dir, spec.path)
    partial = direct_incomplete_path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if force_download:
        delete_local_file(partial)
    if partial.is_symlink() or (partial.exists() and not partial.is_file()):
        raise RuntimeError(f"Incomplete path is not a regular file: {partial}")
    if partial.is_file() and partial.stat().st_size > spec.size:
        console_print(f"[DELETE] Oversized incomplete file: {partial}")
        delete_local_file(partial)

    offset = partial.stat().st_size if partial.is_file() else 0
    if offset == spec.size:
        if not partial.exists():
            with partial.open("wb") as file_handle:
                file_handle.flush()
                os.fsync(file_handle.fileno())
        os.replace(partial, target)
        return target

    required_bytes = spec.size - offset
    free_bytes = shutil.disk_usage(target.parent).free
    if free_bytes < required_bytes + MIN_FREE_SPACE_RESERVE:
        raise InsufficientDiskSpace(
            f"insufficient disk space for {spec.path}: "
            f"need {(required_bytes + MIN_FREE_SPACE_RESERVE) / (1024**3):.2f} GiB "
            f"including reserve, free {free_bytes / (1024**3):.2f} GiB"
        )

    check_cancelled(stop_event)
    console_print(
        f"[DOWNLOAD] {spec.path} from byte {offset} via {MIRROR_ENDPOINT}"
    )
    request_headers = build_hf_headers(
        token=token,
        headers={
            "Accept-Encoding": "identity",
            "Range": f"bytes={offset}-",
        },
    )
    timeout = httpx.Timeout(120.0, connect=30.0, write=30.0, pool=30.0)
    owns_client = client is None
    active_client = client or httpx.Client(timeout=timeout, follow_redirects=False)
    progress = tqdm(
        total=spec.size,
        initial=offset,
        desc=f"download {spec.path}",
        unit="B",
        unit_scale=True,
        dynamic_ncols=True,
        leave=False,
        disable=not show_progress,
    )
    response: httpx.Response | None = None
    try:
        response = open_mirror_download_response(
            active_client,
            mirror_download_url(repo_id, commit, spec.path),
            request_headers,
        )
        validate_range_response(response, offset, spec.size)
        mode = "ab" if offset else "wb"
        with partial.open(mode) as file_handle:
            for chunk in response.iter_bytes(chunk_size=DOWNLOAD_CHUNK_SIZE):
                check_cancelled(stop_event)
                if not chunk:
                    continue
                file_handle.write(chunk)
                offset += len(chunk)
                progress.update(len(chunk))
                if offset > spec.size:
                    raise RuntimeError(
                        f"Download exceeded remote size for {spec.path}: "
                        f"local={offset}, remote={spec.size}"
                    )
            file_handle.flush()
            os.fsync(file_handle.fileno())
    finally:
        if response is not None:
            response.close()
        progress.close()
        if owns_client:
            active_client.close()

    actual_size = partial.stat().st_size if partial.is_file() else 0
    if actual_size != spec.size:
        raise RuntimeError(
            f"Incomplete HTTP response for {spec.path}: "
            f"local={actual_size}, remote={spec.size}"
        )
    os.replace(partial, target)
    return target


def needs_direct_mirror_fallback(exc: Exception) -> bool:
    message = str(exc).casefold()
    return _XET_SIZE_LIMIT_MARKER in message and "hf_xet" in message


def ensure_file(
    *,
    repo_id: str,
    commit: str,
    local_dir: Path,
    spec: RemoteFileSpec,
    manifest: Sha256Manifest,
    token: str | None,
    attempts: int,
    show_hash_progress: bool,
    download_impl: Callable[..., str] = hf_hub_download,
    sleeper: Callable[[float], None] = time.sleep,
    stop_event: Event | None = None,
) -> bool:
    check_cancelled(stop_event)
    target = local_path_for(local_dir, spec.path)
    expected_manifest_sha256 = manifest.expected_sha256(spec)
    target_present = target.exists() or target.is_symlink()
    force_download = False

    if target_present:
        console_print(f"[VERIFY] {spec.path}")
        result = verify_local_file(
            target,
            spec,
            manifest_sha256=expected_manifest_sha256,
            show_progress=show_hash_progress,
            stop_event=stop_event,
        )
        if result.ok:
            if result.manifest_sha256_matches is False:
                console_print(f"[WARN] Refreshing stale SHA-256 manifest entry: {spec.path}")
            if result.sha256 is None:
                raise RuntimeError(f"Verifier returned no SHA-256 for {spec.path}")
            manifest.record(spec, result.sha256)
            manifest.save()
            console_print(f"[OK] {spec.path} (SHA-256 {result.sha256})")
            return True

        console_print(f"[INVALID] {spec.path}: {result.reason}")
        console_print(f"[DELETE] {target}")
        delete_local_file(target)
        force_download = True
    else:
        console_print(
            f"[MISSING] {spec.path}; downloading or resuming an .incomplete file"
        )

    use_direct_http = False
    for attempt in range(1, attempts + 1):
        check_cancelled(stop_event)
        try:
            if use_direct_http:
                downloaded_path = direct_mirror_download(
                    repo_id=repo_id,
                    commit=commit,
                    local_dir=local_dir,
                    spec=spec,
                    token=token,
                    force_download=force_download,
                    show_progress=show_hash_progress,
                    stop_event=stop_event,
                )
            else:
                try:
                    downloaded_path = Path(
                        download_impl(
                            repo_id=repo_id,
                            filename=spec.path,
                            repo_type="dataset",
                            revision=commit,
                            local_dir=str(local_dir),
                            endpoint=MIRROR_ENDPOINT,
                            token=token,
                            etag_timeout=30,
                            force_download=force_download,
                            headers={"Accept-Encoding": "identity"},
                        )
                    )
                except Exception as exc:
                    if not needs_direct_mirror_fallback(exc):
                        raise
                    use_direct_http = True
                    console_print(
                        f"[INFO] {spec.path} requires Xet; switching to resumable "
                        f"HTTP Range through {MIRROR_ENDPOINT}"
                    )
                    downloaded_path = direct_mirror_download(
                        repo_id=repo_id,
                        commit=commit,
                        local_dir=local_dir,
                        spec=spec,
                        token=token,
                        force_download=force_download,
                        show_progress=show_hash_progress,
                        stop_event=stop_event,
                    )
            force_download = False
            check_cancelled(stop_event)
        except DownloadCancelled:
            raise
        except KeyboardInterrupt:
            console_print("\n[STOP] Download interrupted. Run the same script to resume.")
            raise
        except InsufficientDiskSpace as exc:
            console_print(f"[FAILED] {spec.path}: {exc}")
            return False
        except Exception as exc:
            # After the first clean restart, preserve the new .incomplete file so
            # the next attempt can resume instead of discarding received bytes.
            force_download = False
            console_print(
                f"[WARN] Download failed for {spec.path} ({attempt}/{attempts}): {exc}"
            )
            if attempt < attempts:
                delay = min(2 ** (attempt - 1), 30)
                console_print(f"[INFO] Retrying in {delay} seconds...")
                sleep_or_cancel(delay, stop_event, sleeper)
            continue

        console_print(f"[VERIFY] Downloaded file: {spec.path}")
        result = verify_local_file(
            downloaded_path,
            spec,
            manifest_sha256=expected_manifest_sha256,
            show_progress=show_hash_progress,
            stop_event=stop_event,
        )
        if result.ok:
            if result.sha256 is None:
                raise RuntimeError(f"Verifier returned no SHA-256 for {spec.path}")
            manifest.record(spec, result.sha256)
            manifest.save()
            console_print(f"[OK] {spec.path} (SHA-256 {result.sha256})")
            return True

        console_print(f"[INVALID] Downloaded file {spec.path}: {result.reason}")
        delete_local_file(downloaded_path)
        force_download = True
        if attempt < attempts:
            delay = min(2 ** (attempt - 1), 30)
            console_print(
                f"[INFO] Deleted invalid file; clean retry in {delay} seconds..."
            )
            sleep_or_cancel(delay, stop_event, sleeper)

    console_print(f"[FAILED] {spec.path} failed after {attempts} attempts")
    return False


def process_files(
    *,
    repo_id: str,
    commit: str,
    destination: Path,
    specs: list[RemoteFileSpec],
    manifest: Sha256Manifest,
    token: str | None,
    attempts: int,
    workers: int,
    show_hash_progress: bool,
) -> tuple[list[str], bool]:
    failures: list[str] = []
    stop_event = Event()

    def process_one(index: int, spec: RemoteFileSpec) -> bool:
        console_print(f"\n[FILE {index}/{len(specs)}] {spec.path}")
        return ensure_file(
            repo_id=repo_id,
            commit=commit,
            local_dir=destination,
            spec=spec,
            manifest=manifest,
            token=token,
            attempts=attempts,
            show_hash_progress=show_hash_progress,
            stop_event=stop_event if workers > 1 else None,
        )

    if workers == 1:
        for index, spec in enumerate(specs, start=1):
            try:
                success = process_one(index, spec)
            except KeyboardInterrupt:
                console_print("\n[STOP] Interrupted. Run the same command to resume.")
                return failures, True
            except Exception as exc:
                console_print(f"[FAILED] {spec.path}: {exc}")
                success = False
            if not success:
                failures.append(spec.path)
        return failures, False

    executor = ThreadPoolExecutor(
        max_workers=workers,
        thread_name_prefix="hf-mirror-download",
    )
    pending: dict[Future[bool], tuple[int, RemoteFileSpec]] = {}
    spec_iterator = iter(enumerate(specs, start=1))
    interrupted = False

    def submit_next() -> bool:
        try:
            index, spec = next(spec_iterator)
        except StopIteration:
            return False
        pending[executor.submit(process_one, index, spec)] = (index, spec)
        return True

    for _ in range(min(workers, len(specs))):
        submit_next()

    try:
        while pending:
            completed, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
            for future in completed:
                _, spec = pending.pop(future)
                try:
                    success = future.result()
                except DownloadCancelled:
                    success = False
                except Exception as exc:
                    console_print(f"[FAILED] {spec.path}: {exc}")
                    success = False
                if not success:
                    failures.append(spec.path)
                if not stop_event.is_set():
                    submit_next()
    except KeyboardInterrupt:
        interrupted = True
        stop_event.set()
        console_print(
            "\n[STOP] Cancelling queued files and waiting for active requests "
            "to reach a safe boundary..."
        )
        for future in pending:
            future.cancel()
    finally:
        executor.shutdown(wait=True, cancel_futures=True)

    return failures, interrupted


def default_local_dir(repo_id: str, data_root: Path | None = None) -> Path:
    root = data_root
    if root is None:
        configured_root = os.environ.get(DATA_ROOT_ENV)
        root = Path(configured_root) if configured_root else Path(__file__).resolve().parent
    return root.expanduser().resolve().joinpath("huggingface", *repo_id.split("/"))


def download_dataset(
    repo_id: str,
    *,
    revision: str = "main",
    local_dir: Path | None = None,
    data_root: Path | None = None,
    attempts: int = 3,
    workers: int = 2,
    show_hash_progress: bool = True,
) -> int:
    if attempts < 1:
        raise ValueError("attempts must be at least 1")
    if workers < 1:
        raise ValueError("workers must be at least 1")
    if local_dir is not None and data_root is not None:
        raise ValueError("local_dir and data_root cannot be used together")

    destination = (
        local_dir.expanduser().resolve()
        if local_dir is not None
        else default_local_dir(repo_id, data_root)
    )
    destination.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("HF_TOKEN") or None
    api = HfApi(endpoint=MIRROR_ENDPOINT, token=token)

    print(f"[INFO] Endpoint: {MIRROR_ENDPOINT}")
    print(f"[INFO] Dataset: {repo_id}@{revision}")
    print(f"[INFO] Destination: {destination}")
    print(f"[INFO] Concurrent file workers: {workers}")
    print("[INFO] Fetching remote metadata...")

    try:
        commit, specs = fetch_snapshot(api, repo_id, revision, token, attempts)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"[FATAL] Cannot load dataset metadata: {exc}")
        return 1

    print(f"[INFO] Resolved commit: {commit}")
    print(f"[INFO] Files to verify: {len(specs)}")

    manifest, warning = Sha256Manifest.load(destination, repo_id, commit)
    if warning is not None:
        print(f"[WARN] {warning}")
    manifest.retain({spec.path for spec in specs})
    manifest.save()

    failures, interrupted = process_files(
        repo_id=repo_id,
        commit=commit,
        destination=destination,
        specs=specs,
        manifest=manifest,
        token=token,
        attempts=attempts,
        workers=workers,
        show_hash_progress=show_hash_progress,
    )
    if interrupted:
        return 130

    manifest.save()
    if failures:
        print(f"\n[SUMMARY] {len(failures)} file(s) failed:")
        for path in failures:
            print(f"  - {path}")
        return 1

    print(f"\n[SUMMARY] All {len(specs)} files passed integrity verification.")
    return 0


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return parsed


def main(
    argv: Sequence[str] | None = None,
    *,
    default_repo: str | None = None,
) -> int:
    parser = argparse.ArgumentParser(
        description="Download and verify a Hugging Face dataset through hf-mirror.com."
    )
    parser.add_argument(
        "repo_id",
        nargs="?" if default_repo is not None else None,
        default=default_repo,
        help="Dataset repository ID, for example kensho/PubTables-v2.",
    )
    parser.add_argument("--revision", default="main", help="Branch, tag, or commit.")
    location_group = parser.add_mutually_exclusive_group()
    location_group.add_argument(
        "--local-dir",
        type=Path,
        help="Exact dataset destination directory.",
    )
    location_group.add_argument(
        "--data-root",
        type=Path,
        help=(
            "Data workspace root. The destination becomes "
            "<data-root>/huggingface/<repo_id>. Can also be set with "
            f"{DATA_ROOT_ENV}."
        ),
    )
    parser.add_argument(
        "--attempts",
        type=_positive_int,
        default=3,
        help="Maximum download/metadata attempts per operation (default: 3).",
    )
    parser.add_argument(
        "--workers",
        type=_positive_int,
        default=2,
        help="Concurrent file download workers (default: 2). Use 1 for serial mode.",
    )
    parser.add_argument(
        "--no-hash-progress",
        action="store_true",
        help="Hide progress bars while hashing large files.",
    )
    args = parser.parse_args(argv)
    if not args.repo_id:
        parser.error("repo_id is required")

    return download_dataset(
        args.repo_id,
        revision=args.revision,
        local_dir=args.local_dir,
        data_root=args.data_root,
        attempts=args.attempts,
        workers=args.workers,
        show_hash_progress=not args.no_hash_progress,
    )


if __name__ == "__main__":
    raise SystemExit(main())
