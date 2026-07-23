#!/usr/bin/env python3
"""Audit existing ModelScope dataset snapshots against remote manifests."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from modelscope_batch_downloader import (
    DEFAULT_ENV_PATH,
    file_sha256,
    human_size,
    load_dotenv,
    safe_target,
)
from modelscope_ocr_catalog import (
    classify,
    dataset_key,
    load_or_collect,
    should_download,
)


DEFAULT_ROOTS = (
    Path(r"E:\modelscope"),
    Path(r"F:\modelscope"),
    Path(r"G:\modelscope"),
)
DEFAULT_HASH_MAX_MIB = 64.0


@dataclass
class AuditResult:
    root: str
    dataset: str
    path: str
    status: str
    remote_files: int = 0
    metadata_files: int = 0
    expected_bytes: int = 0
    local_bytes: int = 0
    missing: list[str] | None = None
    size_mismatch: list[str] | None = None
    revision_mismatch: list[str] | None = None
    hash_mismatch: list[str] | None = None
    error: str = ""


def configure_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure:
            reconfigure(encoding="utf-8", errors="replace")


def normalized_path(value: Any) -> str:
    return str(value).replace("\\", "/").lstrip("/")


def revisions_match(cached: str, remote: str) -> bool:
    return bool(cached and remote) and (
        cached.startswith(remote) or remote.startswith(cached)
    )


def load_completion_index(target: Path) -> dict[str, str]:
    marker = target / ".msc"
    if not marker.is_file():
        return {}
    with marker.open("rb") as handle:
        entries = pickle.load(handle)
    if not isinstance(entries, list):
        raise ValueError(f"Unexpected metadata payload in {marker}")
    result: dict[str, str] = {}
    for entry in entries:
        if not isinstance(entry, dict) or "Path" not in entry:
            raise ValueError(f"Unexpected metadata entry in {marker}")
        result[normalized_path(entry["Path"])] = str(entry.get("Revision") or "")
    return result


def fetch_manifest(key: str, token: str | None) -> list[dict[str, Any]]:
    from modelscope.hub.api import HubApi
    from modelscope.hub.snapshot_download import fetch_repo_files

    api = HubApi(token=token)
    endpoint = api.get_endpoint_for_read(repo_id=key, repo_type="dataset")
    files = fetch_repo_files(api, key, "master", endpoint)
    blobs = [item for item in files or [] if item.get("Type") == "blob"]
    if not blobs:
        raise RuntimeError("remote API returned no files")
    return blobs


def candidate_paths(root: Path, record: dict[str, Any]) -> list[Path]:
    conventional = root / str(record["Owner"]) / str(record["Name"])
    candidates = [conventional, safe_target(root, record)]
    result: list[Path] = []
    for candidate in candidates:
        resolved = candidate.resolve(strict=False)
        if resolved not in result and resolved.is_dir():
            result.append(resolved)
    return result


def audit_repository(
    root: Path,
    key: str,
    target: Path,
    manifest: list[dict[str, Any]] | Exception,
    hash_max_bytes: int,
) -> AuditResult:
    base = AuditResult(root=str(root), dataset=key, path=str(target), status="error")
    if isinstance(manifest, Exception):
        base.error = f"{type(manifest).__name__}: {manifest}"
        return base

    try:
        metadata = load_completion_index(target)
    except Exception as exc:
        base.error = f"metadata {type(exc).__name__}: {exc}"
        return base

    missing: list[str] = []
    size_mismatch: list[str] = []
    revision_mismatch: list[str] = []
    hash_mismatch: list[str] = []
    expected_bytes = 0
    local_bytes = 0

    for item in manifest:
        relative = normalized_path(item["Path"])
        local = target.joinpath(*relative.split("/"))
        expected_size = int(item.get("Size") or 0)
        expected_bytes += expected_size
        if not local.is_file():
            missing.append(relative)
            continue

        local_size = local.stat().st_size
        local_bytes += local_size
        expected_hash = str(item.get("Sha256") or "").lower()
        hash_checked = bool(expected_hash) and (
            not bool(item.get("IsLFS")) or local_size <= hash_max_bytes
        )
        hash_matches = True
        if hash_checked:
            actual_hash = file_sha256(local, normalize_crlf=False).lower()
            hash_matches = actual_hash == expected_hash
            if not hash_matches and not bool(item.get("IsLFS")):
                normalized_hash = file_sha256(local, normalize_crlf=True).lower()
                hash_matches = normalized_hash == expected_hash
            if not hash_matches:
                hash_mismatch.append(relative)

        if local_size != expected_size and not (
            not bool(item.get("IsLFS")) and hash_checked and hash_matches
        ):
            size_mismatch.append(
                f"{relative} ({local_size} != {expected_size})"
            )

        cached_revision = metadata.get(relative, "")
        remote_revision = str(item.get("Revision") or "")
        if not revisions_match(cached_revision, remote_revision):
            revision_mismatch.append(relative)

    complete = not any(
        (missing, size_mismatch, revision_mismatch, hash_mismatch)
    )
    return AuditResult(
        root=str(root),
        dataset=key,
        path=str(target),
        status="complete" if complete else "incomplete",
        remote_files=len(manifest),
        metadata_files=len(metadata),
        expected_bytes=expected_bytes,
        local_bytes=local_bytes,
        missing=missing,
        size_mismatch=size_mismatch,
        revision_mismatch=revision_mismatch,
        hash_mismatch=hash_mismatch,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, action="append", dest="roots")
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_PATH)
    parser.add_argument("--dataset-key", action="append", dest="dataset_keys")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--hash-max-mib", type=float, default=DEFAULT_HASH_MAX_MIB)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()
    if args.max_workers < 1 or args.hash_max_mib < 0:
        parser.error("workers must be positive and hash limit cannot be negative")
    return args


def run(args: argparse.Namespace) -> int:
    configure_console()
    load_dotenv(args.env_file)
    token = os.getenv("MODELSCOPE_API_TOKEN") or os.getenv("MODELSCOPE_TOKEN")
    roots = [path.resolve(strict=False) for path in (args.roots or DEFAULT_ROOTS)]
    requested = set(args.dataset_keys or [])
    payload = load_or_collect(False)
    records = [
        record
        for record in payload["datasets"]
        if should_download(record, classify(record))
        and (not requested or dataset_key(record) in requested)
    ]
    found: list[tuple[Path, str, Path]] = []
    for root in roots:
        for record in records:
            key = dataset_key(record)
            found.extend((root, key, path) for path in candidate_paths(root, record))

    manifests: dict[str, list[dict[str, Any]] | Exception] = {}
    keys = sorted({key for _, key, _ in found})
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(fetch_manifest, key, token): key for key in keys}
        for future in as_completed(futures):
            key = futures[future]
            try:
                manifests[key] = future.result()
            except Exception as exc:
                manifests[key] = exc

    hash_max_bytes = int(args.hash_max_mib * 1024**2)
    results = [
        audit_repository(root, key, path, manifests[key], hash_max_bytes)
        for root, key, path in found
    ]
    results.sort(key=lambda item: (item.root.lower(), item.dataset.lower(), item.path.lower()))
    if args.as_json:
        print(json.dumps([asdict(item) for item in results], ensure_ascii=False, indent=2))
    else:
        for item in results:
            issue_count = sum(
                len(values or [])
                for values in (
                    item.missing,
                    item.size_mismatch,
                    item.revision_mismatch,
                    item.hash_mismatch,
                )
            )
            print(
                f"{item.status.upper():10} {item.dataset} @ {item.root} "
                f"files={item.remote_files}/{item.metadata_files} "
                f"bytes={human_size(item.local_bytes)}/{human_size(item.expected_bytes)} "
                f"issues={issue_count}{' error=' + item.error if item.error else ''}"
            )
            if item.status == "incomplete":
                for label, values in (
                    ("missing", item.missing),
                    ("size", item.size_mismatch),
                    ("revision", item.revision_mismatch),
                    ("hash", item.hash_mismatch),
                ):
                    if values:
                        sample = ", ".join(values[:5])
                        suffix = " ..." if len(values) > 5 else ""
                        print(f"  {label}: {sample}{suffix}")
        complete = sum(item.status == "complete" for item in results)
        incomplete = sum(item.status == "incomplete" for item in results)
        errors = sum(item.status == "error" for item in results)
        print(
            f"SUMMARY roots={len(roots)} existing={len(results)} "
            f"complete={complete} incomplete={incomplete} errors={errors}"
        )
    return 0 if all(item.status == "complete" for item in results) else 2


def main() -> None:
    raise SystemExit(run(parse_args()))


if __name__ == "__main__":
    main()
