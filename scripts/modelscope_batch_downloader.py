#!/usr/bin/env python3
"""Download the OCR/document datasets selected in modelscope_ocr_catalog.py."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from modelscope_ocr_catalog import (
    GIANT_DOWNLOAD_KEYS,
    classify,
    dataset_key,
    load_or_collect,
    should_download,
)


DEFAULT_ROOT = Path(r"E:\data\doc\modelscope")
STATE_NAME = "modelscope-download-state.json"
REPORT_NAME = "modelscope-download-status.md"
CACHE_DIR_NAME = ".modelscope-cache"
DEFAULT_RESERVE_GIB = 8.0
DEFAULT_WORKERS = 4
STATUS_VERSION = 1
COMMERCIAL_OWNERS = {"DatatangBeijing", "market.aliyun"}
LEGACY_TARGETS = {"iic/Layout-Instruction-Data": Path("Layout-Instruction-Data")}
FINISHED_STATUSES = {"completed", "sample_downloaded", "existing_verified"}


def configure_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure:
            reconfigure(encoding="utf-8", errors="replace")


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def human_size(value: int | float) -> str:
    size = float(value)
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    for unit in units:
        if size < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(size):,} {unit}"
            return f"{size:,.2f} {unit}"
        size /= 1024
    raise AssertionError("unreachable")


def folder_stats(path: Path) -> tuple[int, int]:
    total = 0
    count = 0
    if not path.exists():
        return total, count
    for base, _, names in os.walk(path):
        for name in names:
            try:
                total += (Path(base) / name).stat().st_size
                count += 1
            except OSError:
                continue
    return total, count


def atomic_write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def load_state(path: Path, root: Path) -> dict[str, Any]:
    if path.exists():
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or not isinstance(value.get("datasets", {}), dict):
            raise ValueError(f"Invalid state file: {path}")
        return value
    return {
        "version": STATUS_VERSION,
        "root": str(root),
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "datasets": {},
    }


def safe_target(root: Path, record: dict[str, Any]) -> Path:
    owner = str(record["Owner"])
    name = str(record["Name"])
    for part in (owner, name):
        if not part or part in {".", ".."} or any(c in part for c in "\\/:"):
            raise ValueError(f"Unsafe dataset path component: {part!r}")
    key = dataset_key(record)
    target = root / LEGACY_TARGETS.get(key, Path(owner) / name)
    resolved_root = root.resolve(strict=False)
    resolved_target = target.resolve(strict=False)
    if resolved_root not in resolved_target.parents:
        raise ValueError(f"Dataset path escapes root: {target}")
    return target


def selected_records() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    payload = load_or_collect(False)
    selected = [
        record
        for record in payload["datasets"]
        if should_download(record, classify(record))
    ]
    excluded = [record for record in selected if dataset_key(record) in GIANT_DOWNLOAD_KEYS]
    targets = [record for record in selected if dataset_key(record) not in GIANT_DOWNLOAD_KEYS]
    return targets, excluded, payload


def scrub_error(exc: BaseException, token: str | None) -> str:
    message = f"{type(exc).__name__}: {exc}"
    if token:
        message = message.replace(token, "<redacted>")

    def strip_query(match: re.Match[str]) -> str:
        raw = match.group(0).rstrip(".,;)]}")
        suffix = match.group(0)[len(raw) :]
        parts = urlsplit(raw)
        return urlunsplit((parts.scheme, parts.netloc, parts.path, "", "")) + suffix

    message = re.sub(r"https?://[^\s]+", strip_query, message)
    return re.sub(r"\s+", " ", message).strip()[:2000]


def classify_failure(message: str, approval_mode: Any) -> str:
    lowered = message.lower()
    access_terms = (
        "401",
        "403",
        "access denied",
        "permission denied",
        "unauthorized",
        "forbidden",
        "approval",
        "authentication",
        "please login",
        "需申请",
        "无权限",
    )
    if approval_mode == 1 or any(term in lowered for term in access_terms):
        return "pending_access"
    if "no space left" in lowered or "disk full" in lowered:
        return "insufficient_space"
    return "failed"


def has_completion_metadata(target: Path) -> bool:
    return target.is_dir() and any(target.glob("*.msc"))


def file_sha256(path: Path, normalize_crlf: bool) -> str:
    digest = hashlib.sha256()
    if normalize_crlf:
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
        return digest.hexdigest()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def verify_existing_repository(
    key: str, target: Path, logger: "BatchLogger"
) -> dict[str, Any]:
    from modelscope.hub.api import HubApi
    from modelscope.hub.snapshot_download import fetch_repo_files

    api = HubApi()
    endpoint = api.get_endpoint_for_read(repo_id=key, repo_type="dataset")
    remote_files = fetch_repo_files(api, key, "master", endpoint)
    blobs = [item for item in remote_files or [] if item.get("Type") == "blob"]
    if not blobs:
        raise RuntimeError(f"No remote files returned for {key}")

    verified_bytes = 0
    for index, item in enumerate(blobs, 1):
        relative = Path(str(item["Path"]))
        local = target / relative
        if not local.is_file():
            raise RuntimeError(f"Missing existing file: {relative}")
        if item.get("IsLFS") and local.stat().st_size != int(item.get("Size") or 0):
            raise RuntimeError(
                f"Size mismatch for {relative}: local {local.stat().st_size}, remote {item.get('Size')}"
            )
        logger.info(
            "VERIFY %s [%d/%d] %s (%s)",
            key,
            index,
            len(blobs),
            relative,
            human_size(local.stat().st_size),
        )
        actual_hash = file_sha256(local, normalize_crlf=not bool(item.get("IsLFS")))
        expected_hash = str(item.get("Sha256") or "").lower()
        if not expected_hash or actual_hash.lower() != expected_hash:
            raise RuntimeError(
                f"SHA-256 mismatch for {relative}: local {actual_hash}, remote {expected_hash or 'missing'}"
            )
        verified_bytes += local.stat().st_size

    return {
        "remote_files": len(blobs),
        "verified_bytes": verified_bytes,
        "verified_at": now_iso(),
    }


def status_summary(state: dict[str, Any]) -> dict[str, int]:
    return dict(Counter(item.get("status", "unknown") for item in state["datasets"].values()))


def markdown_escape(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def write_report(
    path: Path,
    state: dict[str, Any],
    targets: list[dict[str, Any]],
    excluded: list[dict[str, Any]],
    free_bytes: int,
) -> None:
    status_labels = {
        "completed": "完成",
        "sample_downloaded": "商业样例已下载",
        "existing_verified": "已有仓库校验完成",
        "verification_failed": "已有仓库校验失败",
        "downloading": "下载中/可续传",
        "pending_access": "待登录/申请访问",
        "failed": "失败/待重试",
        "insufficient_space": "空间不足，未下载",
        "interrupted": "已中断/可续传",
    }
    rows: list[str] = []
    states = state.get("datasets", {})
    for index, record in enumerate(targets, 1):
        key = dataset_key(record)
        item = states.get(key, {})
        status = item.get("status", "queued")
        label = status_labels.get(status, "待下载" if status == "queued" else status)
        expected = int(record.get("StorageSize") or 0)
        local = int(item.get("local_bytes") or 0)
        note = item.get("error") or item.get("note") or ""
        link = f"https://www.modelscope.cn/datasets/{key}"
        rows.append(
            f"| {index} | [{markdown_escape(key)}]({link}) | "
            f"{markdown_escape(classify(record))} | {human_size(expected)} | "
            f"{human_size(local)} | {markdown_escape(label)} | {markdown_escape(note)} |"
        )

    summary = status_summary(state)
    summary_text = "；".join(f"{status_labels.get(k, k)} {v} 个" for k, v in sorted(summary.items()))
    excluded_text = "、".join(f"`{dataset_key(record)}`" for record in excluded)
    expected_total = sum(int(record.get("StorageSize") or 0) for record in targets)
    lines = [
        "# ModelScope OCR / 文档数据集下载状态",
        "",
        f"> 更新时间：{state.get('updated_at', now_iso())}",
        "",
        f"- 下载根目录：`{state.get('root', '')}`",
        f"- 本批目标：**{len(targets)} 个**，平台标称合计 **{human_size(expected_total)}**。",
        f"- 当前磁盘可用：**{human_size(free_bytes)}**。",
        f"- 状态汇总：{summary_text or '尚未开始'}。",
        f"- 下载日志：`{state.get('last_log') or '尚未创建'}`。",
        f"- 按用户要求暂不下载：{excluded_text}。",
        "- 数据堂和云市场仓库下载到的通常只是页面样例，不代表已取得商业全量数据。",
        "",
        "| # | 数据集 | 类型 | 平台大小 | 本地占用 | 状态 | 备注/错误 |",
        "|---:|---|---|---:|---:|---|---|",
        *rows,
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def persist(
    state_path: Path,
    report_path: Path,
    state: dict[str, Any],
    targets: list[dict[str, Any]],
    excluded: list[dict[str, Any]],
    root: Path,
) -> None:
    state["updated_at"] = now_iso()
    state["summary"] = status_summary(state)
    state["free_bytes"] = shutil.disk_usage(root).free
    atomic_write_json(state_path, state)
    write_report(report_path, state, targets, excluded, state["free_bytes"])


class RunLock:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.handle: Any = None

    def __enter__(self) -> "RunLock":
        import msvcrt

        self.handle = self.path.open("a+b")
        self.handle.seek(0)
        if self.handle.read(1) == b"":
            self.handle.seek(0)
            self.handle.write(b"0")
            self.handle.flush()
        self.handle.seek(0)
        try:
            msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            self.handle.close()
            raise RuntimeError(f"Another downloader is using {self.path}") from exc
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        import msvcrt

        if self.handle is not None:
            self.handle.seek(0)
            try:
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
            finally:
                self.handle.close()


class BatchLogger:
    """Small append-only logger unaffected by third-party logging configuration."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def _write(self, level: str, message: str, *args: Any) -> None:
        rendered = message % args if args else message
        line = f"{datetime.now().isoformat(timespec='seconds')} [{level}] {rendered}"
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def info(self, message: str, *args: Any) -> None:
        self._write("INFO", message, *args)

    def warning(self, message: str, *args: Any) -> None:
        self._write("WARNING", message, *args)

    def error(self, message: str, *args: Any) -> None:
        self._write("ERROR", message, *args)


def make_logger(root: Path) -> tuple[BatchLogger, Path]:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = root / f"modelscope-download-{timestamp}.log"
    log_path.touch(exist_ok=True)
    return BatchLogger(log_path), log_path


def dry_run(root: Path, targets: list[dict[str, Any]], excluded: list[dict[str, Any]], reserve: int) -> int:
    free = shutil.disk_usage(root).free
    expected = sum(int(record.get("StorageSize") or 0) for record in targets)
    existing = sum(folder_stats(safe_target(root, record))[0] for record in targets)
    remaining = max(0, expected - existing)
    print(f"下载根目录: {root}")
    print(f"目标仓库: {len(targets)} 个；排除仓库: {len(excluded)} 个")
    print(f"平台标称: {human_size(expected)}；目标目录已有: {human_size(existing)}")
    print(f"磁盘可用: {human_size(free)}；保留空间: {human_size(reserve)}")
    print(f"估算下载后可用: {human_size(max(0, free - remaining))}")
    print("排除: " + ", ".join(dataset_key(record) for record in excluded))
    if free < remaining + reserve:
        print("预检失败：按平台标称量和保留空间计算，磁盘不足。")
        return 2
    print("预检通过。")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--retries", type=int, default=2, help="Retries after the first attempt.")
    parser.add_argument("--retry-delay", type=float, default=5.0)
    parser.add_argument("--reserve-gib", type=float, default=DEFAULT_RESERVE_GIB)
    parser.add_argument("--retry-access", action="store_true")
    parser.add_argument("--verify-completed", action="store_true")
    args = parser.parse_args()
    if args.max_workers < 1 or args.retries < 0 or args.retry_delay < 0 or args.reserve_gib < 0:
        parser.error("workers must be positive; retries, delay and reserve cannot be negative")
    return args


def run(args: argparse.Namespace) -> int:
    configure_console()
    root = args.root.resolve(strict=False)
    targets, excluded, payload = selected_records()
    reserve = math.ceil(args.reserve_gib * 1024**3)

    if not root.exists():
        if args.dry_run:
            print(f"下载根目录不存在: {root}")
            return 2
        root.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        return dry_run(root, targets, excluded, reserve)

    state_path = root / STATE_NAME
    report_path = root / REPORT_NAME
    cache_dir = root / CACHE_DIR_NAME
    cache_dir.mkdir(exist_ok=True)
    logger, log_path = make_logger(root)
    token = os.getenv("MODELSCOPE_API_TOKEN") or os.getenv("MODELSCOPE_TOKEN")

    from modelscope.hub.snapshot_download import dataset_snapshot_download

    with RunLock(root / ".modelscope-download.lock"):
        state = load_state(state_path, root)
        state.update(
            {
                "version": STATUS_VERSION,
                "root": str(root),
                "source_url": payload.get("source_url"),
                "target_count": len(targets),
                "expected_bytes": sum(int(r.get("StorageSize") or 0) for r in targets),
                "excluded": [dataset_key(record) for record in excluded],
                "reserve_bytes": reserve,
                "last_log": str(log_path),
            }
        )
        persist(state_path, report_path, state, targets, excluded, root)

        logger.info(
            "Starting %d repositories (expected %s); free %s; reserve %s",
            len(targets),
            human_size(state["expected_bytes"]),
            human_size(shutil.disk_usage(root).free),
            human_size(reserve),
        )
        logger.info("Excluded: %s", ", ".join(state["excluded"]))

        for index, record in enumerate(targets, 1):
            key = dataset_key(record)
            target = safe_target(root, record)
            previous = state["datasets"].get(key, {})
            previous_status = previous.get("status")

            if (
                previous_status in FINISHED_STATUSES
                and not args.verify_completed
                and (
                    has_completion_metadata(target)
                    or (previous_status == "existing_verified" and target.is_dir())
                )
            ):
                local_bytes, local_files = folder_stats(target)
                previous.update(
                    {"local_bytes": local_bytes, "local_files": local_files, "last_checked": now_iso()}
                )
                state["datasets"][key] = previous
                logger.info("[%d/%d] SKIP completed %s", index, len(targets), key)
                persist(state_path, report_path, state, targets, excluded, root)
                continue

            if key in LEGACY_TARGETS and target.is_dir():
                logger.info(
                    "[%d/%d] VERIFY existing repository %s at %s",
                    index,
                    len(targets),
                    key,
                    target,
                )
                try:
                    verification = verify_existing_repository(key, target, logger)
                    local_bytes, local_files = folder_stats(target)
                    state["datasets"][key] = {
                        **previous,
                        "dataset": key,
                        "url": f"https://www.modelscope.cn/datasets/{key}",
                        "type": classify(record),
                        "status": "existing_verified",
                        "expected_bytes": int(record.get("StorageSize") or 0),
                        "local_bytes": local_bytes,
                        "local_files": local_files,
                        "path": str(target),
                        "note": "Existing legacy-path repository was SHA-256 verified and reused",
                        "error": "",
                        **verification,
                    }
                    logger.info(
                        "[%d/%d] VERIFIED %s (%s, %d files)",
                        index,
                        len(targets),
                        key,
                        human_size(local_bytes),
                        local_files,
                    )
                except Exception as exc:
                    local_bytes, local_files = folder_stats(target)
                    state["datasets"][key] = {
                        **previous,
                        "dataset": key,
                        "url": f"https://www.modelscope.cn/datasets/{key}",
                        "type": classify(record),
                        "status": "verification_failed",
                        "expected_bytes": int(record.get("StorageSize") or 0),
                        "local_bytes": local_bytes,
                        "local_files": local_files,
                        "path": str(target),
                        "updated_at": now_iso(),
                        "error": scrub_error(exc, token),
                    }
                    logger.error(
                        "[%d/%d] VERIFICATION FAILED %s: %s",
                        index,
                        len(targets),
                        key,
                        state["datasets"][key]["error"],
                    )
                persist(state_path, report_path, state, targets, excluded, root)
                continue

            if previous_status == "pending_access" and not args.retry_access:
                logger.info("[%d/%d] SKIP pending access %s", index, len(targets), key)
                continue

            expected = int(record.get("StorageSize") or 0)
            existing_bytes, existing_files = folder_stats(target)
            free = shutil.disk_usage(root).free
            remaining = max(0, expected - existing_bytes)
            if free < remaining + reserve:
                state["datasets"][key] = {
                    **previous,
                    "dataset": key,
                    "status": "insufficient_space",
                    "expected_bytes": expected,
                    "local_bytes": existing_bytes,
                    "local_files": existing_files,
                    "path": str(target),
                    "updated_at": now_iso(),
                    "error": (
                        f"Need about {human_size(remaining)} plus {human_size(reserve)} reserve; "
                        f"only {human_size(free)} free"
                    ),
                }
                logger.error("[%d/%d] STOP insufficient space before %s", index, len(targets), key)
                persist(state_path, report_path, state, targets, excluded, root)
                break

            target.parent.mkdir(parents=True, exist_ok=True)
            item = {
                **previous,
                "dataset": key,
                "url": f"https://www.modelscope.cn/datasets/{key}",
                "type": classify(record),
                "commercial_sample": record["Owner"] in COMMERCIAL_OWNERS,
                "approval_mode": record.get("ApprovalMode"),
                "expected_bytes": expected,
                "path": str(target),
                "status": "downloading",
                "started_at": now_iso(),
                "local_bytes": existing_bytes,
                "local_files": existing_files,
                "error": "",
            }
            state["datasets"][key] = item
            persist(state_path, report_path, state, targets, excluded, root)
            logger.info(
                "[%d/%d] DOWNLOAD %s (expected %s; existing %s)",
                index,
                len(targets),
                key,
                human_size(expected),
                human_size(existing_bytes),
            )

            for attempt in range(args.retries + 1):
                item["attempts"] = int(item.get("attempts") or 0) + 1
                item["last_attempt_at"] = now_iso()
                try:
                    result = dataset_snapshot_download(
                        key,
                        local_dir=str(target),
                        cache_dir=str(cache_dir),
                        max_workers=args.max_workers,
                        token=token,
                    )
                    if not result:
                        raise RuntimeError("ModelScope SDK returned no local directory")
                    local_bytes, local_files = folder_stats(target)
                    status = (
                        "sample_downloaded"
                        if record["Owner"] in COMMERCIAL_OWNERS
                        else "completed"
                    )
                    item.update(
                        {
                            "status": status,
                            "sdk_result": str(result),
                            "completed_at": now_iso(),
                            "updated_at": now_iso(),
                            "local_bytes": local_bytes,
                            "local_files": local_files,
                            "error": "",
                        }
                    )
                    logger.info(
                        "[%d/%d] DONE %s (%s, %d files)",
                        index,
                        len(targets),
                        key,
                        human_size(local_bytes),
                        local_files,
                    )
                    break
                except KeyboardInterrupt:
                    local_bytes, local_files = folder_stats(target)
                    item.update(
                        {
                            "status": "interrupted",
                            "updated_at": now_iso(),
                            "local_bytes": local_bytes,
                            "local_files": local_files,
                            "error": "Interrupted; rerun the same command to resume",
                        }
                    )
                    state["datasets"][key] = item
                    persist(state_path, report_path, state, targets, excluded, root)
                    logger.warning("Interrupted while downloading %s; progress is resumable", key)
                    return 130
                except Exception as exc:  # Continue the batch after recording a repository failure.
                    error = scrub_error(exc, token)
                    failure_status = classify_failure(error, record.get("ApprovalMode"))
                    local_bytes, local_files = folder_stats(target)
                    item.update(
                        {
                            "status": failure_status,
                            "updated_at": now_iso(),
                            "local_bytes": local_bytes,
                            "local_files": local_files,
                            "error": error,
                        }
                    )
                    logger.warning(
                        "[%d/%d] %s %s: %s",
                        index,
                        len(targets),
                        failure_status.upper(),
                        key,
                        error,
                    )
                    if failure_status != "failed" or attempt >= args.retries:
                        break
                    delay = args.retry_delay * (2**attempt)
                    logger.info("Retrying %s in %.1f seconds", key, delay)
                    time.sleep(delay)

            state["datasets"][key] = item
            persist(state_path, report_path, state, targets, excluded, root)

        persist(state_path, report_path, state, targets, excluded, root)
        logger.info("Finished batch. Summary: %s", state["summary"])
        logger.info("Status report: %s", report_path)
        logger.info("Free space: %s", human_size(state["free_bytes"]))
    return 0


def main() -> None:
    raise SystemExit(run(parse_args()))


if __name__ == "__main__":
    main()
