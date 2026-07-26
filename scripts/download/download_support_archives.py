#!/usr/bin/env python3
"""Download and safely validate support archives, updating the source CSV."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import shutil
import stat
import subprocess
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request

from download_support_documents import (
    REASON,
    STATUS,
    apply_result,
    build_cookie_opener,
    file_is_atrust_page,
    load_chrome_cookies,
    output_name,
    read_csv,
    safe_filename,
    timestamp,
    write_csv,
)


ARCHIVE_EXTENSIONS = frozenset({".zip", ".7z", ".rar", ".gz", ".tar", ".tgz"})
ARCHIVE_DOWNLOADED_STATUS = "已下载待验证"
ARCHIVE_VERIFIED_STATUS = "已下载已验证"
ARCHIVE_FAILED_VALIDATION_STATUS = "已下载验证失败"


def is_archive(row: dict[str, str]) -> bool:
    source = row.get("file_name", "") or row.get("file_path", "")
    return Path(source).suffix.lower() in ARCHIVE_EXTENSIONS


def extraction_name(row: dict[str, str]) -> str:
    source = row.get("file_name", "") or Path(row.get("file_path", "")).name or "archive"
    value = f"{row.get('id', 'unknown')}_{row.get('file_id', 'unknown')}_{Path(source).stem}"
    return safe_filename(value)


def download_archive(
    row: dict[str, str], archive_dir: Path, opener: Any, timeout: int, max_bytes: int
) -> tuple[str, str, Path | None]:
    target = archive_dir / output_name(row)
    if target.is_file() and target.stat().st_size > 0 and not file_is_atrust_page(target):
        return ARCHIVE_DOWNLOADED_STATUS, f"压缩包已存在（{target.stat().st_size} bytes），待验证", target

    url = row.get("download_url", "").strip()
    if not url.startswith(("http://", "https://")):
        return "不能下载", "缺少有效下载链接", None

    request = Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; support-archive-archiver/1.0)"},
    )
    temp_path: Path | None = None
    try:
        with opener.open(request, timeout=timeout) as response:
            final_host = (urlparse(response.geturl()).hostname or "").lower()
            if final_host != "support-admin.atrust.sangfor.com":
                return "不能下载", f"认证会话失效，最终跳转到 {final_host or '未知地址'}", None
            content_type = response.headers.get("Content-Type", "").split(";", 1)[0].lower()
            if content_type.startswith(("text/", "image/", "audio/", "video/")):
                return "不能下载", f"响应不是压缩包：{content_type}", None
            length = response.headers.get("Content-Length")
            if length and int(length) > max_bytes:
                return "不能下载", f"压缩包超过限制（>{max_bytes} bytes）", None

            fd, temp_name = tempfile.mkstemp(prefix=".partial-archive-", dir=archive_dir)
            temp_path = Path(temp_name)
            total = 0
            with os.fdopen(fd, "wb") as handle:
                while chunk := response.read(1024 * 1024):
                    total += len(chunk)
                    if total > max_bytes:
                        raise ValueError(f"压缩包超过限制（>{max_bytes} bytes）")
                    handle.write(chunk)
            if total == 0:
                return "不能下载", "响应内容为空", None
            if file_is_atrust_page(temp_path):
                return "不能下载", "响应为 HTML 登录/验证页面", None
            os.replace(temp_path, target)
            temp_path = None
            return ARCHIVE_DOWNLOADED_STATUS, f"压缩包下载成功（{total} bytes），待验证", target
    except HTTPError as error:
        return "不能下载", f"HTTP {error.code}", None
    except (URLError, TimeoutError) as error:
        detail = error.reason if isinstance(error, URLError) else "超时"
        return "不能下载", f"网络错误：{detail}", None
    except (OSError, ValueError) as error:
        return "不能下载", str(error), None
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def decode_output(value: bytes) -> str:
    for encoding in ("utf-8", "gb18030"):
        try:
            return value.decode(encoding)
        except UnicodeDecodeError:
            pass
    return value.decode("utf-8", errors="replace")


def list_archive(archive: Path, timeout: int) -> tuple[list[bytes], list[str]]:
    option_sets = [[]]
    if archive.suffix.lower() == ".zip":
        option_sets.append(["--options", "hdrcharset=GB18030"])

    errors: list[str] = []
    for options in option_sets:
        process = subprocess.run(
            ["bsdtar", "-tf", str(archive), *options],
            capture_output=True,
            timeout=timeout,
        )
        if process.returncode != 0:
            detail = decode_output(process.stderr).strip() or "bsdtar 无法识别归档格式"
            errors.append(detail[:500])
            continue

        entries = process.stdout.splitlines()
        for entry in entries:
            normalized = entry.replace(b"\\", b"/")
            parts = normalized.split(b"/")
            if normalized.startswith(b"/") or b".." in parts or re.match(br"^[A-Za-z]:", normalized):
                detail = decode_output(entry[:200])
                raise ValueError(f"归档包含不安全路径：{detail}")
        return entries, options

    raise ValueError(errors[-1])


def extract_raw_gzip(
    row: dict[str, str], archive: Path, staging: Path, max_bytes: int
) -> tuple[int, int]:
    source = row.get("file_name", "") or Path(row.get("file_path", "")).name or "archive.gz"
    output = staging / safe_filename(Path(source).stem or "decompressed")
    total = 0
    with gzip.open(archive, "rb") as compressed, output.open("wb") as extracted:
        while chunk := compressed.read(1024 * 1024):
            total += len(chunk)
            if total > max_bytes:
                raise ValueError(f"解压后内容超过限制（>{max_bytes} bytes）")
            extracted.write(chunk)
    return 1, total


def validate_extracted_tree(root: Path, max_bytes: int) -> tuple[int, int]:
    root_resolved = root.resolve()
    files = 0
    total = 0
    for current, directories, filenames in os.walk(root, followlinks=False):
        for name in directories + filenames:
            path = Path(current) / name
            mode = path.lstat().st_mode
            if stat.S_ISLNK(mode):
                resolved = path.resolve()
                if os.path.commonpath((str(root_resolved), str(resolved))) != str(root_resolved):
                    raise ValueError(f"解压结果包含越界符号链接：{path.name}")
            elif not (stat.S_ISDIR(mode) or stat.S_ISREG(mode)):
                raise ValueError(f"解压结果包含不支持的特殊文件：{path.name}")
            if stat.S_ISREG(mode):
                files += 1
                total += path.stat().st_size
                if total > max_bytes:
                    raise ValueError(f"解压后内容超过限制（>{max_bytes} bytes）")
    return files, total


def extract_archive(
    row: dict[str, str], archive: Path, extracted_root: Path,
    timeout: int, max_expanded_bytes: int, reserve_bytes: int,
) -> tuple[bool, str]:
    target = extracted_root / extraction_name(row)
    marker = target / ".complete.json"
    if marker.is_file():
        return True, "已存在完整解压结果，跳过"
    if target.exists():
        return False, "目标解压目录已存在但缺少完成标记，未覆盖"
    if shutil.disk_usage(extracted_root).free < reserve_bytes:
        return False, f"磁盘剩余空间低于保留值（{reserve_bytes} bytes）"

    staging = Path(
        tempfile.mkdtemp(
            prefix=f".partial-extract-{row.get('id', 'unknown')}-",
            dir=extracted_root,
        )
    )
    try:
        try:
            entries, reader_options = list_archive(archive, timeout)
        except ValueError:
            if archive.suffix.lower() != ".gz":
                raise
            files, total = extract_raw_gzip(row, archive, staging, max_expanded_bytes)
            entries = [Path(archive.name).stem.encode("utf-8", errors="replace")]
        else:
            command = [
                "bsdtar", "-xf", str(archive), "-C", str(staging),
                "--no-same-owner", "--no-same-permissions", *reader_options,
            ]
            process = subprocess.run(command, capture_output=True, timeout=timeout)
            if process.returncode != 0:
                detail = decode_output(process.stderr).strip() or "bsdtar 解压失败"
                return False, detail[:500]
            files, total = validate_extracted_tree(staging, max_expanded_bytes)
        marker_data = {
            "archive": archive.name,
            "archive_bytes": archive.stat().st_size,
            "listed_entries": len(entries),
            "extracted_files": files,
            "extracted_bytes": total,
            "completed_at": timestamp(),
        }
        (staging / ".complete.json").write_text(
            json.dumps(marker_data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(staging, target)
        return True, f"解压成功（{files} files，{total} bytes）"
    except (EOFError, OSError, subprocess.TimeoutExpired, ValueError) as error:
        return False, str(error)[:500]
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("/Volumes/SharedData1/supportData"))
    parser.add_argument("--chrome-cookies", type=Path)
    parser.add_argument("--download-workers", type=int, default=8)
    parser.add_argument("--extract-workers", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--extract-timeout", type=int, default=1800)
    parser.add_argument("--max-archive-gb", type=int, default=4)
    parser.add_argument("--max-expanded-gb", type=int, default=20)
    parser.add_argument("--reserve-free-gb", type=int, default=20)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--id", action="append", dest="ids", help="process only this CSV id; repeatable")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--download-only", action="store_true", help="download archives without extracting")
    mode.add_argument("--extract-only", action="store_true", help="validate and extract existing archives without downloading")
    args = parser.parse_args()

    if not args.extract_only and args.chrome_cookies is None:
        parser.error("--chrome-cookies is required unless --extract-only is used")

    rows, fields = read_csv(args.csv)
    selected_ids = set(args.ids or [])
    indexes = [
        index for index, row in enumerate(rows)
        if is_archive(row) and (not selected_ids or row.get("id") in selected_ids)
    ]
    archive_dir = args.output_dir / "archives"
    extracted_root = args.output_dir / "archives_extracted"
    archive_dir.mkdir(parents=True, exist_ok=True)
    extracted_root.mkdir(parents=True, exist_ok=True)
    for path in archive_dir.glob(".partial-archive-*"):
        path.unlink()
    for path in extracted_root.glob(".partial-extract-*"):
        if path.is_dir():
            shutil.rmtree(path)

    lock = threading.Lock()
    archive_paths: dict[int, Path] = {}
    if args.extract_only:
        for index in indexes:
            if (
                not selected_ids
                and rows[index].get(STATUS) == ARCHIVE_FAILED_VALIDATION_STATUS
            ):
                continue
            target = archive_dir / output_name(rows[index])
            if target.is_file() and target.stat().st_size > 0 and not file_is_atrust_page(target):
                archive_paths[index] = target
        print(
            f"仅验证模式：找到 {len(archive_paths)}/{len(indexes)} 个本地压缩包",
            flush=True,
        )
    else:
        cookies = load_chrome_cookies(args.chrome_cookies)
        thread_local = threading.local()

        def process_download(index: int) -> None:
            if not hasattr(thread_local, "opener"):
                thread_local.opener = build_cookie_opener(cookies)
            status, reason, path = download_archive(
                rows[index], archive_dir, thread_local.opener, args.timeout,
                args.max_archive_gb * 1024**3,
            )
            with lock:
                apply_result(rows[index], status, reason)
                if path is not None:
                    archive_paths[index] = path

        print(f"下载阶段：{len(indexes)} 个压缩包", flush=True)
        with ThreadPoolExecutor(max_workers=args.download_workers) as executor:
            futures = [executor.submit(process_download, index) for index in indexes]
            for completed, _future in enumerate(as_completed(futures), start=1):
                if completed % args.checkpoint_every == 0 or completed == len(indexes):
                    with lock:
                        write_csv(args.csv, rows, fields)
                    print(f"下载完成 {completed}/{len(indexes)}", flush=True)

        write_csv(args.csv, rows, fields)
        if args.download_only:
            downloaded = sum(
                1 for index in indexes
                if rows[index].get(STATUS) == ARCHIVE_DOWNLOADED_STATUS
            )
            failed = sum(1 for index in indexes if rows[index].get(STATUS) == "不能下载")
            print(f"仅下载完成：{ARCHIVE_DOWNLOADED_STATUS}={downloaded} 不能下载={failed}")
            return 0

    extractable = [index for index in indexes if index in archive_paths]
    print(f"验证解压阶段：{len(extractable)} 个本地压缩包", flush=True)

    def process_extract(index: int) -> tuple[int, bool, str]:
        try:
            success, detail = extract_archive(
                rows[index], archive_paths[index], extracted_root,
                args.extract_timeout, args.max_expanded_gb * 1024**3,
                args.reserve_free_gb * 1024**3,
            )
        except Exception as error:
            success = False
            detail = f"未预期的 {type(error).__name__}: {error}"[:500]
        return index, success, detail

    with ThreadPoolExecutor(max_workers=args.extract_workers) as executor:
        futures = [executor.submit(process_extract, index) for index in extractable]
        for completed, future in enumerate(as_completed(futures), start=1):
            index, success, detail = future.result()
            if success:
                apply_result(rows[index], ARCHIVE_VERIFIED_STATUS, f"压缩包验证并解压成功：{detail}")
            else:
                apply_result(rows[index], ARCHIVE_FAILED_VALIDATION_STATUS, f"压缩包验证或解压失败：{detail}")
            if completed % args.checkpoint_every == 0 or completed == len(extractable):
                write_csv(args.csv, rows, fields)
                print(f"验证完成 {completed}/{len(extractable)}", flush=True)

    counts = {
        ARCHIVE_VERIFIED_STATUS: 0,
        ARCHIVE_FAILED_VALIDATION_STATUS: 0,
        "不能下载": 0,
    }
    for index in indexes:
        status = rows[index].get(STATUS, "")
        if status in counts:
            counts[status] += 1
    print(" ".join(f"{key}={value}" for key, value in counts.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
