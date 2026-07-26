#!/usr/bin/env python3
"""Download document files from the support-file CSV and record each outcome.

The CSV is updated in place, adding download_status, download_reason, and
download_checked_at. Re-running the script skips files already on disk.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import http.cookiejar
import os
import re
import sqlite3
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from email.message import Message
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.error import HTTPError, URLError
from urllib.request import HTTPCookieProcessor, Request, build_opener


STATUS = "download_status"
REASON = "download_reason"
OUTPUT = "download_path"
CHECKED = "download_checked_at"
RESULT_COLUMNS = (STATUS, REASON, CHECKED)
CHROME_EPOCH_OFFSET = 11_644_473_600

DOCUMENT_EXTENSIONS = frozenset(
    {
        ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
        ".html", ".htm", ".txt", ".rtf", ".odt", ".ods", ".odp",
        ".csv", ".tsv", ".xml", ".xhtml", ".epub", ".md", ".mht",
        ".mhtml", ".eml", ".msg", ".wps", ".wpt", ".vsd", ".vsdx",
        ".pages", ".numbers", ".key", ".djvu", ".tex", ".log", ".chm",
    }
)

NON_DOCUMENT_PREFIXES = ("image/", "audio/", "video/")
NON_DOCUMENT_MIMES = frozenset(
    {
        "application/zip", "application/x-7z-compressed",
        "application/x-rar-compressed", "application/x-tar",
        "application/gzip", "application/x-gzip", "application/x-executable",
        "application/x-msdownload", "application/vnd.android.package-archive",
        "application/json", "application/javascript", "application/wasm",
    }
)


def mime_is_document(content_type: str) -> bool:
    mime = content_type.split(";", 1)[0].strip().lower()
    if not mime or mime in {"application/octet-stream", "binary/octet-stream"}:
        return False
    if mime.startswith(NON_DOCUMENT_PREFIXES) or mime in NON_DOCUMENT_MIMES:
        return False
    if mime.startswith("text/"):
        return True
    return any(
        token in mime
        for token in (
            "pdf", "msword", "ms-excel", "ms-powerpoint", "officedocument",
            "opendocument", "rtf", "epub", "wordperfect", "visio",
            "onenote", "postscript", "xml", "spreadsheet", "presentation",
        )
    )


def row_is_document(row: dict[str, str]) -> bool:
    filename = row.get("file_name", "")
    path = row.get("file_path", "")
    suffix = Path(filename or path).suffix.lower()
    return suffix in DOCUMENT_EXTENSIONS or mime_is_document(row.get("file_type", ""))


def safe_filename(value: str) -> str:
    value = value.replace("/", "_").replace("\\", "_").replace("\x00", "")
    value = re.sub(r"[<>:\\|?*]", "_", value).strip(". ")
    return value[:180] or "unnamed-document"


def filename_from_disposition(header: str) -> str:
    if not header:
        return ""
    message = Message()
    message["content-disposition"] = header
    filename = message.get_filename()
    return filename or ""


def output_name(row: dict[str, str], disposition: str = "") -> str:
    original = row.get("file_name", "") or filename_from_disposition(disposition)
    if not original:
        original = Path(row.get("file_path", "")).name or "document"
    return safe_filename(f"{row.get('id', 'unknown')}_{row.get('file_id', 'unknown')}_{original}")


def row_expects_html(row: dict[str, str]) -> bool:
    suffix = Path(row.get("file_name", "") or row.get("file_path", "")).suffix.lower()
    mime = row.get("file_type", "").split(";", 1)[0].lower()
    return suffix in {".html", ".htm", ".xhtml", ".mht", ".mhtml"} or mime in {
        "text/html", "application/xhtml+xml"
    }


def file_looks_like_html(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            prefix = handle.read(8192).lstrip().lower()
    except OSError:
        return False
    return prefix.startswith((b"<!doctype html", b"<html")) or b"/portal/shortcut" in prefix


def file_is_atrust_page(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            prefix = handle.read(16384).lower()
    except OSError:
        return False
    return (
        b"/portal/#/page_app_handler" in prefix
        or b"shortcut_main.js" in prefix
        or (b"shortcutpreloadtime" in prefix and b"shortcuterrreloadtimes" in prefix)
    )


def cleanup_interrupted_outputs(output_dir: Path) -> tuple[int, int, int]:
    invalid = 0
    partial = 0
    metadata = 0
    for path in output_dir.iterdir():
        if not path.is_file():
            continue
        if path.name.startswith("._"):
            path.unlink()
            metadata += 1
        elif path.name.startswith(".partial-"):
            path.unlink()
            partial += 1
        elif file_is_atrust_page(path):
            path.unlink()
            invalid += 1
    return invalid, partial, metadata


def decrypt_chrome_cookie(encrypted: bytes, host: str, key: bytes) -> str:
    if not encrypted.startswith((b"v10", b"v11")):
        return encrypted.decode("utf-8")
    process = subprocess.run(
        [
            "openssl", "enc", "-d", "-aes-128-cbc", "-nopad",
            "-K", key.hex(), "-iv", (b" " * 16).hex(),
        ],
        input=encrypted[3:], capture_output=True, check=True,
    )
    decrypted = process.stdout
    if not decrypted:
        return ""
    padding = decrypted[-1]
    if 1 <= padding <= 16 and decrypted.endswith(bytes([padding]) * padding):
        decrypted = decrypted[:-padding]
    host_digest = hashlib.sha256(host.encode()).digest()
    if decrypted.startswith(host_digest):
        decrypted = decrypted[len(host_digest):]
    return decrypted.decode("utf-8")


def load_chrome_cookies(database: Path) -> list[http.cookiejar.Cookie]:
    password = subprocess.run(
        ["security", "find-generic-password", "-w", "-s", "Chrome Safe Storage"],
        capture_output=True, check=True,
    ).stdout.rstrip(b"\n")
    key = hashlib.pbkdf2_hmac("sha1", password, b"saltysalt", 1003, dklen=16)
    query = """
        SELECT host_key, name, path, expires_utc, is_secure, is_httponly,
               encrypted_value, value
        FROM cookies
        WHERE host_key = 'support-admin.atrust.sangfor.com'
           OR host_key = 'sdpc.sangfor.com'
           OR host_key = '.atrust.sangfor.com'
    """
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    try:
        records = connection.execute(query).fetchall()
    finally:
        connection.close()

    cookies: list[http.cookiejar.Cookie] = []
    for host, name, path, expires_utc, secure, http_only, encrypted, plain in records:
        value = plain or decrypt_chrome_cookie(encrypted, host, key)
        if not value:
            continue
        expires = None
        if expires_utc and expires_utc > CHROME_EPOCH_OFFSET * 1_000_000:
            expires = int(expires_utc / 1_000_000 - CHROME_EPOCH_OFFSET)
        cookies.append(
            http.cookiejar.Cookie(
                version=0, name=name, value=value, port=None, port_specified=False,
                domain=host, domain_specified=True, domain_initial_dot=host.startswith("."),
                path=path or "/", path_specified=True, secure=bool(secure), expires=expires,
                discard=expires is None, comment=None, comment_url=None,
                rest={"HttpOnly": None} if http_only else {}, rfc2109=False,
            )
        )
    if not cookies:
        raise RuntimeError("Chrome 中未找到 Sangfor 会话 Cookie")
    return cookies


def build_cookie_opener(cookies: list[http.cookiejar.Cookie]):
    jar = http.cookiejar.CookieJar()
    for cookie in cookies:
        jar.set_cookie(cookie)
    return build_opener(HTTPCookieProcessor(jar))


def timestamp() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def apply_result(row: dict[str, str], status: str, reason: str, path: str = "") -> None:
    row[STATUS] = status
    row[REASON] = reason
    row[CHECKED] = timestamp()


def download_one(
    row: dict[str, str], output_dir: Path, timeout: int, max_bytes: int, dry_run: bool,
    opener: Any,
) -> tuple[str, str, str]:
    if not row_is_document(row):
        return "不下载", "非文档文件类型", ""
    expected_target = output_dir / output_name(row)
    if expected_target.is_file() and expected_target.stat().st_size > 0:
        if not file_is_atrust_page(expected_target):
            return "已下载", "新NTFS卷已有文件，跳过请求", str(expected_target)
    url = row.get("download_url", "").strip()
    if not url.startswith(("http://", "https://")):
        return "不能下载", "缺少有效下载链接", ""
    if dry_run:
        return "不下载", "dry-run：符合文档下载条件", ""

    request = Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible; support-document-archiver/1.0)"})
    temp_path: Path | None = None
    try:
        with opener.open(request, timeout=timeout) as response:
            content_type = response.headers.get("Content-Type", "")
            disposition = response.headers.get("Content-Disposition", "")
            final_host = (urlparse(response.geturl()).hostname or "").lower()
            if final_host != "support-admin.atrust.sangfor.com":
                return "不能下载", f"认证会话失效，最终跳转到 {final_host or '未知地址'}", ""
            if content_type and not mime_is_document(content_type) and content_type.split(";", 1)[0].lower() not in {
                "application/octet-stream", "binary/octet-stream"
            }:
                return "不能下载", f"响应不是文档：{content_type}", ""
            target = output_dir / output_name(row, disposition)
            if target.exists() and target.stat().st_size > 0:
                if row_expects_html(row) or not file_looks_like_html(target):
                    return "已下载", "目标文件已存在，跳过", str(target)
                target.unlink()
            declared_length = response.headers.get("Content-Length")
            if declared_length and int(declared_length) > max_bytes:
                return "不能下载", f"响应文件超过限制（>{max_bytes} bytes）", ""
            fd, temp_name = tempfile.mkstemp(prefix=".partial-", dir=output_dir)
            temp_path = Path(temp_name)
            total = 0
            with os.fdopen(fd, "wb") as handle:
                while chunk := response.read(1024 * 1024):
                    total += len(chunk)
                    if total > max_bytes:
                        raise ValueError(f"响应文件超过限制（>{max_bytes} bytes）")
                    handle.write(chunk)
            if total == 0:
                return "不能下载", "响应内容为空", ""
            if not row_expects_html(row) and file_looks_like_html(temp_path):
                return "不能下载", "响应为 HTML 登录/验证页面", ""
            os.replace(temp_path, target)
            temp_path = None
            return "已下载", f"下载成功（{total} bytes）", str(target)
    except HTTPError as error:
        return "不能下载", f"HTTP {error.code}", ""
    except (URLError, TimeoutError) as error:
        return "不能下载", f"网络错误：{error.reason if isinstance(error, URLError) else '超时'}", ""
    except (OSError, ValueError) as error:
        return "不能下载", str(error), ""
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("CSV 缺少表头")
        fields = list(reader.fieldnames)
        return list(reader), fields


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    final_fields = [field for field in fields if field != OUTPUT]
    final_fields += [column for column in RESULT_COLUMNS if column not in final_fields]
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=final_fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temp_name, path)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise


def reconcile_new_volume(rows: list[dict[str, str]], output_dir: Path) -> dict[str, int]:
    """Make CSV status reflect only files physically present on the current volume."""
    counters: dict[str, int] = {"已下载": 0, "待下载": 0, "不下载": 0}
    existing_by_prefix: dict[str, Path] = {}
    for path in output_dir.iterdir():
        if not path.is_file() or path.name.startswith(("._", ".partial-")):
            continue
        match = re.match(r"^(\d+)_([0-9a-fA-F]+)_", path.name)
        if match and path.stat().st_size > 0:
            existing_by_prefix.setdefault(match.group(0), path)
    for row in rows:
        if not row_is_document(row):
            apply_result(row, "不下载", "非文档文件类型", "")
        else:
            prefix = f"{row.get('id', 'unknown')}_{row.get('file_id', 'unknown')}_"
            target = output_dir / output_name(row)
            if not target.is_file():
                target = existing_by_prefix.get(prefix, target)
            if target.is_file() and target.stat().st_size > 0 and not file_is_atrust_page(target):
                apply_result(row, "已下载", "新NTFS卷文件已验证", str(target))
            else:
                apply_result(row, "待下载", "新NTFS卷无对应文件，待下载", "")
        counters[row[STATUS]] += 1
    return counters


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="source CSV, updated in place")
    parser.add_argument("--output-dir", type=Path, default=Path("/Volumes/SharedData1/supportData"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--max-file-mb", type=int, default=250)
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--chrome-cookies",
        type=Path,
        help="Chrome Cookies database; loads only Sangfor session cookies",
    )
    parser.add_argument(
        "--reconcile-new-volume",
        action="store_true",
        help="reset CSV results to match only files physically present in output-dir",
    )
    parser.add_argument(
        "--reset-csv-status",
        action="store_true",
        help="clear all download result fields without changing source columns",
    )
    parser.add_argument(
        "--remove-download-path-column",
        action="store_true",
        help="remove the legacy download_path column while preserving other results",
    )
    args = parser.parse_args()
    if args.workers < 1 or args.timeout < 1 or args.max_file_mb < 1 or args.checkpoint_every < 1:
        parser.error("workers、timeout、max-file-mb 和 checkpoint-every 必须为正数")

    rows, fields = read_csv(args.csv)
    if args.remove_download_path_column:
        write_csv(args.csv, rows, fields)
        print(f"已删除 {OUTPUT} 列，保留行数={len(rows)}")
        return 0
    if args.reset_csv_status:
        for row in rows:
            for column in RESULT_COLUMNS:
                row[column] = ""
        write_csv(args.csv, rows, fields)
        print(f"已清空下载标记={len(rows)}")
        return 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.reconcile_new_volume:
        counters = reconcile_new_volume(rows, args.output_dir)
        write_csv(args.csv, rows, fields)
        print(" ".join(f"{key}={value}" for key, value in counters.items()))
        return 0
    invalid_count, partial_count, metadata_count = cleanup_interrupted_outputs(args.output_dir)
    if invalid_count or partial_count or metadata_count:
        print(
            f"清理无效验证页={invalid_count} 未完成临时文件={partial_count} "
            f"macOS元数据={metadata_count}",
            file=sys.stderr, flush=True,
        )
    max_bytes = args.max_file_mb * 1024 * 1024
    cookies = load_chrome_cookies(args.chrome_cookies) if args.chrome_cookies else []
    lock = threading.Lock()
    thread_local = threading.local()
    counters: dict[str, int] = {"已下载": 0, "不能下载": 0, "不下载": 0}

    def process(index: int) -> None:
        if not hasattr(thread_local, "opener"):
            thread_local.opener = build_cookie_opener(cookies)
        status, reason, path = download_one(
            rows[index], args.output_dir, args.timeout, max_bytes, args.dry_run,
            thread_local.opener,
        )
        with lock:
            apply_result(rows[index], status, reason, path)
            counters[status] += 1

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(process, index) for index in range(len(rows))]
        for completed, _future in enumerate(as_completed(futures), start=1):
            if completed % args.checkpoint_every == 0 or completed == len(rows):
                with lock:
                    write_csv(args.csv, rows, fields)
                print(f"完成 {completed}/{len(rows)}", file=sys.stderr, flush=True)

    write_csv(args.csv, rows, fields)
    print(" ".join(f"{key}={value}" for key, value in counters.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
