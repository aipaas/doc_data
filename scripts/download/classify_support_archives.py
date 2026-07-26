#!/usr/bin/env python3
"""Classify extracted support archives by whether they contain documents."""

from __future__ import annotations

import argparse
import os
from collections import Counter
from pathlib import Path

from download_support_archives import (
    ARCHIVE_EXTENSIONS,
    ARCHIVE_FAILED_VALIDATION_STATUS,
    ARCHIVE_VERIFIED_STATUS,
    extraction_name,
)
from download_support_documents import (
    DOCUMENT_EXTENSIONS,
    REASON,
    STATUS,
    apply_result,
    read_csv,
    write_csv,
)


UNAVAILABLE_STATUS = "已下载不可用"
ADDITIONAL_DOCUMENT_EXTENSIONS = frozenset(
    {
        ".ofd", ".xps", ".oxps", ".one", ".onepkg", ".pub",
        ".docm", ".xlsb", ".xlsm", ".pptm", ".pot", ".potx",
        ".pps", ".ppsx", ".ps", ".eps", ".mobi", ".azw", ".azw3",
    }
)
SCRIPT_SOURCE_EXTENSIONS = frozenset(
    {
        ".py", ".pyw", ".js", ".jsx", ".ts", ".tsx", ".sh", ".bash",
        ".zsh", ".fish", ".bat", ".cmd", ".ps1", ".php", ".java",
        ".c", ".cc", ".cpp", ".h", ".hpp", ".go", ".rs", ".rb",
        ".cs", ".vb", ".vbs", ".fs", ".fsx", ".pl", ".lua", ".swift",
        ".kt", ".kts", ".scala", ".groovy", ".gradle", ".dart", ".r",
        ".m", ".mm", ".asm", ".s", ".awk", ".sed", ".tcl", ".mk",
        ".cmake", ".sql", ".vue", ".svelte", ".qml", ".puml", ".sol",
        ".ipynb", ".proto",
    }
)
CONFIG_TEXT_EXTENSIONS = frozenset(
    {
        ".json", ".jsonl", ".yaml", ".yml", ".toml", ".ini", ".cfg",
        ".conf", ".config", ".properties", ".env", ".schema", ".css",
        ".scss", ".less",
    }
)
IMAGE_EXTENSIONS = frozenset(
    {
        ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tif", ".tiff",
        ".webp", ".svg", ".ico", ".heic", ".heif",
    }
)
USABLE_EXTENSIONS = (
    DOCUMENT_EXTENSIONS
    | ADDITIONAL_DOCUMENT_EXTENSIONS
    | SCRIPT_SOURCE_EXTENSIONS
    | CONFIG_TEXT_EXTENSIONS
    | IMAGE_EXTENSIONS
)


def is_archive(row: dict[str, str]) -> bool:
    source = row.get("file_name", "") or row.get("file_path", "")
    return Path(source).suffix.lower() in ARCHIVE_EXTENSIONS


def extensionless_is_text(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            data = handle.read(4096)
    except OSError:
        return False
    if not data or b"\0" in data:
        return False
    if data.startswith(b"#!"):
        return True
    for encoding in ("utf-8", "gb18030"):
        try:
            text = data.decode(encoding)
        except UnicodeDecodeError:
            continue
        printable = sum(character.isprintable() or character in "\r\n\t" for character in text)
        return printable / max(len(text), 1) >= 0.9
    return False


def inspect_tree(root: Path) -> tuple[int, Counter[str], Counter[str]]:
    total = 0
    extensions: Counter[str] = Counter()
    usable: Counter[str] = Counter()
    for current, _directories, filenames in os.walk(root):
        for filename in filenames:
            if filename == ".complete.json":
                continue
            total += 1
            suffix = Path(filename).suffix.lower() or "[无后缀]"
            extensions[suffix] += 1
            if suffix in USABLE_EXTENSIONS:
                usable[suffix] += 1
                return total, extensions, usable
            if suffix == "[无后缀]" and extensionless_is_text(Path(current) / filename):
                usable["[无后缀文本/脚本]"] += 1
                return total, extensions, usable
    return total, extensions, usable


def extension_summary(extensions: Counter[str], limit: int = 8) -> str:
    if not extensions:
        return "空压缩包"
    return "、".join(f"{suffix}={count}" for suffix, count in extensions.most_common(limit))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/Volumes/SharedData1/supportData"),
    )
    parser.add_argument("--apply", action="store_true", help="write classifications to the CSV")
    args = parser.parse_args()

    rows, fields = read_csv(args.csv)
    extracted_root = args.output_dir / "archives_extracted"
    counts: Counter[str] = Counter()
    unavailable_extensions: Counter[str] = Counter()

    for row in rows:
        if not is_archive(row):
            continue

        status = row.get(STATUS, "")
        reason = row.get(REASON, "")
        if status == ARCHIVE_FAILED_VALIDATION_STATUS:
            counts["验证失败改为不可用"] += 1
            if args.apply:
                apply_result(row, UNAVAILABLE_STATUS, f"压缩包验证失败，不可用：{reason}")
            continue
        if status == UNAVAILABLE_STATUS and reason.startswith("压缩包验证失败，不可用："):
            counts["验证失败保持不可用"] += 1
            continue
        if status == UNAVAILABLE_STATUS and not reason.startswith(
            "解压验证成功但未发现文档类文件"
        ):
            counts["其他不可用状态保持不变"] += 1
            continue
        if status == ARCHIVE_VERIFIED_STATUS:
            counts["原已验证保持不变"] += 1
            continue
        if status != UNAVAILABLE_STATUS:
            counts[f"保持状态：{status or '空'}"] += 1
            continue

        target = extracted_root / extraction_name(row)
        if not (target / ".complete.json").is_file():
            counts["缺少完整解压结果"] += 1
            if args.apply:
                apply_result(row, UNAVAILABLE_STATUS, "缺少完整解压结果，不可用")
            continue

        total, extensions, usable = inspect_tree(target)
        if usable:
            counts["恢复为已验证"] += 1
            if args.apply:
                apply_result(
                    row,
                    ARCHIVE_VERIFIED_STATUS,
                    f"压缩包验证成功，包含可用的文档、脚本、配置或图片：{extension_summary(usable)}",
                )
            continue

        counts["仍无可用文件"] += 1
        unavailable_extensions.update(extensions)
        if args.apply:
            detail = extension_summary(extensions)
            apply_result(
                row,
                UNAVAILABLE_STATUS,
                f"解压验证成功但未发现文档、脚本、配置或图片，不可用；共 {total} 个文件；主要类型：{detail}",
            )

    print(" ".join(f"{key}={value}" for key, value in counts.items()))
    print(f"仍不可用包主要文件类型：{extension_summary(unavailable_extensions, 20)}")
    if args.apply:
        write_csv(args.csv, rows, fields)
        print("CSV 已更新")
    else:
        print("预览模式：CSV 未修改")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
