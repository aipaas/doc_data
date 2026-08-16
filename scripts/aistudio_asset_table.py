#!/usr/bin/env python3
"""Generate a portable inventory table for the currently mounted AI Studio disks."""

from __future__ import annotations

import argparse
import json
import os
import re
import unicodedata
import zipfile
from pathlib import Path
from typing import Any, Sequence
from xml.etree import ElementTree as ET


HEADER = [
    "数据集",
    "来源",
    "类型",
    "状态",
    "团队硬盘",
    "是否多页",
    "是否>500GB",
    "数量",
    "预览",
    "原始链接",
    "存储链接",
    "备注",
    "输入与标签",
    "可用性",
    "review备注",
]
AISTUDIO_ROOT = "aistudio"
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = REPO_ROOT / "data_sources/.aistudio_ocr_cache.json"
DEFAULT_DECISIONS = REPO_ROOT / "data_sources/.aistudio-ocr-decisions.json"
AGGREGATE_ROW = 116

NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
PARTIAL_MARKERS = (".part", ".partial", ".incomplete", ".crdownload", ".chunk")


def clean(value: Any, limit: int = 700) -> str:
    text = str(value or "").replace("\r", " ").replace("\n", " ").replace("\t", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit]


def normalize(value: str) -> str:
    value = unicodedata.normalize("NFKC", value or "").lower()
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", value)


def read_workbook(workbook: Path) -> dict[str, list[int]]:
    """Read only the first (and currently only) worksheet without changing the XLSX."""
    result: dict[str, list[int]] = {}
    with zipfile.ZipFile(workbook) as archive:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            shared = [
                "".join(node.text or "" for node in item.iter(f"{{{NS['m']}}}t"))
                for item in root.findall(".//m:si", NS)
            ]
        root = ET.fromstring(archive.read("xl/worksheets/sheet2.xml"))
        for row in root.findall(".//m:sheetData/m:row", NS):
            row_number = int(row.attrib["r"])
            if row_number == 1:
                continue
            values: dict[str, str] = {}
            for cell in row.findall("m:c", NS):
                match = re.match(r"[A-Z]+", cell.attrib["r"])
                if not match:
                    continue
                cell_type = cell.attrib.get("t")
                if cell_type == "inlineStr":
                    value = "".join(
                        node.text or "" for node in cell.iter(f"{{{NS['m']}}}t")
                    )
                else:
                    node = cell.find("m:v", NS)
                    value = node.text if node is not None else ""
                    if cell_type == "s" and value:
                        value = shared[int(value)]
                values[match.group(0)] = value
            name = clean(values.get("A", ""), limit=300)
            if name:
                result.setdefault(normalize(name), []).append(row_number)
    return result


def list_legacy(volumes: Sequence[Path]) -> list[tuple[Path, str, str, Path]]:
    entries: list[tuple[Path, str, str, Path]] = []
    for volume in volumes:
        root = volume / AISTUDIO_ROOT / "legacy"
        if not root.is_dir():
            continue
        for item in sorted(root.iterdir(), key=lambda path: path.name):
            if not item.is_dir():
                continue
            match = re.match(r"^(\d+)(?:_(.*))?$", item.name)
            if not match:
                continue
            dataset_id = match.group(1)
            local_title = match.group(2) or item.name
            entries.append((item, dataset_id, local_title, volume))
    return entries


def list_repositories(volumes: Sequence[Path]) -> list[tuple[Path, str, Path]]:
    entries: list[tuple[Path, str, Path]] = []
    for volume in volumes:
        root = volume / AISTUDIO_ROOT
        if not root.is_dir():
            continue
        for owner in sorted(root.iterdir(), key=lambda path: path.name):
            if not owner.is_dir() or owner.name == "legacy":
                continue
            for repository in sorted(owner.iterdir(), key=lambda path: path.name):
                if repository.is_dir() and (repository / ".msc").is_file():
                    relative = repository.relative_to(root).as_posix()
                    entries.append((repository, relative, volume))
    return entries


def payload_stats(root: Path, repository: bool) -> dict[str, Any]:
    files: list[str] = []
    total = 0
    nonzero = 0
    partial: list[str] = []
    skip = {".msc", ".mv", "README.md", "dataset_infos.json"} if repository else set()
    for current, dirs, names in os.walk(root):
        dirs[:] = sorted(name for name in dirs if not name.startswith("."))
        for name in sorted(names):
            if name.startswith(".") or name in skip:
                continue
            path = Path(current) / name
            try:
                size = path.stat().st_size
            except OSError:
                continue
            relative = path.relative_to(root).as_posix()
            files.append(relative)
            total += size
            if size:
                nonzero += 1
            if name.lower().endswith(PARTIAL_MARKERS):
                partial.append(relative)
    return {
        "count": len(files),
        "nonzero": nonzero,
        "bytes": total,
        "files": files,
        "partial": partial,
    }


def size_text(size: int) -> str:
    if size >= 1024**3:
        return f"{size / 1024**3:.3f} GiB"
    return f"{size / 1024**2:.3f} MiB"


def storage_link(path: Path, volume: Path, absolute: bool) -> str:
    if absolute:
        return str(path)
    try:
        relative = path.relative_to(volume)
    except ValueError:
        return str(path)
    return f"{volume.name}/{relative.as_posix()}"


def catalog_data(
    catalog_path: Path, decisions_path: Path
) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]]]:
    cache = json.loads(catalog_path.read_text(encoding="utf-8"))["details"]
    details = {int(key): value for key, value in cache.items()}
    decisions = json.loads(decisions_path.read_text(encoding="utf-8"))["datasets"]
    decision_map = {int(item["dataset_id"]): item for item in decisions}
    return details, decision_map


def type_text(detail: dict[str, Any], decision: dict[str, Any]) -> str:
    category = clean(decision.get("dataset_type", ""))
    tags = " ".join(str(tag) for tag in detail.get("tags", []))
    joined = f"{category} {tags} {detail.get('datasetName', '')}"
    result: list[str] = []
    if "表格" in joined or "table" in joined.lower():
        result.append("表格图片")
    elif "公式" in joined or "latex" in joined.lower() or "math" in joined.lower():
        result.append("公式")
    elif "版面" in joined or "layout" in joined.lower():
        result.append("版面")
    elif "文档" in joined or category.startswith("文档"):
        result.append("文档图片")
    if "OCR" in joined or category.startswith("OCR"):
        result.append("OCR")
    return ",".join(dict.fromkeys(result)) or "通用图文"


def input_and_label(type_value: str, annotation: str) -> str:
    if "文档图片" in type_value or "表格图片" in type_value or "版面" in type_value:
        input_name = "文档/页面图像"
    elif "OCR" in type_value or "公式" in type_value:
        input_name = "OCR图像"
    else:
        input_name = "数据集输入"
    annotation = clean(annotation, limit=260)
    if not annotation or annotation == "未说明":
        return input_name
    return f"{input_name} -> {annotation}"


def availability(detail: dict[str, Any], decision: dict[str, Any], stats: dict[str, Any]) -> str:
    if not stats["count"] or stats["partial"] or stats["nonzero"] != stats["count"]:
        return "暂时不可用"
    category = clean(decision.get("dataset_type", ""))
    if category.startswith("非数据") or category.startswith("非目标"):
        return "非数据集"
    name = f"{detail.get('datasetName', '')} {category}".lower()
    if any(token in name for token in ("benchmark", "bench", "评测", "test", "测试")):
        return "仅评测"
    annotation = clean(decision.get("annotation", ""))
    if not annotation or annotation.startswith("未说明"):
        return "缺label"
    return "需处理"


def matches(name: str, local_title: str, dataset_id: int, workbook: dict[str, list[int]]) -> list[int]:
    found: set[int] = set()
    for candidate in (name, local_title):
        found.update(workbook.get(normalize(candidate), []))
    aliases = {
        "docvqatestsubsampled": "DocVQA",
        "documentvqa": "DocVQA",
        "iaocr": "moondream/ia_ocr",
    }
    alias = aliases.get(normalize(name))
    if alias:
        found.update(workbook.get(normalize(alias), []))
    return sorted(found)


def row_for(
    path: Path,
    dataset_id: int | None,
    local_title: str,
    volume: Path,
    repository: bool,
    relative_repo: str | None,
    details: dict[int, dict[str, Any]],
    decisions: dict[int, dict[str, Any]],
    workbook: dict[str, list[int]],
    absolute_storage_paths: bool,
) -> list[str]:
    detail = details.get(dataset_id or -1, {})
    decision = decisions.get(dataset_id or -1, {})
    stats = payload_stats(path, repository)
    name = clean(detail.get("datasetName") or local_title or relative_repo or path.name, limit=300)
    if repository and not detail:
        name = clean(relative_repo or path.name, limit=300)
    type_value = type_text(detail, decision)
    complete = bool(stats["count"]) and not stats["partial"] and stats["nonzero"] == stats["count"]
    row_status = "已在团队硬盘" if complete else "还未确认是否可用"
    catalog_type = clean(decision.get("dataset_type", ""))
    catalog_decision = clean(decision.get("decision", ""))
    annotation = clean(decision.get("annotation", ""))
    matched_rows = matches(name, local_title, dataset_id or -1, workbook)
    preview = "; ".join(stats["files"][:5]) or "无可见payload文件"
    if len(stats["files"]) > 5:
        preview += "; ..."
    quantity = f"{stats['count']}个文件；{size_text(stats['bytes'])}"
    platform_quantity = clean(decision.get("quantity", ""), limit=180)
    if platform_quantity:
        quantity += f"；平台记录：{platform_quantity}"
    source_url = (
        f"https://aistudio.baidu.com/datasetdetail/{dataset_id}"
        if dataset_id
        else ""
    )
    category = clean(decision.get("authenticity", ""))
    granularity = clean(decision.get("granularity", ""))
    license_name = clean(detail.get("protocolName", ""))
    remarks = (
        f"AI Studio ID={dataset_id or '未知'}；类型决策={catalog_type or '未记录'}；"
        f"真实性={category or '未记录'}；粒度={granularity or '未记录'}；"
        f"许可={license_name or '未记录'}；目录决策={catalog_decision or '未记录'}"
    )
    if repository and relative_repo:
        remarks += f"；repo={relative_repo}"
    partial_text = "；发现部分下载标记=" + ",".join(stats["partial"]) if stats["partial"] else ""
    if matched_rows:
        match_text = "原表单项匹配=" + ",".join(map(str, matched_rows))
    else:
        match_text = f"原表无同名单项；由原表第{AGGREGATE_ROW}行aistudio聚合项覆盖"
    review = (
        f"AI Studio ID={dataset_id or '未知'}；本地逐文件stat={stats['count']}个，"
        f"非零={stats['nonzero']}个；{match_text}；"
        f"本地资产库存检查，未重新取得远程manifest，也未解压归档/做内容哈希；"
        f"卷={volume.name}{partial_text}"
    )
    if repository and relative_repo:
        review += f"；repo路径={relative_repo}"
    return [
        name,
        "aistudio",
        type_value,
        row_status,
        volume.name,
        "□",
        "□",
        quantity,
        preview,
        source_url,
        storage_link(path, volume, absolute_storage_paths),
        remarks,
        input_and_label(type_value, annotation),
        availability(detail, decision, stats),
        clean(review, limit=1000),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workbook",
        type=Path,
        required=True,
        help="Path to the local XLSX asset workbook.",
    )
    parser.add_argument(
        "--volume",
        dest="volumes",
        type=Path,
        action="append",
        required=True,
        help="Mounted asset volume; repeat for each volume to scan.",
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=DEFAULT_CATALOG,
        help=f"Catalog cache JSON (default: {DEFAULT_CATALOG}).",
    )
    parser.add_argument(
        "--decisions",
        type=Path,
        default=DEFAULT_DECISIONS,
        help=f"Decision JSON (default: {DEFAULT_DECISIONS}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output TSV path; use a path outside the repository for local exports.",
    )
    parser.add_argument(
        "--absolute-storage-paths",
        action="store_true",
        help="Include machine-specific absolute storage paths in the output.",
    )
    args = parser.parse_args()
    workbook_path = args.workbook.expanduser()
    volumes = [path.expanduser() for path in args.volumes]
    catalog_path = args.catalog.expanduser()
    decisions_path = args.decisions.expanduser()
    output_path = args.output.expanduser()
    if not workbook_path.is_file():
        parser.error(f"workbook does not exist: {workbook_path}")
    missing_volumes = [path for path in volumes if not path.is_dir()]
    if missing_volumes:
        parser.error("volume does not exist: " + ", ".join(map(str, missing_volumes)))
    if not catalog_path.is_file():
        parser.error(f"catalog does not exist: {catalog_path}")
    if not decisions_path.is_file():
        parser.error(f"decisions do not exist: {decisions_path}")
    workbook = read_workbook(workbook_path)
    details, decisions = catalog_data(catalog_path, decisions_path)
    legacy_entries = list_legacy(volumes)
    repository_entries = list_repositories(volumes)
    rows: list[list[str]] = []
    for path, dataset_id_text, local_title, volume in legacy_entries:
        rows.append(
            row_for(
                path,
                int(dataset_id_text),
                local_title,
                volume,
                False,
                None,
                details,
                decisions,
                workbook,
                args.absolute_storage_paths,
            )
        )
    repo_map = {
        Path(clean(value.get("repoName"))).name: (int(key), value)
        for key, value in details.items()
        if value.get("repoName")
    }
    for path, relative_repo, volume in repository_entries:
        repo_key = Path(relative_repo).name
        dataset_id = repo_map.get(repo_key, (None, {}))[0]
        local_title = repo_map.get(repo_key, (None, {}))[1].get("datasetName", relative_repo)
        rows.append(
            row_for(
                path,
                dataset_id,
                clean(local_title, limit=300),
                volume,
                True,
                relative_repo,
                details,
                decisions,
                workbook,
                args.absolute_storage_paths,
            )
        )
    rows.sort(key=lambda row: (row[4], row[0], row[10]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("\t".join(HEADER) + "\n")
        for row in rows:
            handle.write("\t".join(clean(value, limit=1200) for value in row) + "\n")
    matched = sum("原表单项匹配=" in row[-1] for row in rows)
    print(f"output={output_path}")
    print(f"rows={len(rows)} legacy={len(legacy_entries)} repositories={len(repository_entries)}")
    print(f"individual_workbook_matches={matched}")


if __name__ == "__main__":
    main()
