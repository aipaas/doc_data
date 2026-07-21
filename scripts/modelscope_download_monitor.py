#!/usr/bin/env python3
"""Write live progress for a running ModelScope batch downloader."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import psutil


DEFAULT_ROOT = Path(r"E:\data\doc\modelscope")
STATE_NAME = "modelscope-download-state.json"
LOG_NAME = "modelscope-download-live.log"


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def human_size(value: int) -> str:
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024 or unit == "TiB":
            return f"{int(size):,} {unit}" if unit == "B" else f"{size:,.2f} {unit}"
        size /= 1024
    raise AssertionError("unreachable")


def folder_stats(path: Path) -> tuple[int, int, str]:
    total = 0
    count = 0
    newest = 0.0
    if not path.exists():
        return total, count, "-"
    for base, _, names in os.walk(path):
        for name in names:
            try:
                stat = (Path(base) / name).stat()
            except OSError:
                continue
            total += stat.st_size
            count += 1
            newest = max(newest, stat.st_mtime)
    newest_text = datetime.fromtimestamp(newest).astimezone().isoformat(timespec="seconds") if newest else "-"
    return total, count, newest_text


def read_state(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def process_running(pid: int) -> bool:
    if not psutil.pid_exists(pid):
        return False
    try:
        return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False


def progress_line(root: Path, state: dict[str, Any], pid: int) -> str:
    datasets = state.get("datasets", {})
    summary = Counter(item.get("status", "unknown") for item in datasets.values())
    active = [(key, item) for key, item in datasets.items() if item.get("status") == "downloading"]
    if active:
        key, item = active[-1]
        local_bytes, local_files, newest = folder_stats(Path(item["path"]))
        active_text = f"current={key} files={local_files} bytes={human_size(local_bytes)} newest={newest}"
    else:
        active_text = "current=-"
    summary_text = ",".join(f"{key}:{value}" for key, value in sorted(summary.items()))
    free = shutil.disk_usage(root).free
    return (
        f"{now()} pid={pid} state_updated={state.get('updated_at', '-')} "
        f"summary={summary_text or '-'} {active_text} free={human_size(free)}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--interval", type=float, default=15.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.pid <= 0 or args.interval <= 0:
        parser.error("pid and interval must be positive")
    return args


def main() -> None:
    args = parse_args()
    root = args.root.resolve(strict=False)
    state_path = root / STATE_NAME
    log_path = root / LOG_NAME
    while True:
        try:
            state = read_state(state_path)
            line = progress_line(root, state, args.pid)
        except Exception as exc:
            line = f"{now()} monitor_error={type(exc).__name__}: {exc}"
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        if args.once or not process_running(args.pid):
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
