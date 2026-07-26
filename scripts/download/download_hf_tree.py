#!/usr/bin/env python3
"""Download and verify every file described by a Hugging Face tree response."""

import argparse
import concurrent.futures
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.parse
from pathlib import Path, PurePosixPath


CHUNK_BYTES = 8 * 1024**2
MIN_FREE_BYTES = 20 * 1024**3


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--tree", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--mount", type=Path, default=Path("/Volumes/SharedData"))
    parser.add_argument("--base-url", default="https://hf-mirror.com")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--attempts", type=int, default=30)
    parser.add_argument("--chunk-mib", type=int, default=32)
    return parser.parse_args()


def git_blob_sha1(path, size):
    digest = hashlib.sha1()
    digest.update(("blob %d\0" % size).encode("ascii"))
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(CHUNK_BYTES)
            if not chunk:
                return digest.hexdigest()
            digest.update(chunk)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(CHUNK_BYTES)
            if not chunk:
                return digest.hexdigest()
            digest.update(chunk)


class Downloader:
    def __init__(self, args, files):
        self.args = args
        self.files = files
        self.stop_event = threading.Event()
        self.process_lock = threading.Lock()
        self.active_processes = set()
        self.record_lock = threading.Lock()
        self.verified_path = args.root / "metadata" / "verified.jsonl"

    def mount_ready(self):
        if not self.args.mount.is_mount():
            self.stop_event.set()
            print("Mount disappeared: %s" % self.args.mount, flush=True)
            return False
        return True

    @staticmethod
    def is_verified(item, path):
        if not path.is_file() or path.stat().st_size != item["size"]:
            return False
        if item.get("sha256"):
            return sha256(path) == item["sha256"]
        return git_blob_sha1(path, item["size"]) == item["oid"]

    def record_verified(self, item):
        record = {
            "hash_kind": "sha256" if item.get("sha256") else "git-blob-sha1",
            "digest": item.get("sha256") or item["oid"],
            "path": item["path"],
            "size": item["size"],
            "verified_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        with self.record_lock:
            self.verified_path.parent.mkdir(parents=True, exist_ok=True)
            with self.verified_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())

    def download_one(self, item):
        if self.stop_event.is_set() or not self.mount_ready():
            return False
        destination = self.args.root / item["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_suffix(destination.suffix + ".part")
        if self.is_verified(item, destination):
            self.record_verified(item)
            return True
        destination.unlink(missing_ok=True)
        if partial.exists() and partial.stat().st_size > item["size"]:
            partial.unlink()

        quoted_path = urllib.parse.quote(item["path"], safe="/")
        url = "%s/datasets/%s/resolve/%s/%s" % (
            self.args.base_url.rstrip("/"),
            self.args.repo,
            self.args.revision,
            quoted_path,
        )
        while True:
            if self.stop_event.is_set() or not self.mount_ready():
                return False
            resume = partial.stat().st_size if partial.exists() else 0
            if resume == item["size"]:
                if self.is_verified(item, partial):
                    os.replace(partial, destination)
                    self.record_verified(item)
                    return True
                partial.unlink()
                resume = 0
            if shutil.disk_usage(self.args.mount).free - (item["size"] - resume) < MIN_FREE_BYTES:
                raise RuntimeError("Free-space reserve reached")
            if item["size"] == 0:
                partial.touch()
                continue

            range_end = min(item["size"] - 1, resume + self.args.chunk_mib * 1024**2 - 1)
            expected_chunk_size = range_end - resume + 1
            chunk = partial.with_suffix(partial.suffix + ".chunk")
            downloaded = False
            for attempt in range(1, self.args.attempts + 1):
                chunk.unlink(missing_ok=True)
                print(
                    "Downloading %s attempt=%d range=%d-%d resume=%.2f MiB"
                    % (item["path"], attempt, resume, range_end, resume / 1024**2),
                    flush=True,
                )
                command = [
                    "/usr/bin/curl",
                    "--location",
                    "--fail",
                    "--silent",
                    "--show-error",
                    "--retry",
                    "4",
                    "--retry-all-errors",
                    "--retry-delay",
                    "3",
                    "--connect-timeout",
                    "30",
                    "--speed-limit",
                    "1024",
                    "--speed-time",
                    "180",
                    "--max-time",
                    "300",
                    "--range",
                    "%d-%d" % (resume, range_end),
                    "--output",
                    str(chunk),
                    url,
                ]
                process = subprocess.Popen(command)
                with self.process_lock:
                    self.active_processes.add(process)
                return_code = process.wait()
                with self.process_lock:
                    self.active_processes.discard(process)
                if self.stop_event.is_set():
                    return False
                if (
                    return_code == 0
                    and chunk.is_file()
                    and chunk.stat().st_size == expected_chunk_size
                ):
                    with partial.open("ab") as output, chunk.open("rb") as source:
                        shutil.copyfileobj(source, output, length=CHUNK_BYTES)
                        output.flush()
                        os.fsync(output.fileno())
                    chunk.unlink()
                    downloaded = True
                    break
                chunk.unlink(missing_ok=True)
                time.sleep(min(30, attempt * 3))
            if not downloaded:
                raise RuntimeError("Failed after retries: %s" % item["path"])

    def stop(self, signum, _frame):
        print("Received signal %d; stopping" % signum, flush=True)
        self.stop_event.set()
        with self.process_lock:
            processes = list(self.active_processes)
        for process in processes:
            process.terminate()

    def run(self):
        signal.signal(signal.SIGINT, self.stop)
        signal.signal(signal.SIGTERM, self.stop)
        completed = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.args.workers) as executor:
            futures = {executor.submit(self.download_one, item): item for item in self.files}
            for future in concurrent.futures.as_completed(futures):
                item = futures[future]
                try:
                    ok = future.result()
                except Exception as error:
                    print("Failed %s: %s" % (item["path"], error), flush=True)
                    self.stop_event.set()
                    ok = False
                if ok:
                    completed += 1
                    if completed % 25 == 0 or completed == len(self.files):
                        print("Progress %d/%d" % (completed, len(self.files)), flush=True)
                if self.stop_event.is_set():
                    for queued in futures:
                        queued.cancel()
                    break
        return 1 if self.stop_event.is_set() else 0


def main():
    args = parse_args()
    if not args.mount.is_mount():
        raise SystemExit("External disk is not mounted: %s" % args.mount)
    raw = json.loads(args.tree.read_text(encoding="utf-8"))
    files = []
    for item in raw:
        if item.get("type") != "file":
            continue
        path = PurePosixPath(item["path"])
        if path.is_absolute() or ".." in path.parts:
            raise RuntimeError("Unsafe repository path: %s" % path)
        if path.name == ".DS_Store":
            continue
        lfs = item.get("lfs") or {}
        files.append(
            {
                "path": str(path),
                "size": int(item["size"]),
                "oid": item["oid"],
                "sha256": lfs.get("oid"),
            }
        )
    files.sort(key=lambda item: item["path"])
    args.root.mkdir(parents=True, exist_ok=True)
    metadata = args.root / "metadata"
    metadata.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.tree, metadata / "tree.json")
    (metadata / "repo.json").write_text(
        json.dumps({"id": args.repo, "revision": args.revision}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "Audit: repo=%s files=%d total=%.3f GiB"
        % (args.repo, len(files), sum(item["size"] for item in files) / 1024**3),
        flush=True,
    )
    return Downloader(args, files).run()


if __name__ == "__main__":
    sys.exit(main())
