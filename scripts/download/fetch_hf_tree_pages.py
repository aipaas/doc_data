#!/usr/bin/env python3
"""Fetch all pages of a Hugging Face dataset tree API response."""

import argparse
import json
import re
import subprocess
import tempfile
from pathlib import Path
from urllib.parse import urlparse, urlunparse


NEXT_RE = re.compile(r"<([^>]+)>;\s*rel=\"next\"")


def mirror_url(url):
    parsed = urlparse(url)
    return urlunparse(("https", "hf-mirror.com", parsed.path, parsed.params, parsed.query, parsed.fragment))


def fetch(url, directory, page):
    headers = directory / ("page-%03d.headers" % page)
    payload = directory / ("page-%03d.json" % page)
    command = [
        "/usr/bin/curl", "--silent", "--show-error", "--location", "--fail",
        "--retry", "8", "--retry-all-errors", "--retry-delay", "3",
        "--max-time", "180", "-D", str(headers), "-o", str(payload), url,
    ]
    subprocess.run(command, check=True)
    items = json.loads(payload.read_text(encoding="utf-8"))
    if not isinstance(items, list):
        raise RuntimeError("Unexpected tree page response: %s" % type(items).__name__)
    next_url = None
    for line in headers.read_text(encoding="iso-8859-1").splitlines():
        match = NEXT_RE.search(line)
        if match:
            next_url = mirror_url(match.group(1))
            break
    return items, next_url


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    url = (
        "https://hf-mirror.com/api/datasets/%s/tree/%s"
        "?expand=true&recursive=true&limit=%d"
        % (args.repo, args.revision, args.limit)
    )
    all_items = []
    page = 1
    while url:
        items, url = fetch(url, args.work, page)
        all_items.extend(items)
        print("page=%d entries=%d total=%d" % (page, len(items), len(all_items)), flush=True)
        page += 1
    args.output.write_text(json.dumps(all_items, ensure_ascii=True, sort_keys=True) + "\n", encoding="utf-8")
    print("wrote=%s entries=%d" % (args.output, len(all_items)), flush=True)


if __name__ == "__main__":
    main()
