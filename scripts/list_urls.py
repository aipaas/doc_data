#!/usr/bin/env python3
"""Export mirror download URLs for one Hugging Face dataset repository."""

from __future__ import annotations

import argparse
import os
from collections import defaultdict
from pathlib import Path
from urllib.parse import quote

from huggingface_hub import HfApi


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo_id", help="Dataset repository ID, for example owner/name.")
    parser.add_argument("--revision", default="main", help="Branch, tag, or commit.")
    parser.add_argument(
        "--endpoint",
        default="https://hf-mirror.com",
        help="Hugging Face-compatible API endpoint.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("download_urls.txt"),
        help="Output text file (default: ./download_urls.txt).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    endpoint = args.endpoint.rstrip("/")
    api = HfApi(endpoint=endpoint, token=os.environ.get("HF_TOKEN") or None)
    files = api.list_repo_files(
        repo_id=args.repo_id,
        repo_type="dataset",
        revision=args.revision,
    )

    groups: dict[str, list[str]] = defaultdict(list)
    for file_path in files:
        directory = file_path.rsplit("/", 1)[0] if "/" in file_path else ""
        groups[directory].append(file_path)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    repo_path = quote(args.repo_id, safe="/")
    revision_path = quote(args.revision, safe="")
    with args.output.open("w", encoding="utf-8") as handle:
        for directory in sorted(groups):
            for file_path in sorted(groups[directory]):
                encoded_path = quote(file_path, safe="/")
                handle.write(
                    f"{endpoint}/datasets/{repo_path}/resolve/"
                    f"{revision_path}/{encoded_path}\n"
                )

    print(f"Exported {len(files)} URLs to {args.output}")


if __name__ == "__main__":
    main()
