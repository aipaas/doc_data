---
name: ocr-data-assets
description: Manage OCR and document dataset acquisition, integrity validation, evidence-bounded metadata correction, and atomic asset-inventory delivery. Use for dataset workbooks or TSV/JSON inventories, ModelScope or Hugging Face sources, external disks, blocked-download reassessment, mirrors, and resumable downloads.
---

# OCR Data Assets

This skill is self-contained and requires no sibling or project documentation at runtime.

## Required Inputs

- Require either an inventory path or an explicit list of dataset identities. Accept XLSX, CSV, TSV, or JSON inventories; each record must have a stable row key or dataset name.
- Before a full download, require an explicit target directory on a mounted filesystem. Discover current free space instead of trusting a historical disk note.
- Treat output paths, controlled vocabularies, column mappings, safety reserve, mirror policy, accepted prior checkpoints, and project-specific rules as optional user inputs. When omitted, use the defaults in this skill.
- Map non-default inventories by field meaning rather than column letter. Preserve unknown columns and the source order unless the user asks for a projection.
- A single named dataset may be processed without an inventory. In that case, create only the requested evidence or output record and do not invent a larger catalog.

## Start

1. Read this skill and the inventory, target list, prior evidence, or rules explicitly supplied by the user.
2. Identify the active inventory and schema. For file-based inventories, keep the source read-only unless explicitly authorized and record its path, mtime, SHA-256, headers, row count, stable row key, and current output count.
3. Check only mounted disks, registered paths, and the selected target. Do not recursively scan every external disk.

## Candidate Gate

- Derive scope from the user's current request and active inventory; never hard-code a row range from an earlier session. Preserve any canonical source URL already supported by evidence.
- De-duplicate by name, alias, official repository ID, URL, family, and durable completion evidence. Skip rows already sourced, acquired, stored, or present in the current verified TSV.
- Prefer the largest complete OCR/document target that fits one disk. Do not split a dataset across disks or put partial data in TSV.
- Before starting a target, compute `required_free = complete payload + peak verification temporary space + safety reserve`. Use the user's/project's reserve, or 20 GiB when none is specified. If current free space is below `required_free`, do not create a partial payload; append a `space insufficient` blocker and continue.
- Try `ModelScope -> hf-mirror -> official Hugging Face`. J is always the real official source; mirrors are transport only.
- For GitHub release assets, approved transport mirrors may be tried after a bounded `Range: 0-1023` check. Require `206`, `Accept-Ranges: bytes`, expected length, and per-range size checks. Run one short 1 MiB probe to rank mirrors before a long transfer; record the chosen mirror in metadata/P, never as provenance.
- For large GitHub raw files, use the same bounded Range probe; if raw.githubusercontent.com stalls with zero-byte growth, resume through a verified mirror such as `ghfast.top` and keep the GitHub URL as provenance.

## Download

- Use one final directory, a fixed revision/manifest, a PID, and an append-only log. A timeout means resume the same target; never start a second downloader.
- Never overwrite, merge, empty, or silently rename complete payloads or old partials. Write missing ranges to new `.resume.part/.piece` files, preserve old partials, and promote only an exact-size result.
- If a legacy partial is larger than the current range, preserve it with an explicit `.oversize-preserved-*` name and redownload that range; never truncate it. An overnight watchdog may restart only after the exact downloader PID exits and the parent mount is available.
- Keep one dataset bound to one disk until its file list, sizes, and samples pass. For range downloads, start with 2 workers; raise to 4, then 8 only when short probes show higher aggregate throughput and stable disk writes, otherwise return to the last faster setting. Use bounded request windows and switch mirrors after about 10 minutes without byte growth; resume the same partial target and never start a second downloader.
- ModelScope Git-LFS repositories may be stored as official `.zip.part-*` files. Discover the fixed revision and LFS pointer sizes/digests, download each part resumably, verify each full SHA-256, reconstruct only a temporary validation ZIP, sample its boundary members, then delete the reconstructed ZIP and samples while retaining the official parts.
- Persist compact status where practical: `dataset`, `source`, `revision`, `disk`, `expected_bytes`, `downloaded_bytes`, `pid`, `state`, `last_error`, `next_action`, `verified_at`.
- If the user explicitly chooses download-only mode, stop after remote manifest/file-size checks: do not sample archives, extract payloads, refresh TSV, or infer labels. Write `downloaded.json`/`downloaded.jsonl`; leave unlabelled `M` empty until the later validation pass.

## Low-Token Protocol

- Keep long output in logs/metadata. User updates contain only `target`, `state`, `progress`, `disk`, `next action`, and one error.
- Use one bounded probe: PID, `tail -n 8`, part sizes, mount, free space, and completion marker. Batch read-only checks; avoid repeated network probes and full tree output.
- Use at most four updates per long target: started, meaningful progress/state change, complete/blocked, TSV delivery. Do not refresh TSV per shard.
- End each target with one compact JSON/TSV validation command and one atomic TSV refresh.
- In download-only mode the final two actions are deferred; report only the download marker and resume target.

## Validation

- Check fixed-revision paths, safety, existence, non-zero status, and exact sizes. For large files sample first/middle/last 1 MiB; for archives inspect members and read representative members; for JSONL/Parquet/XML read schema/samples.
- Keep remote digest and local sample digest separate. Do not claim full hash/CRC when only sampling was done. Only durable complete evidence enters TSV.
- Keep evidence classes explicit: reused evidence, official dataset documentation/schema/tree metadata, representative sample, and complete local asset are different verification levels. Never describe metadata-only inspection as sample or payload validation.

## Reassess Undownloaded Rows

Failure to complete a download is still evidence for field correction. Reassess the row before moving to another target, but keep the conclusion within the evidence boundary.

1. Resolve the canonical dataset identity, official URL, release/revision, license, access condition, and whether the row is a dataset, catalog, portal, code repository, or composite collection.
2. Inspect official dataset documentation, schema, fixed-revision tree, manifest, API listing, or a bounded `Range`/sample probe. Prefer mirrors for transport, but keep the official URL in J.
3. Recompute only fields the evidence supports. In the default 16-column schema this means B/C from source and payload schema; F from physical multi-page or ordered page-group evidence; G/H from an exact remote manifest or clearly bounded official release; M from real input/annotation pairing; N from task usability; D from acquisition state. Map these concepts to equivalent field names in other schemas.
4. Record what was not opened locally. Never fill disk/path fields, claim local readability, claim full quality, or report observed page-count distributions when no complete local payload exists.

Treat D and N as independent axes:

- Public, fixed, schema-readable data blocked only by disk capacity: D=`确认可用还未下载`; N remains `可用`, `仅评测`, `缺label`, or `需处理` according to labels and task fit. Capacity alone is never `暂时不可用`.
- Authentication, token, password, dead endpoint, persistent 403, corruption, or transport failure after a bounded real attempt: D=`无法下载或部分下载`; N=`暂时不可用` when the necessary payload cannot be obtained.
- Search page, name list, index, tool/model repository, or isolated examples: N=`非数据集`; do not use a failed download to disguise its identity.
- Reliable schema/tree proves concrete inputs but no paired target: M names only the input and N=`缺label`, even when the full corpus is not downloaded.
- Paired labels exist but require join, conversion, regrouping, extraction, or task adaptation: M=`input -> label`; N=`需处理`, unless the higher-priority `仅评测` or `缺label` rule applies.

Write undownloaded corrections to the user-selected correction output; when none is specified, use `undownloaded-dataset-field-corrections.tsv` in the working directory. Preserve the active inventory schema, keep one dataset per row, and state `not downloaded`, evidence class, revision/release boundary, probe result, exact blocker, supported changes, and unsupported claims in the review field.

## TSV

- Write the user-selected completed-asset output atomically; when none is specified, use `completed-datasets.tsv` in the working directory. Never edit the source inventory or create a replacement copy unless explicitly authorized.
- Keep complete local assets and undownloaded corrections in separate TSVs. Never put a metadata-only, sampled, partial, or blocked target in the completed-download TSV.
- Preserve the active schema exactly. When the project uses the default 16-column form, keep `row key`, A-N, and P; do not invent an O column. Recompute row keys from the latest inventory when they are positional.
- Include only rows meeting the user's selection rule and complete local validation. The source field keeps canonical provenance, the storage field points to an existing path, sizes use explicit binary or decimal units, and the input/label field uses `input -> label` only with a real target; otherwise name only the concrete input. Resolve overlapping usability as `evaluation-only > missing-label > needs-processing`. The review field states revision, evidence class, file/size result, sample boundary, structure/pair result, and limitations.
- Before replace, validate header, 16 fields, row/name/path, H/M/N/P semantics, and stale deleted paths.

## Recovery And Final Report

- On resume check PID, target, manifest, and log first. Preserve complete payloads when a disk is unavailable. Clean only temporary extraction/sample files created by the current run.
- After a target is complete and TSV is refreshed, remove only its resolved entries from the blocker Markdown; keep active blockers and durable evidence.
- Finish with one compact line for the inventory fingerprint/schema/source rows/output rows, then one decision line per candidate (`download`, `resume`, `skip`, `blocked`) with its evidence path. Full details stay in metadata/logs.
