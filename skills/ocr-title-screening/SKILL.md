---
name: ocr-title-screening
description: Validate and classify OCR/document datasets for multi-page document title or heading correction, preserving prior evidence and producing checkpoint JSON, TSV, or a single-sheet XLSX. Use for inventory review, input-to-label mappings, physical multi-page status, usability, task relevance, undownloaded field reassessment, and bounded mirror-assisted sampling.
---

# OCR Title Screening

Produce evidence-backed rows for every assigned inventory record. Reuse existing labels and local validation before doing any network work. This skill is self-contained and requires no sibling or project documentation at runtime.

## Required Inputs

- Require either an inventory path or an explicit list of dataset identities. Accept XLSX, CSV, TSV, or JSON inventories; each record must have a stable row key or exact dataset name.
- Require an output path or use the defaults in `Checkpoint Contract`. Never overwrite the source inventory unless the user explicitly authorizes it.
- Treat an existing checkpoint, accepted prior verification, registered local paths, project vocabularies, column mappings, target-task refinements, mirror policy, and project rules as optional inputs. When absent, start with no prior evidence and use this skill's defaults.
- Map differently named fields by meaning. Preserve source order, exact dataset names, and extra source columns unless the user requests a smaller projection.
- External project documents may be supplied as optional context. Explicit user rules override these defaults, but no external document may be assumed to exist.

## Source Of Truth

- Preserve the source record's stable identity and dataset name exactly.
- Read the source record, any user-supplied project rules, prior checkpoint, and registered local evidence first.
- Treat a populated original input/label field as reusable evidence unless a stronger local or official source disproves it.
- Do not repeat a completed download or validation. Do not touch another session's active download directory.
- Retain every assigned row, including blocked, composite, index-only, repository-only, and unavailable entries.

## Evidence Order

1. Reuse the source inventory and accepted prior verification.
2. Inspect already downloaded local files and metadata without copying them.
3. Read the official dataset card, paper, repository, schema, or file listing.
4. Only when still unresolved, download a small representative sample into a dedicated temporary directory.

Record the evidence class exactly. `official_metadata` means official dataset documentation/Dataset Card/schema/tree inspection and must never be reported as `sample_verified`; `sample_verified` requires reading representative payload records/files; `locally_verified` requires an existing local asset check. Reuse remains `reused`, even when the reused evidence originally came from a sample.

For Hugging Face transport, try `hf-mirror.com` before the official endpoint. For GitHub transport, try `gh-proxy.com` before the official endpoint. Keep the canonical official URL as provenance. Bound probes by file and byte count; do not fetch a full corpus merely to inspect schema. Remove only temporary files created by the current task after recording the evidence.

## Undownloaded Field Review

Do not equate `not downloaded` with `not reviewable`. Official evidence can support precise field corrections without a full local corpus, provided the note states the boundary.

| Evidence obtained | Fields that may be corrected | Claims that remain prohibited |
| --- | --- | --- |
| Canonical page/repository and license | source, official URL, dataset identity, access condition | payload readability, labels, multi-page status |
| Official dataset documentation or Dataset Card | documented input, task, split, license, reported scale | observed schema/content quality unless the documentation exposes them exactly |
| Official schema or representative record | data type, concrete input, real paired labels, processing requirement | full-corpus integrity or consistency |
| Fixed revision tree/manifest | file count, exact remote bytes, release boundary, >500 GB flag | local completeness or local path |
| Explicit `document_id + page_number`, ordered pages, or physical multi-page format | multi-page status | observed page-count distribution without payload inspection |
| Bounded mirror/official access probe | reachable, gated, forbidden, missing, corrupt, or transport-blocked state | completion of files not fully transferred |

Keep acquisition state and task usability independent:

- Capacity-only blocker with a public fixed release: compute `complete payload + peak verification temporary space + safety reserve` first, using a 20 GiB reserve when the project supplies none. Keep acquisition as `确认可用还未下载`; classify usability from the label schema. Do not create a partial payload or write `暂时不可用`.
- Required token/password, persistent 403, dead link, corruption, or failed mirror/official bounded probe: acquisition is `无法下载或部分下载`; use `暂时不可用` only when that blocker prevents obtaining the necessary payload.
- Reliable schema confirms inputs but no paired task target: write only the concrete input and classify `缺label`; a full download is not required to invent or disprove a label.
- Schema confirms paired targets requiring conversion/join/regrouping: write `input -> label` and classify `需处理`, subject to `仅评测 > 缺label > 需处理`.
- Index, portal, name list, tool/model repository, or isolated examples: classify `非数据集`; do not turn it into a dataset because a file-like URL appears in metadata.

For every undownloaded correction, state `未下载` plus the exact evidence class and limitation in the note. Never populate a local disk/path, claim sample inspection from metadata, or claim complete validation.

## Required Classification

Fill every field. Never emit `待核验`, `未知`, `待确认`, `待验证`, an empty string, or a placeholder.

Classify usability, physical multi-page status, and title-correction relevance independently. Usability measures readiness and processing cost; multi-page status measures physical page grouping; relevance measures how directly the paired evidence supervises title correction. Never derive one field solely from another.

### Input And Label

Use one field, not separate input/output columns.

- Write `输入 -> 标签` when a paired target or annotation exists. Name concrete representations, for example `页面图像 -> 行级OCR文本+bbox+层级标签`.
- When a dataset genuinely has no corresponding target annotation, write only the concrete input carrier in `input_and_label` (for example `多页 PDF` or `文档页面图像`). Never write bare `缺label` or append `-> 缺label`; record `缺label` in `availability` instead.
- Do not infer missing labels merely because the original cell is empty. Check the official schema or representative annotation first.
- Resolve overlapping availability categories in this order: `仅评测` > `缺label` > `需处理`.
- For a composite catalog, collection, search entry, or tool bundle, keep the row and name its actual aggregate input, but skip child-by-child validation.
- QA answers, class labels, layout regions, reading order, Markdown, HTML, OCR text, bounding boxes, hierarchy, and linking are labels when paired with an input.

### Data Type

Use the project's controlled vocabulary when one is supplied. Otherwise use this default comma-separated multi-select: `table image`, `document image`, `chart`, `OCR`, `document file`, `web page`, `formula`, `general image-text`, and `layout`. Localized equivalents are allowed, but one output must use one vocabulary consistently.

- Reuse a populated source type exactly when it is valid under the active vocabulary.
- Infer missing types from the actual payload/schema, not merely the dataset name.
- Use multiple values when the asset genuinely spans categories, for example `文档图片, OCR`.
- Do not repeat an option or invent a new option.

### Usability

Choose exactly one:

- `可用`: necessary assets are accessible and paired input/target support the current evaluation or training task without manual inference, cross-source joins, or task reconstruction.
- `仅评测`: official purpose, split, or release boundary limits the asset to benchmark/test use.
- `缺label`: a representative sample or reliable schema is readable and confirms concrete input but no paired target.
- `需处理`: paired evidence exists but deterministic label extraction, conversion, regrouping, OCR alignment, archive extraction, page reconstruction, or task adaptation is needed.
- `暂时不可用`: after an actual local, preferred-mirror, or official-source sample attempt, permission, password, mounting, corruption, dead links, or transport still blocks the necessary asset.
- `非数据集`: a search/index/tool/model repository or isolated examples rather than a standalone dataset.

Missing local data, an incomplete full download, or metadata-only evidence does not by itself mean `暂时不可用`. Attempt a bounded documentation, schema, Range, or representative-sample fetch first and record the channel/result. A successful sample must be classified as `仅评测`, `缺label`, `需处理`, or `可用`. Resolve overlap as `仅评测 > 缺label > 需处理`.

### Multi-Page

Choose `✅` only when one logical sample is a physical multi-page PDF/document or has explicit ordered page membership that can reconstruct such a document. Record direct evidence such as page count, document/page IDs, ordered page lists, or cross-page links.

Choose `❌` for independent page images, ordinary XML/HTML/JSONL containers, scanned single pages, multiple unrelated files, or a web page. File structure alone is not multi-page evidence.

### Title-Correction Relevance

Choose exactly one of `高`, `中`, `低`, or `不相关` and write a dataset-specific reason.

- Apply an identity gate first: when `availability=非数据集`, set relevance to `不相关`. Indexes, portals without a released corpus, model/tool repositories, schema-less collections, and isolated fixtures do not become relevant datasets merely because their descriptions mention documents.
- Require a pairing between title evidence and an OCR-addressable carrier: a page image, PDF, reproducible page render, page-level OCR output, or ordered page images. A standalone title list, metadata record, or XML/HTML structure without a page/render mapping cannot be `高`; it is at most `中` when rendering, joining, or alignment can construct that mapping. Require an official schema with explicit paired fields or an inspected representative/full sample for `高`; description-only evidence is at most `中`.
- `高`: paired OCR-addressable evidence directly identifies a document title or chapter/section heading through text, a title/heading role and region, heading level or parent-child relation, Markdown `#` syntax, HTML `h1-h6`, or a table-of-contents mapping. Deterministic filtering must form the target without first inferring which span is a heading.
- `中`: no title-specific label exists but complete page/document context plus OCR text, generic layout boxes, reading order, document grouping, ordered pages, or raw physical documents supports candidate or pseudo-label construction; or structured headings exist but still require rendering/joining to an OCR input. Heading identity, boundary, corrected text, or input alignment still requires heuristic, model, human inference, or cross-source processing.
- `低`: only indirect transfer remains, such as table/formula labels, KIE or form fields, DocVQA, scene/crop OCR, generic single-page OCR, or domain text without paired document-title supervision.
- `不相关`: the row is not an independent dataset or lacks a credible construction path.

Count document titles, chapter/section headings, heading roles, heading text/boxes, hierarchy, and TOC mappings as direct title signals. Do not count page headers/footers, table headers or captions, figure/chart titles, form field names, KIE `HEADER`/`QUESTION`, QA answers, plain OCR text, or generic Markdown/HTML without heading syntax as document-heading supervision.

Physical multi-page evidence is not a gate for `高`. A single-page dataset with direct heading supervision may be `高` while `multipage=❌`; it supports the title subtask but is excluded when a report separately filters a multi-page core pool. Conversely, multi-page input without direct title labels is not automatically `高`.

Programmatically generated, weak, or official structural labels may still be directly relevant. Record their provenance and quality boundary; reflect required extraction or quality work in usability instead of automatically downgrading relevance.

Name all four facts in the reason: the direct title label or missing signal, physical page-order evidence, required transformation, and label provenance (`human`, `official_structure`, `programmatic`, or `weak`). Avoid generic phrases such as “可用于标题修正”, “has Markdown”, or one template repeated across datasets.

### Core-Pool Reporting

Report the field-derived candidate pool separately from the evidence-qualified core pool:

- `C0 = multipage=✅ AND relation in {高, 中} AND availability != 非数据集`.
- `C1 = C0` plus explicit page-order evidence and an independent fixed release or frozen snapshot/batch. Use `C1` as the acquisition-coverage denominator.
- Split `C1` into direct multi-page supervision (`relation=高`) and raw/pseudo-label material (`relation=中`).
- Keep `multipage=❌ AND relation=高` in a single-page title-support pool, not the multi-page core.
- Keep unbounded portals, name lists, schema-less composites, and source pointers outside `C1` until a reproducible boundary is frozen.

Report row count, known-family-deduplicated count, and actual document/page volume separately. Deduplicate aliases, mirrors, explicit subsets, and direct derivatives for raw-source coverage while retaining distinct label products for processing workload. Apply any `高=2, 中=1` score only inside `C1` and describe it as a planning weight, never as data volume, label quality, or model benefit.

## Difficult Entries

- For a composite collection, identify it as a composite and classify from documented components; skip bulk download.
- For a search index or benchmark catalog, mark it as index-only and explain why it is not a standalone paired dataset.
- For permission, registration, expired URL, or repeated transport failure, skip downloading after bounded checks and give a decisive classification from available official evidence.
- For a code repository without released samples, do not treat source code or example configuration as labeled data.

## Checkpoint Contract

Write completed chunks incrementally. A checkpoint item must use this shape:

```json
{
  "source_row": 2,
  "dataset": "exact source name",
  "data_type": "文档文件, OCR",
  "input_and_label": "document PDF -> structured Markdown",
  "availability": "可用",
  "multipage": "✅",
  "relation": "高",
  "relation_reason": "Specific title, hierarchy, and page-order evidence for this dataset.",
  "verification": "reused|locally_verified|official_metadata|sample_verified|download_skipped",
  "note": "Evidence source, inspected sample/schema, access limitation, and any processing needed."
}
```

Use the exact key names required by the active generator when they differ; inspect its loader before writing. Keep `skipped` empty when a row can be conclusively classified without downloading. Checkpoint after a small batch instead of waiting for the full range.

When the active deliverable is asset-table correction, keep two outputs separate:

- Completed local assets: use the user-selected path or default to `completed-datasets.tsv`.
- Undownloaded but evidence-backed corrections: use the user-selected path or default to `undownloaded-dataset-field-corrections.tsv`.

Both preserve the active inventory schema, one dataset per row. Never add an undownloaded row to the completed-local TSV.

## Final Audit

- Confirm all assigned stable record IDs occur exactly once and dataset names match the source inventory.
- Confirm no required field is blank and no forbidden placeholder remains.
- Confirm every input/label value names a concrete input. Labeled rows use `input -> label`; missing-label rows contain only the input carrier and set `availability=缺label` unless the higher-priority `仅评测` category applies.
- Confirm availability overlap follows `仅评测 > 缺label > 需处理`.
- Confirm every data type is from the fixed nine-option list and original populated types were reused.
- Recheck every `availability=缺label` row against known public annotation formats.
- Recheck every multi-page `✅` for physical-document or explicit page-order evidence.
- Recheck every `relation=高` for a named direct title signal; do not require or infer multi-page status from relevance.
- Reject page/table/form headers, captions, QA, KIE, and generic OCR as direct document-heading supervision.
- Confirm every `availability=非数据集` row has `relation=不相关`.
- Confirm usability, multi-page status, and relevance were each supported independently.
- Report `C0` and `C1` separately and keep unfrozen sources out of the coverage denominator.
- Detect repeated/template reasons and replace them with dataset-specific evidence.
- Confirm difficult rows remain present with explicit conclusions.
- Confirm temporary validation data has been removed and durable downloaded assets were untouched.
