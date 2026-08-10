# HWP preprocessing options and experiment plan

Date: 2026-07-10

## Context

The current MVP works, but retrieval quality is weak. The likely causes are
document extraction and chunking quality, especially for Korean `.hwp` files.

Current source dataset extension counts:

```text
1468 pdf
1331 hwp
67 xlsx
47 zip
8 jpg
7 pptx
5 xls
4 jsonl
3 png
3 hwpx
2 doc
1 mp4
1 csv
```

Because `.hwp` files are almost half of the dataset, HWP extraction quality is a
high-leverage area for retrieval improvements. The current parser includes a
custom HWP binary text extraction path, but retrieval quality suggests that
plain text extraction and fixed-size chunking are not enough for Korean public
documents.

## Meeting summary

We should test every serious option, but treat them as different pipeline
roles instead of competing one-for-one replacements.

Recommended position for discussion:

- Hancom Office conversion is likely the strongest quality path for `.hwp`.
- Apache Tika and pyhwp are still worth testing because they are easier to
  automate on non-Windows machines.
- HWPX direct parsing should be implemented because it gives the cleanest
  structure when HWPX is available or when HWP can be converted to HWPX.
- OCR should be a fallback for image-only PDFs and failed conversion outputs,
  not the default path for normal HWP.
- LibreOffice/OpenOffice should stay out of the main pipeline.

The target architecture should be a multi-parser preprocessing pipeline:

```text
source file
-> route by extension/MIME
-> run one or more extractors
-> score extraction quality
-> choose best structured output
-> structure-aware chunking
-> BM25/vector/hybrid index
```

## Excluded option: LibreOffice/OpenOffice

LibreOffice/OpenOffice conversion should be excluded for this project.

Reasons:

- Modern Korean public-sector HWP documents are not reliably supported.
- Conversion can fail silently or produce broken/empty text.
- This makes it risky as a primary RAG preprocessing path.
- It may still be useful as a narrow fallback for very old HWP files, but should
  not be part of the main pipeline.

## Option comparison

| Option | Best use | Expected quality | Automation cost | Main risk |
| --- | --- | --- | --- | --- |
| Hancom Office/HwpCtrl | Primary `.hwp` conversion on Windows | Highest | Medium-high | Windows/license/COM stability |
| HWPX direct parser | `.hwpx` and converted HWPX | High | Medium | Custom parser work |
| Apache Tika | Cross-platform `.hwp` first-pass or fallback | Medium | Low-medium | Structure loss |
| pyhwp/hwp5txt | Python fallback and diagnostics | Medium | Medium | Old project/licensing |
| PaddleOCR | Image-only PDFs and scanned docs | Medium-high for scans | Medium-high | Heavy runtime/OCR noise |
| Tesseract | Simple OCR fallback | Medium-low | Low-medium | Weak layout/table handling |

## Remaining options in detail

### Apache Tika HWP parser

Apache Tika is a Java-based document extraction toolkit. Recent Tika versions
include HWP v5 parser support.

Best role:

- Primary first-pass parser for `.hwp`.
- Useful for quickly testing all 1,331 HWP files and comparing extraction
  success against the current parser.

Pros:

- Good fit for batch processing.
- Supports many document types through one interface.
- Easy to run as CLI or server and call from Python.
- Easier to log extraction length, MIME type, and failure status.

Cons:

- Extracts text more than document structure.
- Tables, headings, footnotes, headers, and layout can be flattened.
- Output may still need strong post-processing for RAG chunking.
- It may parse a file successfully while losing context that matters for search.

Team discussion points:

- Should be tested because it is easy to run over all HWP files quickly.
- Good baseline for "how much does a mature generic parser improve extraction?"
- Less likely to be the final best path if table/section structure matters.
- Useful even if Hancom becomes primary, because it can run on macOS/Linux.

Experiment:

- Run Tika over a representative HWP sample and then the full 1,331 files.
- Log extracted character count, Hangul ratio, paragraph count, and failure
  count.
- Compare against current custom parser output.
- Manually inspect 20 high-value documents with tables/regulations/notices.

### pyhwp / hwp5txt

pyhwp is a Python HWP v5 parser and includes command-line tools such as
`hwp5txt`.

Best role:

- Secondary parser or diagnostic fallback for files where Tika performs poorly.
- Useful for inspecting HWP internals when a file has bad retrieval behavior.

Pros:

- Fits naturally into the existing Python preprocessing pipeline.
- Can expose lower-level HWP structure for debugging.
- Useful for parser comparison and quality scoring.

Cons:

- Older project with dated dependency/runtime assumptions.
- Some conversion paths are experimental.
- AGPL licensing needs review before service distribution.
- Should be validated against project samples before using as the main parser.

Team discussion points:

- Worth trying because it gives a second independent extraction path.
- More useful as a diagnostic/fallback tool than as the only main parser.
- Licensing should be checked before depending on it in a distributed product.

Experiment:

- Run `hwp5txt` or pyhwp extraction over the same HWP sample used for Tika.
- Compare extracted text length and quality against Tika and current parser.
- Identify whether pyhwp recovers files that Tika misses.
- Decide whether pyhwp should be installed in the main pipeline or kept as an
  offline diagnostic tool.

### HWPX direct XML parsing

HWPX is a zipped XML-based format. It can be parsed directly with Python
`zipfile` plus XML parsing.

Best role:

- Dedicated high-quality parser for `.hwpx`.
- Also useful if `.hwp` files are converted to `.hwpx` in a separate workflow.

Pros:

- Better structure preservation than binary HWP parsing.
- Can keep paragraphs, sections, tables, headings, and metadata.
- Good foundation for structure-aware RAG chunking.
- Avoids heavy external runtime dependencies.

Cons:

- Needs custom parser implementation.
- Must handle XML namespaces, table structures, paragraph styles, and controls.
- Current dataset has only 3 `.hwpx` files, so immediate impact is limited.

Team discussion points:

- Current direct impact is small because only 3 files are HWPX.
- Strategic value is high if Hancom conversion can produce HWPX from HWP.
- HWPX gives us the best chance at structure-aware chunks: headings, sections,
  paragraphs, table rows, and appendices.

Experiment:

- Implement a minimal HWPX extractor for the 3 existing files.
- Extract paragraph blocks and table cells separately.
- Preserve block metadata such as section path, paragraph index, and table row.
- If Hancom can batch-convert HWP to HWPX reliably, use this parser for the
  converted output.

### Hancom Office / HwpCtrl / COM automation

This uses Hancom Office on Windows to convert HWP files through the official
renderer/automation interface.

Best role:

- Primary high-quality offline conversion path for `.hwp`, since a Windows
  desktop is available.
- Convert HWP files to HWPX or HTML for downstream structure-aware extraction.
- Keep TXT/PDF/DOCX as secondary conversion formats for comparison.

Pros:

- Highest compatibility with real-world Korean public-sector HWP files.
- Better chance of preserving tables, styles, and special HWP features.
- Converted HTML/HWPX/DOCX can support better structure-aware parsing.

Cons:

- Requires Windows and Hancom Office licensing.
- GUI/COM automation can be fragile in large batch jobs.
- Not ideal for CI, macOS, Linux, or realtime server-side processing.
- Better as an offline preprocessing machine than an app runtime dependency.

Team discussion points:

- This is likely the highest-impact path because the official renderer should
  understand real Korean public-sector HWP files better than generic parsers.
- It should not be an app runtime dependency. It should be an offline batch
  preprocessing step on the Windows desktop.
- Best target format is probably HWPX or HTML, not plain TXT. TXT loses too much
  structure. PDF is useful for visual verification but weaker for structured
  RAG extraction.
- If HWPX conversion is reliable, the final pipeline becomes much cleaner:
  HWP -> HWPX -> XML parser -> structured chunks.

Experiment:

- On Windows, batch-convert a sample of HWP files to HWPX, HTML, TXT, and PDF.
- Compare which output format preserves headings/tables best.
- Track conversion failures, popups, password issues, and corrupted files.
- For the best format, run extraction and chunking on 100-200 sample files.
- If results are strong, run all 1,331 HWP files offline and commit/store the
  converted artifacts or processed JSONL output.

Operational notes:

- Use the Windows desktop as a preprocessing worker, not as part of the web app.
- Store conversion logs and generated artifacts with deterministic paths.
- Make the conversion idempotent: skip files where source hash and output
  manifest already match.
- Keep source hash, converter version, output format, and conversion timestamp
  in a manifest.

### OCR fallback: PaddleOCR / Tesseract

OCR is not an HWP parser, but it is needed for scanned PDFs, images, and
documents with poor text layers.

Best role:

- Fallback for image-only PDFs and image files.
- Last-resort path after structured extraction fails.

PaddleOCR pros:

- Better fit for document AI workflows.
- Can output richer structured formats than plain OCR text.
- More promising for tables and layout-aware extraction.

PaddleOCR cons:

- Heavier install and runtime.
- Model downloads, CPU/GPU performance, and memory need planning.
- Should only run on low-text or image-only documents, not every file.

Tesseract pros:

- Lightweight, mature, and easy to automate.
- Korean language data is available.

Tesseract cons:

- Weaker on complex public-document layout and tables.
- Needs separate layout analysis for high-quality RAG chunks.
- OCR noise can reduce retrieval quality if used too broadly.

Team discussion points:

- OCR is important because there are 1,468 PDFs and some image files.
- It should be selective. Running OCR on every document will add cost and noise.
- First detect whether a PDF has a useful text layer. Only low-text or
  image-only files should go to OCR.
- PaddleOCR is more promising for layout-aware extraction. Tesseract is simpler
  and lighter.

Experiment:

- Detect PDF pages with low embedded text.
- Run PaddleOCR and Tesseract on a small sample of image-only PDFs.
- Compare Korean recognition quality, table handling, runtime, and output
  structure.
- Decide whether OCR output should be indexed directly or only used when no
  better parser output exists.

## Extraction quality scoring

Every parser should write comparable metrics so the pipeline can choose the best
output automatically.

Suggested fields:

- `source_path`
- `source_sha256`
- `extension`
- `parser`
- `parser_version`
- `output_format`
- `status`
- `error_message`
- `extracted_chars`
- `hangul_chars`
- `hangul_ratio`
- `paragraph_count`
- `line_count`
- `table_like_line_count`
- `avg_line_length`
- `control_char_count`
- `source_size_bytes`
- `output_path`
- `created_at`

Basic quality rules:

- Reject or fallback if extracted text is empty or below a minimum length.
- Penalize very low Hangul ratio for Korean institutions.
- Penalize excessive control characters.
- Penalize outputs where most text is one huge line.
- Prefer outputs with preserved paragraph/table boundaries.
- Keep multiple parser outputs during experiments so manual comparison is
  possible.

## Chunking implications

Parser quality alone will not solve retrieval if chunking remains too flat.

Target chunking approach:

- Preserve document title, institution, source path, section title, and page or
  paragraph location in every chunk.
- Split first by structural blocks: headings, article numbers, sections,
  appendices, FAQ items, table rows, and notices.
- Keep table rows/cells together with their header context.
- Only apply overlapping character chunks inside oversized blocks.
- Store chunk type metadata: `paragraph`, `article`, `table`, `appendix`,
  `notice`, `faq`, `ocr`.
- Index Korean-friendly normalized text in addition to original text.

## Proposed experiment sequence

1. Build a gold sample set:
   - 30 HWP from 부산대학교.
   - 30 HWP from 금융감독원.
   - 20 HWP from KISA or 한국해양과학기술원.
   - Include files with tables, regulations, forms, and notices.
2. Run current parser, Tika, pyhwp, and Hancom conversion on the same sample.
3. For Hancom, compare HWPX, HTML, TXT, PDF outputs.
4. Score parser outputs automatically and manually inspect representative files.
5. Implement HWPX/HTML structure extraction for the best Hancom output format.
6. Rebuild chunks and BM25 index from the best output.
7. Evaluate with a fixed query set before changing frontend behavior.

Suggested query evaluation set:

- 30 real user-style Korean questions.
- Include institution-specific queries.
- Include table/value lookup queries.
- Include regulation/procedure queries.
- Measure whether top 5 chunks contain the relevant document and passage.

## Recommended team decision

Given that a Windows desktop is available, Hancom conversion should be treated
as the likely primary path for `.hwp`. The other methods should still be tested
because they are useful as fallbacks and help validate extraction quality.

Recommended target pipeline:

```text
.hwp
-> Hancom batch conversion to HWPX or HTML
-> direct HWPX/HTML structure parser
-> quality scoring
-> structure-aware chunks
-> index

fallbacks:
Hancom failure -> Tika -> pyhwp -> OCR only if converted to image/PDF and needed
```

## Initial pipeline direction

1. Route files by extension and MIME type.
2. For `.hwp`, prioritize Hancom Office offline conversion if the Windows
   desktop workflow is available.
3. Test HWPX and HTML as preferred conversion formats.
4. Keep Tika and pyhwp as automated fallback paths.
5. For `.hwpx`, implement direct XML parsing.
6. For PDFs, extract embedded text first and use OCR only for image-only or
   low-text files.
7. Log extraction quality metrics per file:
   - extracted character count
   - Hangul ratio
   - paragraph count
   - detected table-like content
   - parser used
   - fallback path
   - failure reason

The biggest retrieval improvement is likely to come from combining better HWP
extraction with structure-aware chunking by headings, articles, tables,
appendices, FAQ entries, and only then applying overlapping chunks to oversized
sections.
