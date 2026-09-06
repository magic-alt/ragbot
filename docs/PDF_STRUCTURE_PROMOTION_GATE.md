# PDF Structure Golden Suite and promotion gate

This workflow turns the parser/splitter experiments into an explicit promotion process for Ragbot's PDF ingestion default.

The current control is:

```text
PyPDF2 → raw page blocks → Ragbot fixed-window splitter
```

The current vNext candidate is:

```text
PyMuPDF → page/section-aware block coalescing → LlamaIndex SentenceSplitter
```

The promotion runner does not mutate production configuration. It only produces a source-backed decision artifact: `PROMOTE` or `HOLD`.

## Why a separate Golden Suite

The DeepSeek in Action benchmark is useful for development regression, but it contains one technical book and twelve cases. The factorial benchmark established the component-level causal chain; it is not broad enough to justify a repository-wide PDF default change.

A production suite should cover structurally different PDFs and contain reviewed stable labels. The recommended starting coverage is:

- `technical_book`
- `multi_column_paper`
- `datasheet`
- `table_report`
- `figure_report`
- `cjk_document`
- `mixed_language_document`
- `scanned_document`

Recommended case tags include:

- `numeric`
- `page-specific`
- `paraphrase`
- `table`
- `figure-caption`
- `cross-lingual`

These recommendations are emitted by the suite initializer but remain explicit dataset requirements so a project can tighten or adapt them rather than hiding policy in benchmark code.

## 1. Build a corpus

Create one directory containing the PDFs to validate. Use stable filenames because `doc_id` / `path` relevance labels are derived from paths relative to this corpus root.

Example:

```text
data/pdf-structure-suite/
├── book.pdf
├── paper-two-column.pdf
├── motor-driver-datasheet.pdf
├── annual-report.pdf
├── figure-heavy-report.pdf
├── chinese-manual.pdf
├── bilingual-spec.pdf
└── scanned-manual.pdf
```

Do not commit proprietary PDFs to a public repository. The suite manifest pins their local relative paths and SHA256 hashes, so the labels can remain versioned without storing the source bytes in Git.

## 2. Initialize the authoring skeleton

```bash
./.venv/bin/python scripts/rag_pdf_suite.py init \
  --corpus data/pdf-structure-suite \
  --output eval/datasets/pdf_structure_golden.json
```

The initializer records:

- relative path;
- SHA256;
- page count;
- a `document_type: TODO` placeholder;
- empty language metadata;
- no generated questions.

It deliberately does **not** invent answers or relevance labels.

Fill in the document metadata and author cases directly from the source PDFs. A production case should have stable relevance whenever possible, for example:

```json
{
  "id": "datasheet-current-limit-001",
  "category": "numeric",
  "tags": ["numeric", "page-specific", "datasheet"],
  "query": "What is the absolute maximum continuous output current?",
  "answer": "...source-backed answer...",
  "relevance": {
    "doc_ids": ["motor-driver-datasheet.pdf"],
    "pages": [12],
    "max_rank": 5
  },
  "review": {
    "status": "approved",
    "notes": "Checked against page 12 in the pinned PDF."
  }
}
```

The machine-readable schema is `eval/datasets/pdf_structure_suite.schema.json`.

## 3. Validate the suite before paying embedding cost

```bash
./.venv/bin/python scripts/rag_pdf_suite.py validate \
  --dataset eval/datasets/pdf_structure_golden.json \
  --corpus data/pdf-structure-suite \
  --profile production
```

Production validation composes the existing Golden Dataset production audit with structure-specific checks. By default this means at least 50 cases, at least 80% stable labels, at least three categories, plus the suite's declared document types, case tags, review rate, exact corpus paths and optional SHA256 identity.

A development dataset without a `pdf_structure_suite` manifest may still be validated with `--profile development`; this is intentionally not sufficient for a production promotion decision.

## 4. Install candidate dependencies

```bash
./.venv/bin/python -m pip install -e \
  ".[worker,benchmark-frameworks,parser-pymupdf]"
```

Configure the same semantic embedding model used in the live deployment. For the existing local Qwen3 setup:

```bash
export EMBEDDING_MODEL="qwen3-embedding:8b"
export EMBEDDING_BASE_URL="http://127.0.0.1:11434"
export QDRANT_DIM="4096"
export EMBEDDING_TIMEOUT_SECONDS="300"
export EMBEDDING_BATCH_SIZE="8"
```

## 5. Run Control vs Candidate

```bash
./.venv/bin/python scripts/rag_pdf_promotion.py \
  --dataset eval/datasets/pdf_structure_golden.json \
  --corpus data/pdf-structure-suite \
  --profile production \
  --embedding env \
  --chunk-size 800 \
  --chunk-overlap 100 \
  --coalesce-target-multiplier 4 \
  --top-k 10 \
  --repetitions 3
```

Reports are written to:

```text
reports/pdf-promotion/
├── pdf-promotion-<timestamp>.json
├── pdf-promotion-<timestamp>.md
├── latest.json
└── latest.md
```

Exit code is `0` for `PROMOTE`, `1` for `HOLD`, and `2` for invalid input/runtime setup. Use `--no-fail-on-gate` when collecting exploratory evidence without making the gate fail a shell or CI job.

## Default promotion contract

The candidate must satisfy all default checks:

| Check | Default |
| --- | ---: |
| Suite audit | pass |
| Δ Hit@5 | >= 0.00 |
| Δ MRR@10 | >= -0.02 |
| Δ nDCG@10 | >= -0.02 |
| Per-category Δ MRR | >= -0.05 |
| Candidate label-text coverage | >= 99% |
| Δ page metadata coverage | >= -0.5 pp |
| Chunk-count ratio | <= 1.15x |
| Embedding-time ratio | <= 1.15x |
| Peak Python allocation ratio | <= 1.25x |
| Non-boundary-end ratio | <= 0.50x control |
| New retrieval failures | 0 |

The JSON report also records per-case top-three hits and rank regressions. Rank regression is diagnostic by default; a case becomes blocking when it crosses its declared `max_rank` and therefore becomes a new retrieval failure.

## Promotion sequence

A successful offline result should advance through these stages:

1. **Component evidence** — completed by splitter/parser and factorial benchmarks.
2. **PDF Structure Golden Suite** — this gate; diverse source-pinned corpus and reviewed labels.
3. **Isolated production-store A/B** — fresh Qdrant/PostgreSQL collections with identical corpus scope; compare index wall time, vector count, FTS/RRF and reranker behavior.
4. **Live shadow/canary validation** — citation fidelity, ACL/multi-tenant semantics, operational errors and latency under the normal API path.
5. **Default migration** — change the default parser/chunker contract only after the prior gates pass, with an explicit release note because chunk identities change and existing sources require re-indexing to adopt the new pipeline.

A `PROMOTE` result from this script means "eligible for the next gate", not "production default has already changed".
