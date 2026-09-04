# Parser Port and Normalized Document Model

Ragbot owns the production ingestion lifecycle but does not need to reimplement mature document parsing algorithms. Parser Port separates acquisition, parsing and chunking so connectors remain transport-oriented while parsing libraries can be benchmarked and replaced independently.

## Architecture

```text
Source / durable Job / generation fence
        -> Connector
             fetch bytes + MIME + URI + remote version
        -> Parser Port
             -> ragbot/text
             -> ragbot/html
             -> ragbot/pypdf2        (PDF compatibility default)
             -> pymupdf/blocks
             -> docling/document
             -> unstructured/elements
        -> NormalizedDocument
             -> DocumentBlock[]
                  text
                  kind
                  page (1-based)
                  section
                  bbox
                  parser metadata
        -> Chunker Port
             -> ragbot/fixed | structural
             -> langchain/recursive | code
             -> llamaindex/sentence
        -> Ragbot Chunk
        -> reuse / embedding / PostgreSQL + Qdrant
```

The external parser frameworks do **not** own Source lifecycle, retries, DLQ, ACL, tenant isolation, embedding publication, retrieval fusion, Agent behavior or citations. They are adapters behind `services/worker/parsing/`.

## Parser index contract

Each parsed chunk carries a stable parser identity:

- `parser_provider`
- `parser_strategy`
- `parser_version`
- `parser_config_hash`

`parser_config_hash` covers provider, strategy, version and canonical parser options. The ingestion reuse key includes the parser identity. Changing parser implementation or parser options therefore forces the affected content to be reparsed and re-embedded instead of silently reusing stale chunks.

Drive metadata-first reuse also checks parser identity before skipping a remote download. This is required because an unchanged remote file may still need rebuilding when the local parsing strategy changes.

## Source configuration

PDF using the current compatibility parser:

```json
{
  "path": "/data/manual.pdf",
  "parsing": {
    "provider": "ragbot",
    "strategy": "pypdf2",
    "version": 1
  }
}
```

PDF using PyMuPDF block extraction:

```json
{
  "path": "/data/manual.pdf",
  "parsing": {
    "provider": "pymupdf",
    "strategy": "blocks",
    "version": 1,
    "options": {
      "sort": true
    }
  }
}
```

Docling for a local Office corpus:

```json
{
  "path": "/data/office",
  "extensions": [".docx", ".pptx", ".xlsx"],
  "parsing": {
    "provider": "docling",
    "strategy": "document",
    "version": 1
  }
}
```

Unstructured is available through:

```json
{
  "parsing": {
    "provider": "unstructured",
    "strategy": "elements",
    "version": 1
  }
}
```

Parser configuration is validated in the Source control plane. `repo`, `notion` and `confluence` do not currently accept `config.parsing` because those ingestion paths already consume structured/provider-native content rather than generic document bytes.

## Dependencies

Parser dependencies are deliberately isolated:

```bash
pip install -e ".[worker,parser-pymupdf]"
pip install -e ".[worker,parser-docling]"
pip install -e ".[worker,parser-unstructured]"
```

The default worker dependency set remains small. Selecting a parser whose optional dependency is not installed fails explicitly with installation guidance.

## Defaults and migration policy

- PDF keeps `ragbot/pypdf2` as the compatibility default until a real Golden Dataset justifies promotion.
- HTML defaults to Ragbot's small BeautifulSoup-based structural adapter.
- plain text / Markdown defaults to `ragbot/text`; Markdown heading recovery remains available at the chunk boundary.
- Office resources explicitly included in a Source default to `docling/document` because binary Office data cannot safely fall back to UTF-8 text decoding.
- local filesystem and S3 default extension sets are not automatically widened during this migration. Add Office/PDF extensions explicitly when desired.

Changing a parser contract is an index-contract change. Expect affected documents to rebuild once.

## Promotion benchmark

Use the same Golden Dataset relevance shape as the existing framework benchmark:

```bash
python -m benchmarks.parser_compare \
  --corpus-dir ./data/manuals \
  --golden ./eval/datasets/manuals.json \
  --embedding env \
  --backends pypdf2,pymupdf,docling,unstructured \
  --chunk-size 800 \
  --chunk-overlap 100 \
  --output parser-benchmark.json
```

The controlled PDF benchmark holds constant:

- raw PDF bytes;
- Ragbot chunker and chunk budget;
- query set;
- embedding backend;
- cosine retrieval implementation;
- `top_k`.

Only the parser changes.

The benchmark reports parsing throughput, memory, block/page/bbox/table structure, Hit@K, MRR and query latency. Hash embeddings are only a deterministic smoke test. Parser promotion decisions must use the production semantic embedding and a labeled real corpus.

### Recommended promotion gate

Do not promote a new default because it is more feature-rich. Promote only when a representative Golden Dataset demonstrates that the candidate:

1. does not regress Hit@5 / MRR materially;
2. improves page/citation provenance and layout-sensitive evidence where relevant;
3. improves table or multi-column retrieval for the target corpus where relevant;
4. stays within an explicit indexing latency and peak-memory budget;
5. remains deterministic enough that unchanged documents produce stable parser/chunker identities and predictable reindex behavior.

PyMuPDF is the first candidate for the default PDF fast path because it provides block bounding boxes with a comparatively small dependency footprint. Docling and Unstructured should be evaluated primarily on complex layout, table and Office-document corpora before broader promotion.

## Relationship to staged generations

Parser Port solves transformation ownership and index identity. It does not make PostgreSQL and Qdrant publication atomic. The next architecture layer is staged knowledge generations: write a complete candidate generation invisibly, verify both stores, then atomically switch an authoritative PostgreSQL active-generation pointer and clean old generations asynchronously through an outbox/reconciler.
