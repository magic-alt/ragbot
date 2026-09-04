# Document transformation architecture

Ragbot owns the production knowledge lifecycle. LangChain and LlamaIndex are
optional document-transformation providers, not orchestration runtimes.

## Architecture boundary

```text
Source / durable Job / lease / retry / DLQ / fencing
                    |
                Connector
                    |
             normalized text
                    |
             Ragbot Chunker port
       +------------+-------------+
       |            |             |
 ragbot/*     langchain/*    llamaindex/*
       |            |             |
       +------------+-------------+
                    |
                  Chunk
                    |
       dedup / incremental reuse
                    |
               Embedder port
                    |
          PostgreSQL + Qdrant
```

The framework adapters must not own:

- Source lifecycle or durable Job semantics;
- tenant/RBAC/ACL filtering;
- incremental SaaS version checks;
- PostgreSQL/Qdrant publication and stale-data cleanup;
- adaptive vector/lexical fusion;
- Agent routing, citations or observability.

This keeps framework API churn behind `services/worker/chunking/`.

## Installation

The default `ragbot/fixed` and repository `ragbot/structural` strategies require
no new dependency.

Install optional framework adapters with:

```bash
pip install -e ".[worker,document-transformers]"
```

`document-transformers` currently installs:

- `langchain-text-splitters`;
- `llama-index-core`.

## Source configuration

Existing top-level fields remain supported:

```json
{
  "chunk_size": 800,
  "chunk_overlap": 100
}
```

A Source can opt into a provider with a nested `chunking` object:

```json
{
  "chunk_size": 800,
  "chunk_overlap": 100,
  "chunking": {
    "provider": "langchain",
    "strategy": "recursive",
    "version": 1
  }
}
```

Nested `chunking.chunk_size` and `chunking.chunk_overlap` take precedence over
legacy top-level values when supplied.

### Generic text, Web, PDF, SaaS

Supported strategies:

```text
ragbot/fixed
langchain/recursive
llamaindex/sentence
```

Examples:

```json
{
  "chunking": {
    "provider": "langchain",
    "strategy": "recursive",
    "chunk_size": 900,
    "chunk_overlap": 120
  }
}
```

```json
{
  "chunking": {
    "provider": "llamaindex",
    "strategy": "sentence",
    "chunk_size": 900,
    "chunk_overlap": 120
  }
}
```

LlamaIndex `sentence` deliberately uses the existing Ragbot character budget so
old Source configuration keeps one unit model. A future token-budget strategy
must use a new strategy/version and therefore a distinct index contract.

### Git repositories

Repository ingestion keeps `ragbot/structural` as the default because it already
preserves Python symbols and C-like function boundaries.

To delegate code splitting to LangChain:

```json
{
  "chunking": {
    "provider": "langchain",
    "strategy": "code",
    "chunk_size": 1000,
    "chunk_overlap": 100
  }
}
```

Ragbot detects common source-file languages and asks LangChain for its
language-aware separators. Unknown languages fall back to LangChain recursive
splitting rather than adding another Ragbot-maintained separator table.

## Index contract

Every new Chunk records:

```text
parser_provider
parser_version
chunker_provider
chunker_strategy
chunker_version
chunker_config_hash
chunker_language (when applicable)
chunk_size
chunk_overlap
embedding_model
embedding_dimension
```

The chunker hash is deterministic over provider, strategy, version, size,
overlap and language. Pipeline reuse includes parser/chunker identity in addition
to content checksum, ACL/version and embedding identity.

Therefore all of the following are index-contract changes:

- parser provider/version change;
- chunker provider/strategy/version change;
- chunk size or overlap change;
- language-aware code strategy change;
- embedding model/dimension change.

Changed contracts are reprocessed instead of silently reusing stale vectors.

## Incremental SaaS behavior

Drive, Notion and Confluence perform metadata-first refresh. Before this change,
an unchanged remote version could reuse old chunks even after local chunking
configuration changed.

They now require both:

```text
remote version unchanged
AND
chunker contract unchanged
```

Legacy chunks do not contain explicit chunker identity, so the first scheduled
sync after upgrading this architecture intentionally rebuilds them once. This is
a correctness migration, not a recurring cost. Later unchanged syncs reuse the
new explicit contract normally.

## PDF page identity

PDF parsing now preserves the original 1-based page number before chunking.
Local/remote PDF Sources produce page-scoped chunks and retrieval citations can
include:

```text
doc-id:path/to/file.pdf:page=17:chunk=42
```

This improves page-labelled Golden Dataset evaluation and citation inspection.
Because old PDF chunks were created after flattening all pages into one string,
PDF ingestion after this upgrade is an index-contract change and should be
allowed to rebuild the Source.

S3/Drive PDF extraction still flattens downloaded PDF bytes internally; making
those remote PDF parsers block/page-aware is a follow-up parser adapter task.

## Retrieval performance changes

The query kernel remains Ragbot-owned. Hybrid retrieval now fans out the two
independent first-stage branches concurrently:

```text
                 +-> query embedding -> Qdrant --+
query -----------+                                +-> adaptive RRF -> rerank
                 +-> PostgreSQL FTS/CJK ----------+
```

The reranker also consumes the Chunk objects already returned by PostgreSQL FTS
or text already present in Qdrant payloads. It no longer performs one
`repo.get_chunk()` call per candidate.

These changes preserve the existing candidate pool, adaptive fusion policy,
security filters and reranker contract while reducing avoidable latency/I/O.

## Benchmark and promotion policy

Do not promote a framework strategy because it is more popular or produces
cleaner-looking chunks. Use the controlled benchmark in
`benchmarks/rag_framework_compare.py`, followed by the live Golden Dataset in
`scripts/rag_eval.py`.

Recommended promotion sequence:

```text
controlled splitter benchmark
        -> real corpus Golden Dataset
        -> vector/hybrid ablation
        -> reranker off/on
        -> answer/citation evaluation
        -> production opt-in
```

Keep the embedding model, corpus, top-k and retrieval mode fixed when comparing
splitters. A Source should move from `ragbot/*` to a framework adapter only when
its own quality/latency/cost measurements justify the reindex.
