"""Small deterministic CJK lexical retrieval regression benchmark.

This is a release regression corpus, not a claim about customer-domain quality.
It runs against the same PostgresRepo FTS path used in production and fails CI
when Recall@5 or MRR falls below the recorded v1 floor.
"""
from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

from services.api.app.retrieval.pg_fts import fts_search
from services.api.app.storage.models import Chunk, Document
from services.api.app.storage.pg_repo import PostgresRepo

FIXTURE = Path(__file__).with_name("fixtures") / "cjk_lexical.json"
RECALL_AT_5_FLOOR = 0.90
MRR_FLOOR = 0.80


def main() -> int:
    dsn = os.getenv("POSTGRES_TEST_DSN") or os.getenv("POSTGRES_DSN")
    if not dsn:
        raise RuntimeError("POSTGRES_TEST_DSN or POSTGRES_DSN is required")

    corpus = json.loads(FIXTURE.read_text(encoding="utf-8"))
    tenant_id = f"cjk-eval-{uuid.uuid4().hex}"
    repo = PostgresRepo(dsn, pool_min=1, pool_max=2)
    try:
        _load_corpus(repo, tenant_id, corpus["documents"])
        recall, mrr = _evaluate(repo, tenant_id, corpus["queries"])
    finally:
        repo.close()

    print(f"CJK lexical benchmark: Recall@5={recall:.3f} MRR={mrr:.3f}")
    if recall < RECALL_AT_5_FLOOR:
        raise AssertionError(f"Recall@5 {recall:.3f} < floor {RECALL_AT_5_FLOOR:.3f}")
    if mrr < MRR_FLOOR:
        raise AssertionError(f"MRR {mrr:.3f} < floor {MRR_FLOOR:.3f}")
    return 0


def _load_corpus(repo: PostgresRepo, tenant_id: str, documents: list[dict]) -> None:
    for index, item in enumerate(documents):
        doc_id = f"{tenant_id}:{item['doc_id']}"
        repo.add_document(
            Document(
                doc_id=doc_id,
                tenant_id=tenant_id,
                source_type="local_fs",
                title=item["doc_id"],
                uri=f"eval://{item['doc_id']}",
                version="1.0",
                doc_updated_at="2026-09-01T00:00:00+00:00",
                ingested_at="2026-09-01T00:00:00+00:00",
                tags=["cjk-eval"],
            )
        )
        repo.add_chunk(
            Chunk(
                chunk_id=f"{tenant_id}:chunk:{index}",
                doc_id=doc_id,
                tenant_id=tenant_id,
                chunk_index=0,
                text=item["text"],
                metadata={"source_type": "local_fs", "tags": ["cjk-eval"], "acl_hash": "public"},
            )
        )


def _evaluate(repo: PostgresRepo, tenant_id: str, queries: list[dict]) -> tuple[float, float]:
    recall_hits = 0
    reciprocal_rank_sum = 0.0
    for item in queries:
        hits = fts_search(
            repo,
            item["query"],
            {"tenant_id": tenant_id, "tags": ["cjk-eval"]},
            top_k=5,
        )
        ranked = [chunk.doc_id.split(":", 1)[1] for chunk, _score in hits]
        relevant = set(item["relevant"])
        if relevant.intersection(ranked):
            recall_hits += 1
        for rank, doc_id in enumerate(ranked, start=1):
            if doc_id in relevant:
                reciprocal_rank_sum += 1.0 / rank
                break

    count = len(queries) or 1
    return recall_hits / count, reciprocal_rank_sum / count


if __name__ == "__main__":
    raise SystemExit(main())
