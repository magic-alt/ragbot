"""Large PDF corpus benchmark for Ragbot.

This benchmark generates deterministic software/engineering PDFs, ingests them
through the real PDF parser + Source pipeline into PostgreSQL and Qdrant, then
measures retrieval quality/latency and unchanged re-ingestion reuse.

Example:
    python -m benchmarks.pdf_scale --documents 1000 --queries 100
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import statistics
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from services.api.app.retrieval.embedder import HashEmbedder
from services.api.app.retrieval.qdrant import QdrantClientAdapter
from services.api.app.retrieval.service import Retriever
from services.api.app.storage.migrations import apply_migrations
from services.api.app.storage.models import Source
from services.api.app.storage.pg_repo import PostgresRepo
from services.worker.pipeline import run_ingest_pipeline

TENANT_ID = "benchmark-pdf-scale"
TAG = "scale-benchmark"
CATEGORIES = (
    "distributed-systems",
    "databases",
    "networking",
    "operating-systems",
    "compilers",
    "cybersecurity",
    "observability",
    "robotics",
    "motor-control",
    "embedded-systems",
)


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(round((len(ordered) - 1) * percentile))))
    return ordered[index]


def _make_pdf(path: Path, document_index: int, pages: int) -> None:
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
    except ImportError as exc:  # pragma: no cover - workflow dependency guard
        raise RuntimeError("reportlab is required for the PDF scale benchmark") from exc

    marker = f"ENGINEERING_DOC_{document_index:06d}"
    category = CATEGORIES[document_index % len(CATEGORIES)]
    c = canvas.Canvas(str(path), pagesize=letter, pageCompression=1)
    width, height = letter

    for page_index in range(pages):
        text = c.beginText(42, height - 48)
        text.setFont("Helvetica", 8.5)
        lines = [
            f"{marker} | Software and Engineering Reference | category={category}",
            f"Document index {document_index}; page {page_index + 1} of {pages}.",
            "Purpose: deterministic RAG corpus for indexing, retrieval and capacity validation.",
        ]
        # Enough distinct technical prose to produce several realistic chunks per PDF.
        for section in range(34):
            retry_budget = 3 + ((document_index + section) % 8)
            window_ms = 10 + ((document_index * 7 + section * 3) % 190)
            queue_depth = 32 + ((document_index + section * 11) % 480)
            lines.append(
                f"{marker} section {section:02d}: {category} design discusses service boundaries, "
                f"backpressure, retry budget {retry_budget}, timeout window {window_ms} ms, "
                f"queue depth {queue_depth}, idempotency, telemetry, fault isolation, capacity "
                "planning, rollout safety, testing strategy, interface contracts and recovery."
            )
        for line in lines:
            # ReportLab does not wrap automatically. Fixed-width slices preserve searchable text.
            for offset in range(0, len(line), 105):
                text.textLine(line[offset : offset + 105])
        c.drawText(text)
        c.showPage()
    c.save()


def _generate_corpus(directory: Path, documents: int, pages: int) -> list[Path]:
    paths: list[Path] = []
    for index in range(documents):
        path = directory / f"engineering-{index:06d}.pdf"
        _make_pdf(path, index, pages)
        paths.append(path)
    return paths


def _query_indexes(documents: int, query_count: int) -> list[int]:
    if query_count >= documents:
        return list(range(documents))
    return sorted({round(i * (documents - 1) / max(1, query_count - 1)) for i in range(query_count)})


def _postgres_database_size(dsn: str) -> int:
    import psycopg

    with psycopg.connect(dsn) as conn:
        row = conn.execute("SELECT pg_database_size(current_database())").fetchone()
    return int(row[0])


def run_benchmark(args: argparse.Namespace) -> dict:
    dsn = args.postgres_dsn or os.getenv("POSTGRES_TEST_DSN") or os.getenv("POSTGRES_DSN", "")
    if not dsn:
        raise RuntimeError("POSTGRES_TEST_DSN or POSTGRES_DSN is required")
    qdrant_url = args.qdrant_url or os.getenv("QDRANT_URL", "http://127.0.0.1:6333")
    collection = args.collection or os.getenv("QDRANT_COLLECTION", f"ragbot_pdf_scale_{int(time.time())}")

    apply_migrations(dsn)
    repo = PostgresRepo(dsn, pool_min=2, pool_max=8)
    embedder = HashEmbedder(dim=args.dimension)
    qdrant = QdrantClientAdapter(
        url=qdrant_url,
        api_key=os.getenv("QDRANT_API_KEY") or None,
        collection_name=collection,
        dim=args.dimension,
    )
    retriever = Retriever(repo, qdrant, embedder=embedder)

    started = time.perf_counter()
    failures: list[dict] = []
    sources: list[Source] = []
    total_chunks = 0

    try:
        with tempfile.TemporaryDirectory(prefix="ragbot-pdf-scale-") as temp_dir:
            corpus_dir = Path(temp_dir)
            generation_start = time.perf_counter()
            pdf_paths = _generate_corpus(corpus_dir, args.documents, args.pages_per_document)
            generation_seconds = time.perf_counter() - generation_start

            ingest_start = time.perf_counter()
            for index, pdf_path in enumerate(pdf_paths):
                source = Source(
                    source_id=f"bench-pdf-source-{index:06d}",
                    tenant_id=TENANT_ID,
                    source_type="pdf",
                    name=f"Engineering document {index:06d}",
                    config={
                        "path": str(pdf_path),
                        "doc_id": f"bench-pdf-doc-{index:06d}",
                        "version": "1.0",
                        "chunk_size": args.chunk_size,
                        "chunk_overlap": args.chunk_overlap,
                    },
                    tags=[TAG, CATEGORIES[index % len(CATEGORIES)]],
                )
                repo.add_source(source)
                job = run_ingest_pipeline(
                    source,
                    repo,
                    qdrant,
                    job_id=f"bench-ingest-{index:06d}",
                    embedder=embedder,
                )
                if job.status != "completed":
                    failures.append({"index": index, "error": job.error})
                    continue
                total_chunks += int(job.stats.get("chunks_total", 0))
                sources.append(source)
                if (index + 1) % 100 == 0 or index + 1 == args.documents:
                    print(
                        f"ingest progress: {index + 1}/{args.documents} PDFs, "
                        f"chunks={total_chunks}, failures={len(failures)}",
                        flush=True,
                    )
            ingestion_seconds = time.perf_counter() - ingest_start

            filters = {
                "tenant_id": TENANT_ID,
                "source_types": ["pdf"],
                "tags": [TAG],
                "security_scope": ["public"],
            }
            latencies_ms: list[float] = []
            reciprocal_ranks: list[float] = []
            hits_at_5 = 0
            query_indexes = _query_indexes(args.documents, args.queries)
            for index in query_indexes:
                marker = f"ENGINEERING_DOC_{index:06d}"
                query = f"{marker} backpressure retry budget engineering reference"
                query_start = time.perf_counter()
                results = retriever.retrieve(query, filters, top_k=5)
                latencies_ms.append((time.perf_counter() - query_start) * 1000.0)
                expected = f"bench-pdf-doc-{index:06d}"
                ranked_docs = [result.doc_id for result in results]
                if expected in ranked_docs:
                    rank = ranked_docs.index(expected) + 1
                    hits_at_5 += 1
                    reciprocal_ranks.append(1.0 / rank)
                else:
                    reciprocal_ranks.append(0.0)

            recall_at_5 = hits_at_5 / max(1, len(query_indexes))
            mrr = statistics.fmean(reciprocal_ranks) if reciprocal_ranks else 0.0

            reuse_sample = sources[: min(args.reingest_documents, len(sources))]
            reused = 0
            rewritten = 0
            reingest_start = time.perf_counter()
            for index, source in enumerate(reuse_sample):
                job = run_ingest_pipeline(
                    source,
                    repo,
                    qdrant,
                    job_id=f"bench-reingest-{index:06d}",
                    embedder=embedder,
                )
                if job.status != "completed":
                    failures.append({"reingest": index, "error": job.error})
                    continue
                reused += int(job.stats.get("chunks_reused", 0))
                rewritten += int(job.stats.get("chunks_ingested", 0))
            reingest_seconds = time.perf_counter() - reingest_start
            reuse_rate = reused / max(1, reused + rewritten)

        qdrant_points = qdrant.count()
        result = {
            "configuration": {
                "documents": args.documents,
                "pages_per_document": args.pages_per_document,
                "queries": len(query_indexes),
                "reingest_documents": len(reuse_sample),
                "chunk_size": args.chunk_size,
                "chunk_overlap": args.chunk_overlap,
                "embedding": embedder.model_name,
                "dimension": args.dimension,
                "qdrant_collection": collection,
            },
            "ingestion": {
                "pdf_generation_seconds": round(generation_seconds, 3),
                "seconds": round(ingestion_seconds, 3),
                "documents_succeeded": len(sources),
                "documents_failed": len(failures),
                "chunks": total_chunks,
                "documents_per_second": round(len(sources) / max(ingestion_seconds, 1e-9), 3),
                "chunks_per_second": round(total_chunks / max(ingestion_seconds, 1e-9), 3),
            },
            "retrieval": {
                "recall_at_5": round(recall_at_5, 6),
                "mrr": round(mrr, 6),
                "latency_ms_p50": round(_percentile(latencies_ms, 0.50), 3),
                "latency_ms_p95": round(_percentile(latencies_ms, 0.95), 3),
                "latency_ms_mean": round(statistics.fmean(latencies_ms) if latencies_ms else 0.0, 3),
            },
            "reingestion": {
                "seconds": round(reingest_seconds, 3),
                "chunks_reused": reused,
                "chunks_rewritten": rewritten,
                "reuse_rate": round(reuse_rate, 6),
            },
            "storage": {
                "qdrant_points": qdrant_points,
                "postgres_database_bytes": _postgres_database_size(dsn),
                "max_rss_kb": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            },
            "failures": failures[:20],
            "total_seconds": round(time.perf_counter() - started, 3),
        }

        if len(sources) != args.documents:
            raise AssertionError(f"Only {len(sources)}/{args.documents} documents ingested")
        if qdrant_points != total_chunks:
            raise AssertionError(f"Qdrant point count {qdrant_points} != chunks {total_chunks}")
        if recall_at_5 < args.min_recall:
            raise AssertionError(f"Recall@5 {recall_at_5:.3f} < {args.min_recall:.3f}")
        if mrr < args.min_mrr:
            raise AssertionError(f"MRR {mrr:.3f} < {args.min_mrr:.3f}")
        if reuse_rate < args.min_reuse:
            raise AssertionError(f"Re-ingestion reuse {reuse_rate:.3f} < {args.min_reuse:.3f}")
        return result
    finally:
        qdrant.close()
        repo.close()


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", type=int, default=1000)
    parser.add_argument("--pages-per-document", type=int, default=2)
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--reingest-documents", type=int, default=50)
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--chunk-overlap", type=int, default=120)
    parser.add_argument("--dimension", type=int, default=128)
    parser.add_argument("--postgres-dsn", default="")
    parser.add_argument("--qdrant-url", default="")
    parser.add_argument("--collection", default="")
    parser.add_argument("--output", default="pdf-scale-results.json")
    parser.add_argument("--min-recall", type=float, default=0.98)
    parser.add_argument("--min-mrr", type=float, default=0.95)
    parser.add_argument("--min-reuse", type=float, default=0.99)
    args = parser.parse_args(argv)
    if args.documents < 1 or args.pages_per_document < 1 or args.queries < 1:
        parser.error("documents, pages-per-document and queries must be positive")
    return args


def main() -> int:
    args = parse_args()
    result = run_benchmark(args)
    output = Path(args.output)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"benchmark result written to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
