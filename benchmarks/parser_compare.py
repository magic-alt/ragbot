"""Compare document parsers under the same Ragbot chunking/retrieval harness.

The benchmark changes only the parser implementation. Documents, chunker,
embedding backend, cosine search and top-k remain fixed. The corpus mode accepts
PDFs so PyPDF2, PyMuPDF, Docling and Unstructured can be compared apples-to-apples
with the same Golden Dataset format used by ``rag_framework_compare``.

Examples:
    python -m benchmarks.parser_compare --synthetic-documents 16
    python -m benchmarks.parser_compare \
        --corpus-dir ./data/manuals \
        --golden ./eval/datasets/manuals.json \
        --embedding env \
        --backends pypdf2,pymupdf,docling,unstructured
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from benchmarks.rag_framework_compare import QueryCase, load_golden
from services.api.app.retrieval.embedder import HashEmbedder, build_embedder
from services.worker.parsing import iter_document_segments, parse_document


@dataclass(frozen=True)
class RawDocument:
    doc_id: str
    data: bytes
    path: str
    media_type: str = "application/pdf"


@dataclass(frozen=True)
class IndexedChunk:
    chunk_id: str
    doc_id: str
    path: str
    text: str


def _parser_config(backend: str) -> dict:
    normalized = backend.strip().lower()
    if normalized == "pypdf2":
        return {"provider": "ragbot", "strategy": "pypdf2"}
    if normalized == "pymupdf":
        return {"provider": "pymupdf", "strategy": "blocks"}
    if normalized == "docling":
        return {"provider": "docling", "strategy": "document"}
    if normalized == "unstructured":
        return {"provider": "unstructured", "strategy": "elements"}
    raise ValueError(f"Unsupported parser backend: {backend}")


def _normalize(vector: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(float(value) * float(value) for value in vector)) or 1.0
    return [float(value) / norm for value in vector]


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    index = max(0, min(len(ordered) - 1, round((len(ordered) - 1) * q)))
    return ordered[index]


def _matches(case: QueryCase, chunk: IndexedChunk) -> bool:
    if case.expected_doc_ids and chunk.doc_id in case.expected_doc_ids:
        return True
    if case.path_contains and any(value in chunk.path for value in case.path_contains):
        return True
    return False


def _search(
    query_vector: Sequence[float],
    vectors: Sequence[Sequence[float]],
    chunks: Sequence[IndexedChunk],
    top_k: int,
) -> list[tuple[IndexedChunk, float]]:
    query = _normalize(query_vector)
    scored = [
        (chunk, sum(q * d for q, d in zip(query, vector)))
        for chunk, vector in zip(chunks, vectors)
    ]
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[:top_k]


def load_pdf_corpus(directory: Path) -> list[RawDocument]:
    documents: list[RawDocument] = []
    for path in sorted(item for item in directory.rglob("*.pdf") if item.is_file()):
        relative = path.relative_to(directory).as_posix()
        documents.append(
            RawDocument(
                doc_id=relative,
                path=relative,
                data=path.read_bytes(),
            )
        )
    if not documents:
        raise ValueError(f"No PDF documents found under {directory}")
    return documents


def synthetic_pdf_dataset(documents: int, queries: int) -> tuple[list[RawDocument], list[QueryCase]]:
    try:
        import pymupdf
    except ImportError as exc:  # pragma: no cover - optional benchmark dependency
        raise RuntimeError(
            "Synthetic parser benchmark requires PyMuPDF; install ragbot[parser-pymupdf]"
        ) from exc

    result: list[RawDocument] = []
    for index in range(documents):
        marker = f"RBPARSER{index:05d}"
        pdf = pymupdf.open()
        page = pdf.new_page()
        lines = [
            f"{marker} servo commissioning manual {index}",
            f"Current-loop bandwidth is {800 + index * 5} Hz.",
            f"EtherCAT watchdog timeout is {20 + index} ms.",
            "The next section documents encoder alignment and fault recovery.",
        ]
        y = 72
        for line in lines:
            page.insert_text((72, y), line)
            y += 24
        second = pdf.new_page()
        second.insert_text((72, 72), f"{marker} diagnostic appendix for document {index}")
        data = pdf.tobytes()
        pdf.close()
        doc_id = f"synthetic-{index:05d}.pdf"
        result.append(RawDocument(doc_id=doc_id, path=doc_id, data=data))

    selected = list(range(min(documents, queries)))
    if queries < documents and queries > 1:
        selected = sorted({round(i * (documents - 1) / (queries - 1)) for i in range(queries)})
    cases = [
        QueryCase(
            case_id=f"parser-query-{index:05d}",
            query=f"RBPARSER{index:05d} EtherCAT watchdog timeout document {index}",
            expected_doc_ids=(f"synthetic-{index:05d}.pdf",),
        )
        for index in selected
    ]
    return result, cases


def run_backend(
    backend: str,
    documents: Sequence[RawDocument],
    cases: Sequence[QueryCase],
    *,
    chunk_size: int,
    chunk_overlap: int,
    top_k: int,
    embedder,
) -> dict:
    tracemalloc.start()
    started = time.perf_counter()
    parser_seconds = 0.0
    chunking_seconds = 0.0
    block_count = 0
    page_blocks = 0
    bbox_blocks = 0
    table_blocks = 0
    indexed: list[IndexedChunk] = []

    parser_config = _parser_config(backend)
    for document in documents:
        parse_started = time.perf_counter()
        normalized, parser_metadata = parse_document(
            document.data,
            parser_config,
            name=Path(document.path).name,
            media_type=document.media_type,
            uri=document.path,
        )
        parser_seconds += time.perf_counter() - parse_started
        block_count += len(normalized.blocks)
        page_blocks += sum(1 for block in normalized.blocks if block.page is not None)
        bbox_blocks += sum(1 for block in normalized.blocks if block.bbox is not None)
        table_blocks += sum(1 for block in normalized.blocks if "table" in block.kind)

        chunk_started = time.perf_counter()
        for index, segment in enumerate(
            iter_document_segments(
                normalized,
                None,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        ):
            indexed.append(
                IndexedChunk(
                    chunk_id=f"{document.doc_id}::{index}",
                    doc_id=document.doc_id,
                    path=document.path,
                    text=segment.text,
                )
            )
        chunking_seconds += time.perf_counter() - chunk_started

    if not indexed:
        raise RuntimeError(f"Parser backend {backend} produced zero chunks")

    embed_started = time.perf_counter()
    vectors = [_normalize(vector) for vector in embedder.embed_batch([chunk.text for chunk in indexed])]
    embedding_seconds = time.perf_counter() - embed_started

    latencies_ms: list[float] = []
    first_ranks: list[int | None] = []
    for case in cases:
        query_started = time.perf_counter()
        embed_query = getattr(embedder, "embed_query", None)
        query_vector = embed_query(case.query) if callable(embed_query) else embedder.embed(case.query)
        hits = _search(query_vector, vectors, indexed, top_k)
        latencies_ms.append((time.perf_counter() - query_started) * 1000.0)
        rank = next((position for position, (chunk, _score) in enumerate(hits, 1) if _matches(case, chunk)), None)
        first_ranks.append(rank)

    current_bytes, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    labeled = max(1, len(first_ranks))
    metadata = parse_document(
        documents[0].data,
        parser_config,
        name=Path(documents[0].path).name,
        media_type=documents[0].media_type,
    )[1]
    return {
        "backend": backend,
        "parser": {
            "provider": metadata["parser_provider"],
            "strategy": metadata["parser_strategy"],
            "version": metadata["parser_version"],
            "config_hash": metadata["parser_config_hash"],
        },
        "embedding_model": embedder.model_name,
        "embedding_dimension": embedder.dimension,
        "documents": len(documents),
        "queries": len(cases),
        "blocks": block_count,
        "chunks": len(indexed),
        "timing": {
            "parser_seconds": round(parser_seconds, 6),
            "chunking_seconds": round(chunking_seconds, 6),
            "embedding_seconds": round(embedding_seconds, 6),
            "total_seconds": round(time.perf_counter() - started, 6),
            "documents_per_second": round(len(documents) / max(parser_seconds, 1e-9), 3),
            "blocks_per_second": round(block_count / max(parser_seconds, 1e-9), 3),
            "query_latency_ms_p50": round(_percentile(latencies_ms, 0.50), 3),
            "query_latency_ms_p95": round(_percentile(latencies_ms, 0.95), 3),
            "query_latency_ms_mean": round(statistics.fmean(latencies_ms), 3) if latencies_ms else 0.0,
        },
        "quality": {
            "hit_at_1": round(sum(1 for rank in first_ranks if rank is not None and rank <= 1) / labeled, 6),
            "hit_at_3": round(sum(1 for rank in first_ranks if rank is not None and rank <= 3) / labeled, 6),
            "hit_at_5": round(sum(1 for rank in first_ranks if rank is not None and rank <= 5) / labeled, 6),
            "hit_at_10": round(sum(1 for rank in first_ranks if rank is not None and rank <= 10) / labeled, 6),
            "mrr_at_10": round(
                statistics.fmean(1.0 / rank if rank is not None and rank <= 10 else 0.0 for rank in first_ranks),
                6,
            ) if first_ranks else 0.0,
        },
        "structure": {
            "page_block_rate": round(page_blocks / max(block_count, 1), 6),
            "bbox_block_rate": round(bbox_blocks / max(block_count, 1), 6),
            "table_block_rate": round(table_blocks / max(block_count, 1), 6),
        },
        "memory": {
            "tracemalloc_current_bytes": current_bytes,
            "tracemalloc_peak_bytes": peak_bytes,
        },
    }


def _backend_list(raw: str) -> list[str]:
    values = [value.strip().lower() for value in raw.split(",") if value.strip()]
    supported = {"pypdf2", "pymupdf", "docling", "unstructured"}
    unknown = [value for value in values if value not in supported]
    if unknown:
        raise ValueError(f"Unsupported parser benchmark backends: {unknown}")
    return values


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backends", default="pypdf2,pymupdf")
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--chunk-overlap", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--embedding", choices=["hash", "env"], default="hash")
    parser.add_argument("--hash-dimension", type=int, default=256)
    parser.add_argument("--corpus-dir", default="")
    parser.add_argument("--golden", default="")
    parser.add_argument("--synthetic-documents", type=int, default=16)
    parser.add_argument("--synthetic-queries", type=int, default=12)
    parser.add_argument("--output", default="parser-benchmark.json")
    args = parser.parse_args(argv)
    if args.chunk_size < 2:
        parser.error("chunk-size must be >= 2")
    if args.chunk_overlap < 0 or args.chunk_overlap >= args.chunk_size:
        parser.error("chunk-overlap must satisfy 0 <= overlap < chunk-size")
    if args.top_k < 1:
        parser.error("top-k must be >= 1")
    if bool(args.corpus_dir) != bool(args.golden):
        parser.error("corpus-dir and golden must be supplied together")
    return args


def run(args: argparse.Namespace) -> dict:
    if args.corpus_dir:
        documents = load_pdf_corpus(Path(args.corpus_dir))
        cases = load_golden(Path(args.golden))
        dataset_mode = "pdf-corpus"
    else:
        documents, cases = synthetic_pdf_dataset(args.synthetic_documents, args.synthetic_queries)
        dataset_mode = "synthetic-pdf-smoke"

    embedder = build_embedder() if args.embedding == "env" else HashEmbedder(args.hash_dimension)
    backends = _backend_list(args.backends)
    results = [
        run_backend(
            backend,
            documents,
            cases,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            top_k=args.top_k,
            embedder=embedder,
        )
        for backend in backends
    ]
    payload = {
        "methodology": {
            "dataset_mode": dataset_mode,
            "controlled_variables": [
                "same raw PDF bytes",
                "same Ragbot chunker",
                "same chunk budget",
                "same query set",
                "same embedding backend",
                "same cosine search",
                "same top_k",
            ],
            "changed_variable": "parser implementation",
            "note": (
                "Hash embedding is smoke/performance only. Use --embedding env with the production semantic "
                "embedder and a labeled Golden Dataset before promoting a parser default."
            ),
        },
        "configuration": {
            "backends": backends,
            "chunk_size": args.chunk_size,
            "chunk_overlap": args.chunk_overlap,
            "top_k": args.top_k,
            "embedding": args.embedding,
            "embedding_model": embedder.model_name,
            "embedding_dimension": embedder.dimension,
            "documents": len(documents),
            "queries": len(cases),
        },
        "results": results,
    }
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def main() -> int:
    run(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
