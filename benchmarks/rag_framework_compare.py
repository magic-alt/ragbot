"""Compare Ragbot, LangChain and LlamaIndex chunking under one retrieval harness.

The benchmark intentionally holds the embedding backend and vector search
implementation constant. This isolates the effect of chunking/segmentation from
framework-specific embedding clients, vector-store wrappers and retrievers.

Examples:
    python -m benchmarks.rag_framework_compare --synthetic-documents 60
    python -m benchmarks.rag_framework_compare \
        --corpus-dir ./data \
        --golden ./eval/datasets/my_framework_golden.json \
        --embedding env \
        --backends ragbot,langchain,llamaindex
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol, Sequence

from services.api.app.retrieval.embedder import HashEmbedder, build_embedder


@dataclass(frozen=True)
class BenchmarkDocument:
    doc_id: str
    text: str
    path: str = ""


@dataclass(frozen=True)
class QueryCase:
    case_id: str
    query: str
    expected_doc_ids: tuple[str, ...] = ()
    path_contains: tuple[str, ...] = ()


@dataclass(frozen=True)
class IndexedChunk:
    chunk_id: str
    doc_id: str
    path: str
    text: str


class Chunker(Protocol):
    name: str

    def split(self, text: str) -> list[str]: ...


class RagbotFixedWindowChunker:
    name = "ragbot-fixed-window"

    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, text: str) -> list[str]:
        segments: list[str] = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            segment = text[start:end].strip()
            if segment:
                segments.append(segment)
            start = end - self.chunk_overlap if end < len(text) else end
        return segments


class LangChainRecursiveChunker:
    name = "langchain-recursive-character"

    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "LangChain benchmark backend requires langchain-text-splitters"
            ) from exc
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

    def split(self, text: str) -> list[str]:
        return [item.strip() for item in self._splitter.split_text(text) if item.strip()]


class LlamaIndexSentenceChunker:
    name = "llamaindex-sentence"

    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        try:
            from llama_index.core.node_parser import SentenceSplitter
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("LlamaIndex benchmark backend requires llama-index-core") from exc

        # SentenceSplitter normally measures tokenizer units while Ragbot and
        # LangChain's baseline here measure characters. ``list`` makes one
        # Unicode code point equal one budget unit so chunk_size/overlap are
        # directly comparable across all three backends.
        self._splitter = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            tokenizer=list,
        )

    def split(self, text: str) -> list[str]:
        nodes = self._splitter.get_nodes_from_documents(_llama_documents(text))
        return [node.get_content().strip() for node in nodes if node.get_content().strip()]


def _llama_documents(text: str):
    try:
        from llama_index.core import Document
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("LlamaIndex benchmark backend requires llama-index-core") from exc
    return [Document(text=text)]


def build_chunker(name: str, chunk_size: int, chunk_overlap: int) -> Chunker:
    normalized = name.strip().lower()
    if normalized == "ragbot":
        return RagbotFixedWindowChunker(chunk_size, chunk_overlap)
    if normalized == "langchain":
        return LangChainRecursiveChunker(chunk_size, chunk_overlap)
    if normalized == "llamaindex":
        return LlamaIndexSentenceChunker(chunk_size, chunk_overlap)
    raise ValueError(f"Unsupported backend: {name}")


def _normalize(vector: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(float(v) * float(v) for v in vector)) or 1.0
    return [float(v) / norm for v in vector]


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    index = max(0, min(len(ordered) - 1, round((len(ordered) - 1) * q)))
    return ordered[index]


def _is_boundary_ending(text: str) -> bool:
    stripped = text.rstrip()
    return not stripped or stripped[-1] in ".!?。！？;；:：\n"


def _load_pdf(path: Path) -> str:
    try:
        from PyPDF2 import PdfReader
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("PDF corpus loading requires the Ragbot worker extra (PyPDF2)") from exc
    pages: list[str] = []
    for page in PdfReader(str(path)).pages:
        text = page.extract_text()
        if text:
            pages.append(text.strip())
    return "\n\n".join(pages)


def load_corpus(directory: Path) -> list[BenchmarkDocument]:
    supported = {".txt", ".md", ".rst", ".pdf"}
    documents: list[BenchmarkDocument] = []
    for path in sorted(item for item in directory.rglob("*") if item.is_file()):
        if path.suffix.lower() not in supported:
            continue
        relative = path.relative_to(directory).as_posix()
        if path.suffix.lower() == ".pdf":
            text = _load_pdf(path)
        else:
            text = path.read_text(encoding="utf-8", errors="replace")
        if text.strip():
            documents.append(BenchmarkDocument(doc_id=relative, text=text, path=relative))
    if not documents:
        raise ValueError(f"No supported documents found under {directory}")
    return documents


def load_golden(path: Path) -> list[QueryCase]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = payload.get("cases", payload if isinstance(payload, list) else [])
    result: list[QueryCase] = []
    for index, item in enumerate(cases):
        relevance = item.get("relevance", {})
        query = str(item.get("query", "")).strip()
        if not query:
            continue
        result.append(
            QueryCase(
                case_id=str(item.get("id") or f"case-{index:04d}"),
                query=query,
                expected_doc_ids=tuple(str(v) for v in relevance.get("doc_ids", []) if v),
                path_contains=tuple(
                    [str(relevance["path_contains"])]
                    if isinstance(relevance.get("path_contains"), str)
                    else [str(v) for v in relevance.get("path_contains", []) if v]
                ),
            )
        )
    if not result:
        raise ValueError(f"No query cases found in {path}")
    if not any(case.expected_doc_ids or case.path_contains for case in result):
        raise ValueError(
            "Framework comparison requires doc-level labels via relevance.doc_ids or path_contains"
        )
    return result


def synthetic_dataset(documents: int, queries: int) -> tuple[list[BenchmarkDocument], list[QueryCase]]:
    docs: list[BenchmarkDocument] = []
    for index in range(documents):
        marker = f"RBFRAMEWORK{index:05d}"
        paragraphs = [
            f"{marker}. Engineering note {index} covers deterministic retrieval validation.",
            "The subsystem uses bounded queues, idempotent writes, telemetry and failure isolation.",
            (
                f"The unique control fact is retry budget {3 + index % 11} with timeout "
                f"{20 + (index * 7) % 180} milliseconds for document {index}."
            ),
            "A good chunk boundary should preserve related sentences and avoid arbitrary mid-word cuts.",
        ]
        docs.append(
            BenchmarkDocument(
                doc_id=f"synthetic-{index:05d}.txt",
                path=f"synthetic-{index:05d}.txt",
                text=("\n\n".join(paragraphs) + "\n") * 8,
            )
        )

    selected = list(range(min(documents, queries)))
    if queries < documents and queries > 1:
        selected = sorted(
            {round(i * (documents - 1) / (queries - 1)) for i in range(queries)}
        )
    cases = [
        QueryCase(
            case_id=f"synthetic-query-{index:05d}",
            query=f"RBFRAMEWORK{index:05d} retry budget timeout document {index}",
            expected_doc_ids=(f"synthetic-{index:05d}.txt",),
        )
        for index in selected
    ]
    return docs, cases


def _split_documents(documents: Sequence[BenchmarkDocument], chunker: Chunker) -> list[IndexedChunk]:
    chunks: list[IndexedChunk] = []
    for document in documents:
        for index, text in enumerate(chunker.split(document.text)):
            chunks.append(
                IndexedChunk(
                    chunk_id=f"{document.doc_id}::{index}",
                    doc_id=document.doc_id,
                    path=document.path,
                    text=text,
                )
            )
    return chunks


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
    normalized_query = _normalize(query_vector)
    scored = [
        (chunk, sum(q * d for q, d in zip(normalized_query, vector)))
        for chunk, vector in zip(chunks, vectors)
    ]
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[:top_k]


def run_backend(
    backend: str,
    documents: Sequence[BenchmarkDocument],
    cases: Sequence[QueryCase],
    *,
    chunk_size: int,
    chunk_overlap: int,
    top_k: int,
    embedder,
) -> dict:
    tracemalloc.start()
    started = time.perf_counter()
    chunker = build_chunker(backend, chunk_size, chunk_overlap)

    split_started = time.perf_counter()
    chunks = _split_documents(documents, chunker)
    split_seconds = time.perf_counter() - split_started
    if not chunks:
        raise RuntimeError(f"Backend {backend} produced zero chunks")

    embed_started = time.perf_counter()
    vectors = [_normalize(vector) for vector in embedder.embed_batch([chunk.text for chunk in chunks])]
    embed_seconds = time.perf_counter() - embed_started

    latencies_ms: list[float] = []
    first_ranks: list[int | None] = []
    for case in cases:
        query_started = time.perf_counter()
        embed_query = getattr(embedder, "embed_query", None)
        query_vector = embed_query(case.query) if callable(embed_query) else embedder.embed(case.query)
        hits = _search(query_vector, vectors, chunks, top_k)
        latencies_ms.append((time.perf_counter() - query_started) * 1000.0)
        rank = next((index for index, (chunk, _score) in enumerate(hits, 1) if _matches(case, chunk)), None)
        first_ranks.append(rank)

    current_bytes, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    chunk_lengths = [len(chunk.text) for chunk in chunks]
    labeled = max(1, len(first_ranks))
    result = {
        "backend": backend,
        "chunker": chunker.name,
        "embedding_model": embedder.model_name,
        "embedding_dimension": embedder.dimension,
        "documents": len(documents),
        "queries": len(cases),
        "chunks": len(chunks),
        "timing": {
            "split_seconds": round(split_seconds, 6),
            "embedding_seconds": round(embed_seconds, 6),
            "total_seconds": round(time.perf_counter() - started, 6),
            "split_chunks_per_second": round(len(chunks) / max(split_seconds, 1e-9), 3),
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
        "chunk_shape": {
            "chars_mean": round(statistics.fmean(chunk_lengths), 3),
            "chars_p50": round(_percentile(chunk_lengths, 0.50), 3),
            "chars_p95": round(_percentile(chunk_lengths, 0.95), 3),
            "chars_max": max(chunk_lengths),
            "non_boundary_end_rate": round(
                sum(1 for chunk in chunks if not _is_boundary_ending(chunk.text)) / len(chunks), 6
            ),
        },
        "memory": {
            "tracemalloc_current_bytes": current_bytes,
            "tracemalloc_peak_bytes": peak_bytes,
        },
    }
    return result


def _backend_list(raw: str) -> list[str]:
    values = [value.strip().lower() for value in raw.split(",") if value.strip()]
    supported = {"ragbot", "langchain", "llamaindex"}
    unknown = [value for value in values if value not in supported]
    if unknown:
        raise ValueError(f"Unsupported benchmark backends: {unknown}")
    return values


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backends", default="ragbot,langchain,llamaindex")
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--chunk-overlap", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--embedding", choices=["hash", "env"], default="hash")
    parser.add_argument("--hash-dimension", type=int, default=256)
    parser.add_argument("--corpus-dir", default="")
    parser.add_argument("--golden", default="")
    parser.add_argument("--synthetic-documents", type=int, default=60)
    parser.add_argument("--synthetic-queries", type=int, default=30)
    parser.add_argument("--output", default="rag-framework-benchmark.json")
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
        documents = load_corpus(Path(args.corpus_dir))
        cases = load_golden(Path(args.golden))
        dataset_mode = "corpus"
    else:
        documents, cases = synthetic_dataset(args.synthetic_documents, args.synthetic_queries)
        dataset_mode = "synthetic-smoke"

    embedder = build_embedder() if args.embedding == "env" else HashEmbedder(args.hash_dimension)
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
        for backend in _backend_list(args.backends)
    ]
    payload = {
        "methodology": {
            "dataset_mode": dataset_mode,
            "controlled_variables": [
                "same documents",
                "same query set",
                "same embedding backend",
                "same cosine search implementation",
                "same top_k",
                "character-equivalent chunk budget",
            ],
            "changed_variable": "chunker implementation",
            "note": (
                "Hash embedding is for deterministic smoke/performance only. Use --embedding env "
                "with a real semantic embedding model for retrieval-quality conclusions."
            ),
        },
        "configuration": {
            "backends": _backend_list(args.backends),
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
