"""Controlled local benchmarks for Ragbot splitter and parser components.

This module intentionally avoids whole-framework comparisons. It keeps the
embedding model, cosine retrieval implementation, Golden Dataset and top-k fixed
while changing exactly one local component at a time:

- splitter benchmark: parser output is fixed, splitter implementation changes;
- parser benchmark: Ragbot chunking is fixed, parser implementation changes.

The scorer is shared with ``benchmarks.rag_native_compare`` so development
concept labels and production page/doc/path labels have identical semantics.
"""
from __future__ import annotations

import math
import statistics
import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from benchmarks.rag_framework_compare import build_chunker
from benchmarks.rag_native_compare import (
    CorpusUnit,
    RetrievedHit,
    score_case,
    summarize_scores,
)
from services.worker.parsing import iter_document_segments, parse_document


@dataclass(frozen=True)
class RawPdf:
    doc_id: str
    path: str
    data: bytes
    page_count: int


def _normalize(vector: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(float(value) * float(value) for value in vector)) or 1.0
    return [float(value) / norm for value in vector]


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def _boundary_ending(text: str) -> bool:
    stripped = text.rstrip()
    return not stripped or stripped[-1] in ".!?。！？;；:：\n"


def _query_vector(embedder: Any, query: str) -> Sequence[float]:
    embed_query = getattr(embedder, "embed_query", None)
    return embed_query(query) if callable(embed_query) else embedder.embed(query)


def _cosine_search(
    query: str,
    *,
    embedder: Any,
    vectors: Sequence[Sequence[float]],
    hits: Sequence[RetrievedHit],
    top_k: int,
) -> list[RetrievedHit]:
    query_vector = _normalize(_query_vector(embedder, query))
    scored = [
        (hit, sum(q * d for q, d in zip(query_vector, vector)))
        for hit, vector in zip(hits, vectors)
    ]
    scored.sort(key=lambda item: item[1], reverse=True)
    return [
        RetrievedHit(
            chunk_id=hit.chunk_id,
            doc_id=hit.doc_id,
            path=hit.path,
            text=hit.text,
            score=score,
            page=hit.page,
            metadata=dict(hit.metadata),
        )
        for hit, score in scored[:top_k]
    ]


def _evaluate(
    cases: Sequence[Mapping[str, Any]],
    *,
    hits: Sequence[RetrievedHit],
    vectors: Sequence[Sequence[float]],
    embedder: Any,
    top_k: int,
    repetitions: int,
) -> dict[str, Any]:
    latencies_ms: list[float] = []
    scores = []
    for case in cases:
        first_hits: list[RetrievedHit] | None = None
        for _ in range(max(1, repetitions)):
            started = time.perf_counter()
            current = _cosine_search(
                str(case["query"]),
                embedder=embedder,
                vectors=vectors,
                hits=hits,
                top_k=top_k,
            )
            latencies_ms.append((time.perf_counter() - started) * 1000.0)
            if first_hits is None:
                first_hits = current
        scores.append(score_case(case, first_hits or []))
    return summarize_scores(scores, latencies_ms)


def _splitter_hits(
    units: Sequence[CorpusUnit],
    *,
    splitter: str,
    chunk_size: int,
    chunk_overlap: int,
) -> tuple[list[RetrievedHit], float]:
    chunker = build_chunker(splitter, chunk_size, chunk_overlap)
    started = time.perf_counter()
    hits: list[RetrievedHit] = []
    for unit in units:
        for index, text in enumerate(chunker.split(unit.text)):
            hits.append(
                RetrievedHit(
                    chunk_id=f"{unit.doc_id}::p{unit.page or 0}::{index}",
                    doc_id=unit.doc_id,
                    path=unit.path,
                    text=text,
                    page=unit.page,
                    metadata={"splitter": chunker.name},
                )
            )
    return hits, time.perf_counter() - started


def run_splitter_backend(
    splitter: str,
    units: Sequence[CorpusUnit],
    cases: Sequence[Mapping[str, Any]],
    *,
    embedder: Any,
    chunk_size: int,
    chunk_overlap: int,
    top_k: int,
    repetitions: int = 1,
) -> dict[str, Any]:
    """Benchmark only the splitter while holding parser/embedder/search fixed."""
    tracemalloc.start()
    total_started = time.perf_counter()
    hits, split_seconds = _splitter_hits(
        units,
        splitter=splitter,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    if not hits:
        raise RuntimeError(f"Splitter {splitter} produced zero chunks")

    embed_started = time.perf_counter()
    vectors = [_normalize(v) for v in embedder.embed_batch([hit.text for hit in hits])]
    embedding_seconds = time.perf_counter() - embed_started
    quality = _evaluate(
        cases,
        hits=hits,
        vectors=vectors,
        embedder=embedder,
        top_k=top_k,
        repetitions=repetitions,
    )
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    lengths = [len(hit.text) for hit in hits]
    source_chars = sum(len(unit.text) for unit in units)
    chunk_chars = sum(lengths)
    pages = {(unit.doc_id, unit.page) for unit in units}
    return {
        "component": "splitter",
        "backend": splitter,
        "embedding_model": embedder.model_name,
        "embedding_dimension": embedder.dimension,
        "units": len(units),
        "chunks": len(hits),
        "quality": quality,
        "timing": {
            "split_seconds": round(split_seconds, 6),
            "embedding_seconds": round(embedding_seconds, 6),
            "total_seconds": round(time.perf_counter() - total_started, 6),
        },
        "chunk_shape": {
            "chars_mean": round(statistics.fmean(lengths), 3),
            "chars_p50": round(_percentile(lengths, 0.50), 3),
            "chars_p95": round(_percentile(lengths, 0.95), 3),
            "chars_max": max(lengths),
            "non_boundary_end_rate": round(
                sum(1 for hit in hits if not _boundary_ending(hit.text)) / len(hits), 6
            ),
            "character_inflation_ratio": round(chunk_chars / max(source_chars, 1), 6),
            "chunks_per_page": round(len(hits) / max(len(pages), 1), 3),
        },
        "memory": {"tracemalloc_peak_bytes": peak},
    }


def parser_config(name: str) -> dict[str, str]:
    normalized = name.strip().lower()
    if normalized == "pypdf2":
        return {"provider": "ragbot", "strategy": "pypdf2"}
    if normalized == "pymupdf":
        return {"provider": "pymupdf", "strategy": "blocks"}
    if normalized == "docling":
        return {"provider": "docling", "strategy": "document"}
    if normalized == "unstructured":
        return {"provider": "unstructured", "strategy": "elements"}
    raise ValueError(f"Unsupported parser backend: {name}")


def load_raw_pdfs(source: Path) -> list[RawPdf]:
    try:
        from PyPDF2 import PdfReader
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Parser benchmark requires the worker extra (PyPDF2)") from exc

    paths = [source] if source.is_file() else sorted(source.rglob("*.pdf"))
    root = source.parent if source.is_file() else source
    result: list[RawPdf] = []
    for path in paths:
        if path.suffix.lower() != ".pdf":
            continue
        relative = path.relative_to(root).as_posix()
        data = path.read_bytes()
        result.append(
            RawPdf(
                doc_id=relative,
                path=relative,
                data=data,
                page_count=len(PdfReader(str(path)).pages),
            )
        )
    if not result:
        raise ValueError(f"No PDF documents found at {source}")
    return result


def _label_text_coverage(cases: Sequence[Mapping[str, Any]], text: str) -> float:
    normalized = " ".join(text.casefold().split())
    covered = 0
    labeled = 0
    for case in cases:
        relevance = case.get("relevance") or {}
        all_terms = [" ".join(str(v).casefold().split()) for v in relevance.get("all_terms") or []]
        any_terms = [" ".join(str(v).casefold().split()) for v in relevance.get("any_terms") or []]
        if not all_terms and not any_terms:
            continue
        labeled += 1
        ok_all = all(term in normalized for term in all_terms) if all_terms else True
        ok_any = any(term in normalized for term in any_terms) if any_terms else True
        if ok_all and ok_any:
            covered += 1
    return round(covered / labeled, 6) if labeled else 1.0


def run_parser_backend(
    backend: str,
    documents: Sequence[RawPdf],
    cases: Sequence[Mapping[str, Any]],
    *,
    embedder: Any,
    chunk_size: int,
    chunk_overlap: int,
    top_k: int,
    repetitions: int = 1,
) -> dict[str, Any]:
    """Benchmark only the parser while holding Ragbot chunking/search fixed."""
    tracemalloc.start()
    total_started = time.perf_counter()
    config = parser_config(backend)
    parse_seconds = 0.0
    chunk_seconds = 0.0
    block_count = 0
    page_blocks = 0
    bbox_blocks = 0
    table_blocks = 0
    section_blocks = 0
    extracted_chars = 0
    extracted_pages: set[tuple[str, int]] = set()
    hits: list[RetrievedHit] = []
    parser_identity: dict[str, Any] | None = None
    parsed_text_parts: list[str] = []

    for document in documents:
        started = time.perf_counter()
        normalized, metadata = parse_document(
            document.data,
            config,
            name=Path(document.path).name,
            media_type="application/pdf",
            uri=document.path,
        )
        parse_seconds += time.perf_counter() - started
        parser_identity = parser_identity or metadata
        parsed_text_parts.append(normalized.text)
        extracted_chars += len(normalized.text)
        block_count += len(normalized.blocks)
        page_blocks += sum(block.page is not None for block in normalized.blocks)
        bbox_blocks += sum(block.bbox is not None for block in normalized.blocks)
        table_blocks += sum("table" in block.kind.casefold() for block in normalized.blocks)
        section_blocks += sum(bool(block.section) for block in normalized.blocks)
        extracted_pages.update(
            (document.doc_id, int(block.page))
            for block in normalized.blocks
            if block.page is not None
        )

        chunk_started = time.perf_counter()
        for index, segment in enumerate(
            iter_document_segments(
                normalized,
                None,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        ):
            hits.append(
                RetrievedHit(
                    chunk_id=f"{document.doc_id}::{index}",
                    doc_id=document.doc_id,
                    path=document.path,
                    text=segment.text,
                    page=segment.page,
                    metadata={
                        "parser": backend,
                        "block_kind": segment.block_kind,
                        "section": segment.section,
                    },
                )
            )
        chunk_seconds += time.perf_counter() - chunk_started

    if not hits:
        raise RuntimeError(f"Parser {backend} produced zero chunks")

    embed_started = time.perf_counter()
    vectors = [_normalize(v) for v in embedder.embed_batch([hit.text for hit in hits])]
    embedding_seconds = time.perf_counter() - embed_started
    quality = _evaluate(
        cases,
        hits=hits,
        vectors=vectors,
        embedder=embedder,
        top_k=top_k,
        repetitions=repetitions,
    )
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    expected_pages = sum(document.page_count for document in documents)
    metadata = parser_identity or {}
    return {
        "component": "parser",
        "backend": backend,
        "parser": {
            "provider": metadata.get("parser_provider", config["provider"]),
            "strategy": metadata.get("parser_strategy", config["strategy"]),
            "version": metadata.get("parser_version"),
            "config_hash": metadata.get("parser_config_hash"),
        },
        "embedding_model": embedder.model_name,
        "embedding_dimension": embedder.dimension,
        "documents": len(documents),
        "blocks": block_count,
        "chunks": len(hits),
        "quality": quality,
        "timing": {
            "parser_seconds": round(parse_seconds, 6),
            "chunking_seconds": round(chunk_seconds, 6),
            "embedding_seconds": round(embedding_seconds, 6),
            "total_seconds": round(time.perf_counter() - total_started, 6),
            "documents_per_second": round(len(documents) / max(parse_seconds, 1e-9), 3),
        },
        "fidelity": {
            "characters_extracted": extracted_chars,
            "characters_per_page": round(extracted_chars / max(expected_pages, 1), 3),
            "page_metadata_coverage": round(len(extracted_pages) / max(expected_pages, 1), 6),
            "label_text_coverage": _label_text_coverage(cases, "\n".join(parsed_text_parts)),
            "page_block_rate": round(page_blocks / max(block_count, 1), 6),
            "bbox_block_rate": round(bbox_blocks / max(block_count, 1), 6),
            "table_block_rate": round(table_blocks / max(block_count, 1), 6),
            "section_block_rate": round(section_blocks / max(block_count, 1), 6),
        },
        "memory": {"tracemalloc_peak_bytes": peak},
    }
