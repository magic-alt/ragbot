"""Factorial parser × block-coalescing × splitter benchmark.

The benchmark reuses Ragbot's production parser bridge. It therefore measures
exactly the same optional block-coalescing and chunker adapters that ingestion
would use after a configuration is promoted.
"""
from __future__ import annotations

import statistics
import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from benchmarks.component_compare import (
    RawPdf,
    _boundary_ending,
    _evaluate,
    _label_text_coverage,
    _normalize,
    _percentile,
    parser_config,
)
from benchmarks.rag_native_compare import RetrievedHit
from services.worker.parsing import (
    NormalizedDocument,
    coalesce_document_blocks,
    iter_document_segments,
    parse_document,
)


@dataclass(frozen=True)
class PreparedDocument:
    doc_id: str
    path: str
    page_count: int
    normalized: NormalizedDocument


@dataclass(frozen=True)
class PreparedParser:
    backend: str
    documents: tuple[PreparedDocument, ...]
    parser_seconds: float
    parser_metadata: Mapping[str, Any]


def splitter_config(
    name: str,
    *,
    coalesced: bool,
    chunk_size: int,
    coalesce_target_multiplier: float = 4.0,
) -> dict[str, Any]:
    normalized = name.strip().lower()
    mapping = {
        "ragbot": {"provider": "ragbot", "strategy": "fixed"},
        "langchain": {"provider": "langchain", "strategy": "recursive"},
        "llamaindex": {"provider": "llamaindex", "strategy": "sentence"},
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported factorial splitter: {name}")
    config: dict[str, Any] = dict(mapping[normalized])
    if coalesced:
        target = max(chunk_size, int(round(chunk_size * coalesce_target_multiplier)))
        config["block_coalescing"] = {
            "enabled": True,
            "target_chars": target,
            "respect_page": True,
            "respect_section": True,
        }
    return config


def prepare_parser_backend(backend: str, documents: Sequence[RawPdf]) -> PreparedParser:
    config = parser_config(backend)
    prepared: list[PreparedDocument] = []
    elapsed = 0.0
    parser_metadata: Mapping[str, Any] | None = None
    for document in documents:
        started = time.perf_counter()
        normalized, metadata = parse_document(
            document.data,
            config,
            name=Path(document.path).name,
            media_type="application/pdf",
            uri=document.path,
        )
        elapsed += time.perf_counter() - started
        parser_metadata = parser_metadata or metadata
        prepared.append(
            PreparedDocument(
                doc_id=document.doc_id,
                path=document.path,
                page_count=document.page_count,
                normalized=normalized,
            )
        )
    return PreparedParser(
        backend=backend,
        documents=tuple(prepared),
        parser_seconds=elapsed,
        parser_metadata=dict(parser_metadata or {}),
    )


def _parser_fidelity(prepared: PreparedParser, cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    blocks = [block for document in prepared.documents for block in document.normalized.blocks]
    expected_pages = sum(document.page_count for document in prepared.documents)
    extracted_pages = {
        (document.doc_id, int(block.page))
        for document in prepared.documents
        for block in document.normalized.blocks
        if block.page is not None
    }
    text = "\n".join(document.normalized.text for document in prepared.documents)
    characters = sum(len(document.normalized.text) for document in prepared.documents)
    block_count = len(blocks)
    return {
        "characters_extracted": characters,
        "characters_per_page": round(characters / max(expected_pages, 1), 3),
        "page_metadata_coverage": round(len(extracted_pages) / max(expected_pages, 1), 6),
        "label_text_coverage": _label_text_coverage(cases, text),
        "page_block_rate": round(sum(block.page is not None for block in blocks) / max(block_count, 1), 6),
        "bbox_block_rate": round(sum(block.bbox is not None for block in blocks) / max(block_count, 1), 6),
        "table_block_rate": round(sum("table" in block.kind.casefold() for block in blocks) / max(block_count, 1), 6),
        "section_block_rate": round(sum(bool(block.section) for block in blocks) / max(block_count, 1), 6),
    }


def run_factorial_cell(
    prepared: PreparedParser,
    splitter: str,
    coalescing: str,
    cases: Sequence[Mapping[str, Any]],
    *,
    embedder: Any,
    chunk_size: int,
    chunk_overlap: int,
    top_k: int,
    repetitions: int = 1,
    coalesce_target_multiplier: float = 4.0,
) -> dict[str, Any]:
    mode = coalescing.strip().lower()
    if mode not in {"raw", "coalesced"}:
        raise ValueError("coalescing must be raw or coalesced")
    coalesced = mode == "coalesced"
    config = splitter_config(
        splitter,
        coalesced=coalesced,
        chunk_size=chunk_size,
        coalesce_target_multiplier=coalesce_target_multiplier,
    )

    tracemalloc.start()
    total_started = time.perf_counter()
    raw_blocks = sum(len(document.normalized.blocks) for document in prepared.documents)
    bridge_blocks = 0
    segment_started = time.perf_counter()
    hits: list[RetrievedHit] = []
    for document in prepared.documents:
        if coalesced:
            target = int(config["block_coalescing"]["target_chars"])
            bridge_blocks += len(
                coalesce_document_blocks(
                    document.normalized,
                    target_chars=target,
                    respect_page=True,
                    respect_section=True,
                )
            )
        else:
            bridge_blocks += len(document.normalized.blocks)

        for index, segment in enumerate(
            iter_document_segments(
                document.normalized,
                config,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        ):
            block_metadata = dict(segment.metadata.get("block_metadata") or {})
            hits.append(
                RetrievedHit(
                    chunk_id=f"{document.doc_id}::{splitter}::{mode}::{index}",
                    doc_id=document.doc_id,
                    path=document.path,
                    text=segment.text,
                    page=segment.page,
                    metadata={
                        "parser": prepared.backend,
                        "splitter": splitter,
                        "coalescing": mode,
                        "block_kind": segment.block_kind,
                        "section": segment.section,
                        "source_block_count": int(block_metadata.get("source_block_count") or 1),
                    },
                )
            )
    segment_seconds = time.perf_counter() - segment_started
    if not hits:
        raise RuntimeError(
            f"Factorial cell {prepared.backend}/{splitter}/{mode} produced zero chunks"
        )

    embed_started = time.perf_counter()
    vectors = [_normalize(vector) for vector in embedder.embed_batch([hit.text for hit in hits])]
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
    source_chars = sum(len(document.normalized.text) for document in prepared.documents)
    chunk_chars = sum(lengths)
    source_block_counts = [int(hit.metadata.get("source_block_count") or 1) for hit in hits]
    parser_metadata = dict(prepared.parser_metadata)
    return {
        "component": "factorial",
        "cell": f"{prepared.backend}+{mode}+{splitter}",
        "parser_backend": prepared.backend,
        "splitter_backend": splitter,
        "coalescing": mode,
        "parser": {
            "provider": parser_metadata.get("parser_provider"),
            "strategy": parser_metadata.get("parser_strategy"),
            "version": parser_metadata.get("parser_version"),
            "config_hash": parser_metadata.get("parser_config_hash"),
        },
        "embedding_model": embedder.model_name,
        "embedding_dimension": embedder.dimension,
        "documents": len(prepared.documents),
        "raw_blocks": raw_blocks,
        "bridge_blocks": bridge_blocks,
        "chunks": len(hits),
        "quality": quality,
        "timing": {
            "parser_seconds": round(prepared.parser_seconds, 6),
            "segment_seconds": round(segment_seconds, 6),
            "embedding_seconds": round(embedding_seconds, 6),
            "cell_seconds_excluding_cached_parse": round(time.perf_counter() - total_started, 6),
        },
        "coalescing_metrics": {
            "block_reduction_rate": round(1.0 - bridge_blocks / max(raw_blocks, 1), 6),
            "target_multiplier": coalesce_target_multiplier if coalesced else None,
            "source_blocks_per_chunk_mean": round(statistics.fmean(source_block_counts), 3),
            "source_blocks_per_chunk_p95": round(_percentile(source_block_counts, 0.95), 3),
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
        },
        "fidelity": _parser_fidelity(prepared, cases),
        "memory": {"tracemalloc_peak_bytes": peak},
    }


def compact_factorial_cells(
    parsers: Sequence[str],
    splitters: Sequence[str],
) -> list[tuple[str, str, str]]:
    """Return a six-cell default for two parsers × two splitters.

    PyPDF2 is already page-coalesced by construction, so its redundant
    coalesced rows are omitted. Structured parsers get both raw and coalesced
    rows. With pypdf2,pymupdf × ragbot,llamaindex this yields 3 pipeline levels
    × 2 splitters = 6 cells.
    """
    cells: list[tuple[str, str, str]] = []
    for parser in parsers:
        modes = ("raw",) if parser == "pypdf2" else ("raw", "coalesced")
        for mode in modes:
            for splitter in splitters:
                cells.append((parser, splitter, mode))
    return cells
