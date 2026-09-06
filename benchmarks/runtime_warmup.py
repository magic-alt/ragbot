"""Warm optional benchmark runtimes before steady-state memory measurement."""
from __future__ import annotations

from typing import Any, Mapping

from services.worker.chunking import split_text


def warm_chunker_runtime(
    config: Mapping[str, Any] | None,
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> dict[str, object]:
    """Instantiate/cache the exact configured chunker outside tracemalloc.

    Optional adapters such as LlamaIndex import modules and construct splitter
    objects lazily on first use. Promotion memory gates compare steady-state
    pipeline allocation, so one-time adapter import/setup must not be charged to
    whichever pipeline happens to run first.
    """
    chunks, metadata = split_text(
        "Warm-up sentence one. Warm-up sentence two.",
        config,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    if not chunks:
        raise RuntimeError("Chunker warm-up produced zero chunks")
    return {
        "provider": metadata.get("chunker_provider"),
        "strategy": metadata.get("chunker_strategy"),
        "config_hash": metadata.get("chunker_config_hash"),
        "chunks": len(chunks),
    }
