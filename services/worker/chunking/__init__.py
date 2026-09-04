"""Ragbot-owned document transformation ports.

Framework-specific splitters live behind this package so connectors and the
production ingestion kernel never depend on LangChain/LlamaIndex APIs directly.
"""

from .protocol import Chunker, ChunkingSpec
from .registry import chunking_metadata, resolve_chunking_spec, split_text

__all__ = [
    "Chunker",
    "ChunkingSpec",
    "chunking_metadata",
    "resolve_chunking_spec",
    "split_text",
]
