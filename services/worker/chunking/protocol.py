from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class ChunkingSpec:
    """Stable index-contract identity for one document chunking strategy."""

    provider: str
    strategy: str
    version: int
    chunk_size: int
    chunk_overlap: int
    language: str | None = None

    def __post_init__(self) -> None:
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap must be >= 0")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if self.version < 1:
            raise ValueError("chunker version must be >= 1")

    @property
    def config_hash(self) -> str:
        payload = {
            "provider": self.provider,
            "strategy": self.strategy,
            "version": self.version,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "language": self.language,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:20]

    def metadata(self) -> dict[str, object]:
        result: dict[str, object] = {
            "chunker_provider": self.provider,
            "chunker_strategy": self.strategy,
            "chunker_version": self.version,
            "chunker_config_hash": self.config_hash,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
        }
        if self.language:
            result["chunker_language"] = self.language
        return result


@runtime_checkable
class Chunker(Protocol):
    """Minimal splitter port owned by Ragbot's ingestion kernel."""

    @property
    def spec(self) -> ChunkingSpec: ...

    def split(self, text: str) -> list[str]: ...
