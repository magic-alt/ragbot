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
    block_coalescing_enabled: bool = False
    block_coalescing_target_chars: int | None = None
    block_coalescing_respect_page: bool = True
    block_coalescing_respect_section: bool = True

    def __post_init__(self) -> None:
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap must be >= 0")
        if self.strategy != "structural" and self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if self.version < 1:
            raise ValueError("chunker version must be >= 1")
        if self.block_coalescing_enabled:
            if self.block_coalescing_target_chars is None or self.block_coalescing_target_chars < 1:
                raise ValueError("block_coalescing_target_chars must be >= 1 when enabled")

    @property
    def config_hash(self) -> str:
        payload: dict[str, object] = {
            "provider": self.provider,
            "strategy": self.strategy,
            "version": self.version,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "language": self.language,
        }
        # Preserve legacy hashes for the default disabled path. Once coalescing
        # is enabled it becomes part of the durable chunk/index identity so
        # metadata-first refresh cannot accidentally reuse incompatible chunks.
        if self.block_coalescing_enabled:
            payload["block_coalescing"] = {
                "enabled": True,
                "target_chars": self.block_coalescing_target_chars,
                "respect_page": self.block_coalescing_respect_page,
                "respect_section": self.block_coalescing_respect_section,
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
        if self.block_coalescing_enabled:
            result.update(
                {
                    "block_coalescing_enabled": True,
                    "block_coalescing_target_chars": int(self.block_coalescing_target_chars or 0),
                    "block_coalescing_respect_page": self.block_coalescing_respect_page,
                    "block_coalescing_respect_section": self.block_coalescing_respect_section,
                }
            )
        return result


@runtime_checkable
class Chunker(Protocol):
    """Minimal splitter port owned by Ragbot's ingestion kernel."""

    @property
    def spec(self) -> ChunkingSpec: ...

    def split(self, text: str) -> list[str]: ...
