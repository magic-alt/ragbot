from __future__ import annotations

from .protocol import ChunkingSpec


class FixedWindowChunker:
    """Backward-compatible Ragbot fixed-character splitter."""

    def __init__(self, spec: ChunkingSpec) -> None:
        self._spec = spec

    @property
    def spec(self) -> ChunkingSpec:
        return self._spec

    def split(self, text: str) -> list[str]:
        segments: list[str] = []
        start = 0
        while start < len(text):
            end = start + self._spec.chunk_size
            segment = text[start:end].strip()
            if segment:
                segments.append(segment)
            start = end - self._spec.chunk_overlap if end < len(text) else end
        return segments
