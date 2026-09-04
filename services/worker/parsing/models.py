from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class DocumentBlock:
    """Framework-neutral structural unit emitted by a document parser."""

    block_index: int
    text: str
    kind: str = "text"
    page: int | None = None
    section: str | None = None
    bbox: tuple[float, float, float, float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.block_index < 0:
            raise ValueError("block_index must be >= 0")
        if self.page is not None and self.page < 1:
            raise ValueError("page must be 1-based when present")
        if not self.text.strip():
            raise ValueError("document block text must not be empty")


@dataclass
class NormalizedDocument:
    """Stable hand-off between connector/parser and Ragbot's chunking kernel."""

    name: str
    media_type: str
    blocks: list[DocumentBlock]
    uri: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        return "\n\n".join(block.text for block in self.blocks)

    @property
    def pages(self) -> set[int]:
        return {block.page for block in self.blocks if block.page is not None}
