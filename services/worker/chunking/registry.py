from __future__ import annotations

from functools import lru_cache
from typing import Any, Mapping

from .adapters import LangChainChunker, LlamaIndexChunker
from .fixed import FixedWindowChunker
from .protocol import Chunker, ChunkingSpec

_DEFAULT_STRATEGY = {
    "ragbot": "fixed",
    "langchain": "recursive",
    "llamaindex": "sentence",
}
_VALID_STRATEGIES = {
    "ragbot": {"fixed", "structural"},
    "langchain": {"recursive", "code"},
    "llamaindex": {"sentence"},
}


def resolve_chunking_spec(
    config: Mapping[str, Any] | None,
    *,
    chunk_size: int,
    chunk_overlap: int,
    language: str | None = None,
    default_strategy: str | None = None,
) -> ChunkingSpec:
    """Resolve legacy size fields plus optional nested chunking configuration."""
    raw = dict(config or {})
    provider = str(raw.get("provider") or "ragbot").strip().lower()
    if provider not in _VALID_STRATEGIES:
        raise ValueError(
            f"Unsupported chunker provider {provider!r}; expected one of {sorted(_VALID_STRATEGIES)}"
        )
    implicit_strategy = default_strategy if provider == "ragbot" and default_strategy else _DEFAULT_STRATEGY[provider]
    strategy = str(raw.get("strategy") or implicit_strategy).strip().lower()
    if strategy not in _VALID_STRATEGIES[provider]:
        raise ValueError(
            f"Unsupported chunker strategy {provider}/{strategy}; "
            f"expected one of {sorted(_VALID_STRATEGIES[provider])}"
        )
    return ChunkingSpec(
        provider=provider,
        strategy=strategy,
        version=int(raw.get("version", 1)),
        chunk_size=int(raw.get("chunk_size", chunk_size)),
        chunk_overlap=int(raw.get("chunk_overlap", chunk_overlap)),
        language=str(language).strip().lower() if language else None,
    )


def chunking_metadata(
    config: Mapping[str, Any] | None,
    *,
    chunk_size: int,
    chunk_overlap: int,
    language: str | None = None,
    default_strategy: str | None = None,
) -> dict[str, object]:
    return resolve_chunking_spec(
        config,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        language=language,
        default_strategy=default_strategy,
    ).metadata()


def split_text(
    text: str,
    config: Mapping[str, Any] | None,
    *,
    chunk_size: int,
    chunk_overlap: int,
    language: str | None = None,
    default_strategy: str | None = None,
) -> tuple[list[str], dict[str, object]]:
    spec = resolve_chunking_spec(
        config,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        language=language,
        default_strategy=default_strategy,
    )
    chunker = _build_chunker(spec)
    return chunker.split(text), spec.metadata()


@lru_cache(maxsize=128)
def _build_chunker(spec: ChunkingSpec) -> Chunker:
    if spec.provider == "ragbot" and spec.strategy == "fixed":
        return FixedWindowChunker(spec)
    if spec.provider == "ragbot" and spec.strategy == "structural":
        raise ValueError("ragbot/structural is implemented by repository-aware ingestion, not generic split_text")
    if spec.provider == "langchain":
        return LangChainChunker(spec)
    if spec.provider == "llamaindex":
        return LlamaIndexChunker(spec)
    raise ValueError(f"Unsupported chunker provider: {spec.provider}")
