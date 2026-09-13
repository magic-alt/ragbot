from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class EmbeddingSpec:
    provider_id: str
    model: str
    dimension: int
    revision: str = ""
    distance: str = "cosine"
    normalize: bool = True
    query_instruction: str = ""
    document_instruction: str = ""
    max_batch_items: int = 100
    max_batch_bytes: int = 1_000_000
    multilingual: bool = False
    multimodal: bool = False

    def __post_init__(self) -> None:
        if self.dimension <= 0:
            raise ValueError("embedding dimension must be > 0")
        if self.max_batch_items <= 0 or self.max_batch_bytes <= 0:
            raise ValueError("embedding batch limits must be > 0")
        if self.distance.lower() not in {"cosine", "dot", "euclid", "manhattan"}:
            raise ValueError(f"unsupported embedding distance: {self.distance}")

    @property
    def contract_id(self) -> str:
        canonical = json.dumps(self.as_public_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:24]
        return f"emb-{digest}"

    def as_public_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EmbeddingBatchMetrics:
    requests: int = 0
    inputs: int = 0
    bytes: int = 0
    retries: int = 0
    cache_hits: int = 0
    failures: int = 0
    provider_latency_ms: float = 0.0


def embedding_contract_id(embedder: Any) -> str:
    value = getattr(embedder, "contract_id", None)
    if isinstance(value, str) and value:
        return value
    model = str(getattr(embedder, "model_name", "unknown"))
    dimension = int(getattr(embedder, "dimension", 0) or 0)
    spec = EmbeddingSpec(provider_id=type(embedder).__name__.lower(), model=model, dimension=max(1, dimension))
    return spec.contract_id
