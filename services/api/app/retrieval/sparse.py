from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Optional, Protocol, Sequence, runtime_checkable


@dataclass(frozen=True)
class SparseVector:
    indices: tuple[int, ...]
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.indices) != len(self.values):
            raise ValueError("sparse vector indices/values length mismatch")
        if any(int(index) < 0 for index in self.indices):
            raise ValueError("sparse vector indices must be non-negative")

    def as_qdrant_dict(self) -> dict[str, list[Any]]:
        return {
            "indices": [int(item) for item in self.indices],
            "values": [float(item) for item in self.values],
        }


@dataclass(frozen=True)
class SparseEmbeddingSpec:
    provider_id: str
    model: str
    revision: str = ""
    vector_name: str = "sparse"
    modifier: str = "idf"
    tokenizer: str = ""
    language: str = ""

    def __post_init__(self) -> None:
        if not self.provider_id.strip():
            raise ValueError("sparse provider_id must not be empty")
        if not self.model.strip():
            raise ValueError("sparse model must not be empty")
        if not self.vector_name.strip():
            raise ValueError("sparse vector_name must not be empty")
        if self.modifier.strip().lower() not in {"", "none", "idf"}:
            raise ValueError(f"unsupported sparse modifier: {self.modifier}")

    @property
    def contract_id(self) -> str:
        canonical = json.dumps(
            self.as_public_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:24]
        return f"sparse-{digest}"

    def as_public_dict(self) -> dict[str, Any]:
        return asdict(self)


@runtime_checkable
class SparseEncoder(Protocol):
    @property
    def spec(self) -> SparseEmbeddingSpec: ...

    @property
    def contract_id(self) -> str: ...

    @property
    def model_name(self) -> str: ...

    def embed_query(self, text: str) -> SparseVector: ...

    def embed_documents(self, texts: Sequence[str]) -> list[SparseVector]: ...


class FastEmbedSparseEncoder:
    """Lazy FastEmbed sparse encoder.

    Model weights are not loaded until the first encode operation. This keeps
    API/worker startup cheap while still making the representation contract
    available for IndexVersion planning and diagnostics.
    """

    def __init__(self, spec: SparseEmbeddingSpec, *, batch_size: int = 32) -> None:
        self._spec = spec
        self._batch_size = max(1, int(batch_size))
        self._model: Any = None

    @property
    def spec(self) -> SparseEmbeddingSpec:
        return self._spec

    @property
    def contract_id(self) -> str:
        return self._spec.contract_id

    @property
    def model_name(self) -> str:
        return self._spec.model

    def _get_model(self) -> Any:
        if self._model is None:
            try:
                from fastembed import SparseTextEmbedding
            except ImportError as exc:  # pragma: no cover - dependency guard
                raise RuntimeError(
                    "Sparse retrieval requires FastEmbed; install ragbot[sparse]"
                ) from exc
            self._model = SparseTextEmbedding(
                model_name=self._spec.model,
                batch_size=self._batch_size,
            )
        return self._model

    def embed_query(self, text: str) -> SparseVector:
        value = str(text or "").strip()
        if not value:
            raise ValueError("sparse query must not be empty")
        model = self._get_model()
        query_embed = getattr(model, "query_embed", None)
        if callable(query_embed):
            encoded = list(query_embed([value], batch_size=self._batch_size))
        else:
            encoded = list(model.embed([value], batch_size=self._batch_size))
        if len(encoded) != 1:
            raise RuntimeError(f"Sparse encoder returned {len(encoded)} query vectors")
        return _coerce_sparse_vector(encoded[0])

    def embed_documents(self, texts: Sequence[str]) -> list[SparseVector]:
        values = [str(item or "") for item in texts]
        if not values:
            return []
        model = self._get_model()
        passage_embed = getattr(model, "passage_embed", None)
        if callable(passage_embed):
            encoded = list(passage_embed(values, batch_size=self._batch_size))
        else:
            encoded = list(model.embed(values, batch_size=self._batch_size))
        if len(encoded) != len(values):
            raise RuntimeError(
                f"Sparse encoder returned {len(encoded)} vectors for {len(values)} documents"
            )
        return [_coerce_sparse_vector(item) for item in encoded]


class DeterministicSparseEncoder:
    """Small dependency-free sparse encoder for tests/development only.

    It is deliberately never selected by production environment construction.
    Tokens are hashed into a stable integer space and weighted by term count.
    """

    def __init__(self, *, vector_name: str = "sparse-test") -> None:
        self._spec = SparseEmbeddingSpec(
            provider_id="deterministic-test",
            model="token-hash-v1",
            vector_name=vector_name,
            modifier="none",
        )

    @property
    def spec(self) -> SparseEmbeddingSpec:
        return self._spec

    @property
    def contract_id(self) -> str:
        return self._spec.contract_id

    @property
    def model_name(self) -> str:
        return self._spec.model

    def embed_query(self, text: str) -> SparseVector:
        return self.embed_documents([text])[0]

    def embed_documents(self, texts: Sequence[str]) -> list[SparseVector]:
        output: list[SparseVector] = []
        for text in texts:
            counts: dict[int, float] = {}
            for token in _simple_tokens(str(text or "")):
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=4).digest()
                index = int.from_bytes(digest, "big", signed=False)
                counts[index] = counts.get(index, 0.0) + 1.0
            ordered = sorted(counts.items())
            output.append(
                SparseVector(
                    indices=tuple(index for index, _value in ordered),
                    values=tuple(value for _index, value in ordered),
                )
            )
        return output


def build_sparse_encoder_from_env() -> Optional[SparseEncoder]:
    if not _env_flag("RAGBOT_SPARSE_ENABLED", False):
        return None
    provider = os.getenv("RAGBOT_SPARSE_PROVIDER", "fastembed").strip().lower()
    if provider != "fastembed":
        raise ValueError(f"Unsupported sparse encoder provider: {provider}")
    model = os.getenv("RAGBOT_SPARSE_MODEL", "Qdrant/bm25").strip()
    spec = SparseEmbeddingSpec(
        provider_id="fastembed",
        model=model,
        revision=os.getenv("RAGBOT_SPARSE_REVISION", "").strip(),
        vector_name=os.getenv("RAGBOT_SPARSE_VECTOR_NAME", "sparse").strip() or "sparse",
        modifier=os.getenv("RAGBOT_SPARSE_MODIFIER", "idf").strip().lower() or "idf",
        tokenizer=os.getenv("RAGBOT_SPARSE_TOKENIZER", "").strip(),
        language=os.getenv("RAGBOT_SPARSE_LANGUAGE", "").strip(),
    )
    return FastEmbedSparseEncoder(
        spec,
        batch_size=int(os.getenv("RAGBOT_SPARSE_BATCH_SIZE", "32")),
    )


def sparse_contract_id(encoder: Optional[SparseEncoder]) -> Optional[str]:
    if encoder is None:
        return None
    value = str(getattr(encoder, "contract_id", "") or "").strip()
    return value or None


def sparse_contract_from_index(index_version: Any) -> Optional[dict[str, Any]]:
    if index_version is None:
        return None
    schema = dict(getattr(index_version, "vector_schema", None) or {})
    sparse = schema.get("sparse")
    return dict(sparse) if isinstance(sparse, dict) else None


def _coerce_sparse_vector(value: Any) -> SparseVector:
    indices = getattr(value, "indices", None)
    values = getattr(value, "values", None)
    if indices is None and isinstance(value, dict):
        indices = value.get("indices")
        values = value.get("values")
    if indices is None or values is None:
        raise TypeError(f"Unsupported sparse embedding value: {type(value).__name__}")
    return SparseVector(
        indices=tuple(int(item) for item in list(indices)),
        values=tuple(float(item) for item in list(values)),
    )


def _simple_tokens(text: str) -> Iterable[str]:
    token: list[str] = []
    for char in text.casefold():
        if char.isalnum() or "\u3400" <= char <= "\u9fff":
            token.append(char)
        elif token:
            yield "".join(token)
            token = []
    if token:
        yield "".join(token)


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}
