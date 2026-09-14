from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

from contracts.types import RetrievalChunk

from ..storage.models import Chunk
from ..storage.protocol import Repo
from .async_engine import AsyncRetrievalEngine
from .contracts import (
    RetrievalPlan,
    RetrievalRequest,
    RetrievalResponse,
    resolve_retrieval_plan,
)
from .embedder import Embedder, HashEmbedder
from .lexical import contains_cjk


def build_citation(chunk: Chunk) -> str:
    location = chunk.path or chunk.url
    if location and chunk.page is not None:
        return f"{chunk.doc_id}:{location}:page={chunk.page}:chunk={chunk.chunk_index}"
    if chunk.path:
        return f"{chunk.doc_id}:{chunk.path}:{chunk.chunk_index}"
    if chunk.url:
        return f"{chunk.doc_id}:{chunk.url}:{chunk.chunk_index}"
    return f"{chunk.doc_id}:{chunk.chunk_index}"


class Retriever:
    """Compatibility facade over the typed asynchronous retrieval engine.

    Production async callers should use ``query()`` or ``aretrieve()``. The
    historical synchronous ``retrieve()`` API remains for CLI/tests and runs the
    same engine. Blocking PostgreSQL/Qdrant/reranker calls share one long-lived
    executor owned by the Retriever; hybrid requests never allocate a per-call
    ThreadPoolExecutor.
    """

    def __init__(
        self,
        repo: Repo,
        qdrant: Any,
        embedder: Optional[Embedder] = None,
        reranker: Any = None,
        *,
        blocking_workers: int = 8,
    ) -> None:
        self._repo = repo
        self._qdrant = qdrant
        self._embedder = embedder or HashEmbedder(dim=qdrant.dim)
        self._reranker = reranker
        self._executor = ThreadPoolExecutor(
            max_workers=max(2, int(blocking_workers)),
            thread_name_prefix="ragbot-retrieval",
        )
        self._engine = AsyncRetrievalEngine(
            repo,
            qdrant,
            self._embedder,
            reranker,
            self._executor,
        )

    def diagnostics(
        self,
        query: Optional[str] = None,
        retrieval_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        semantic = not isinstance(self._embedder, HashEmbedder)
        reranker_configured = bool(
            self._reranker
            and hasattr(self._reranker, "enabled")
            and self._reranker.enabled
        )
        warnings: List[str] = []
        if not semantic:
            warnings.append(
                "HashEmbedder is a development fallback, not a semantic embedding model. "
                "Configure EMBEDDING_MODEL and a real embedding endpoint, then re-ingest before "
                "judging semantic retrieval quality."
            )
            if query and contains_cjk(query):
                warnings.append(
                    "The current CJK query is not meaningfully represented by HashEmbedder; "
                    "the invalid hash-vector branch is disabled for this query. Cross-lingual "
                    "retrieval requires a multilingual semantic embedding model."
                )
        result: Dict[str, Any] = {
            "embedding_backend": type(self._embedder).__name__,
            "embedding_model": self._embedder.model_name,
            "embedding_dimension": self._embedder.dimension,
            "semantic_embedding": semantic,
            "vector_store": type(self._qdrant).__name__,
            "repository": type(self._repo).__name__,
            "reranker": type(self._reranker).__name__ if self._reranker is not None else None,
            "reranker_configured": reranker_configured,
            "reranker_enabled": reranker_configured,
            "fusion_mode": "adaptive-hybrid" if semantic else "lexical-first-development",
            "generation_visibility": callable(getattr(self._repo, "active_vector_points", None)),
            "async_retrieval": True,
            "blocking_executor_workers": self._executor._max_workers,
            "retrieval_plans": [item.value for item in RetrievalPlan],
            "warnings": warnings,
        }
        if retrieval_context:
            result.update(retrieval_context)
        return result

    async def query(self, request: RetrievalRequest) -> RetrievalResponse:
        response = await self._engine.execute(request)
        # Backward-compatible trace keys used by the existing workbench/eval
        # surface. `dense` is the new plan vocabulary; `vector` remains an alias
        # until downstream consumers migrate.
        for chunk in response.chunks:
            metadata = chunk.metadata or {}
            trace = metadata.get("_retrieval")
            if not isinstance(trace, dict):
                continue
            trace.setdefault("vector", trace.get("dense"))
            trace.setdefault("lexical", None)
        return response

    async def aretrieve(
        self,
        query: str,
        filters: Dict[str, Any],
        top_k: int = 20,
        *,
        mode: str = "hybrid",
        plan: str | RetrievalPlan | None = None,
        candidate_pool: Optional[int] = None,
        rerank: bool = True,
        deadline_ms: Optional[int] = None,
        diversity: bool = False,
    ) -> List[RetrievalChunk]:
        request = RetrievalRequest(
            query=query,
            filters=dict(filters),
            top_k=top_k,
            plan=resolve_retrieval_plan(plan, legacy_mode=mode),
            candidate_pool=candidate_pool,
            rerank=rerank,
            deadline_ms=deadline_ms,
            diversity=diversity,
        )
        return (await self.query(request)).chunks

    def retrieve(
        self,
        query: str,
        filters: Dict[str, Any],
        top_k: int = 20,
        *,
        mode: str = "hybrid",
        plan: str | RetrievalPlan | None = None,
        candidate_pool: Optional[int] = None,
        rerank: bool = True,
        deadline_ms: Optional[int] = None,
        diversity: bool = False,
    ) -> List[RetrievalChunk]:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(
                self.aretrieve(
                    query,
                    filters,
                    top_k=top_k,
                    mode=mode,
                    plan=plan,
                    candidate_pool=candidate_pool,
                    rerank=rerank,
                    deadline_ms=deadline_ms,
                    diversity=diversity,
                )
            )
        raise RuntimeError(
            "Retriever.retrieve() cannot block inside a running event loop; "
            "use await Retriever.aretrieve() or await Retriever.query()"
        )

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
