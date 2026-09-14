from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import Executor
from functools import partial
from typing import Any, Dict, Iterable, List, Optional, Protocol, Sequence

from contracts.types import RetrievalChunk

from ..storage.models import Chunk
from .contracts import (
    Candidate,
    RetrievalDeadlineExceeded,
    RetrievalPlan,
    RetrievalRequest,
    RetrievalResponse,
    RetrievalTrace,
    UnsupportedRetrievalPlan,
)
from .embedder import Embedder, HashEmbedder
from .lexical import contains_cjk
from .pg_fts import fts_search
from .policy import adaptive_fusion_policy, resolve_candidate_pool
from .rerank import rrf_fuse

logger = logging.getLogger(__name__)


class CandidateRetriever(Protocol):
    name: str

    async def retrieve(
        self,
        request: RetrievalRequest,
        *,
        pool_size: int,
        budget: "QueryBudget",
        trace: RetrievalTrace,
    ) -> List[Candidate]: ...


class QueryBudget:
    def __init__(self, deadline_ms: Optional[int]) -> None:
        self.deadline_ms = int(deadline_ms) if deadline_ms is not None else None
        self.started = time.monotonic()
        self.deadline = (
            self.started + (self.deadline_ms / 1000.0)
            if self.deadline_ms is not None
            else None
        )

    def remaining_seconds(self) -> Optional[float]:
        if self.deadline is None:
            return None
        return max(0.0, self.deadline - time.monotonic())


async def _await_stage(
    stage: str,
    awaitable: Any,
    *,
    budget: QueryBudget,
    trace: RetrievalTrace,
) -> Any:
    started = time.perf_counter()
    try:
        remaining = budget.remaining_seconds()
        if remaining is not None:
            if remaining <= 0:
                raise asyncio.TimeoutError
            return await asyncio.wait_for(awaitable, timeout=remaining)
        return await awaitable
    except asyncio.TimeoutError as exc:
        trace.timed_out = True
        trace.error_stage = stage
        raise RetrievalDeadlineExceeded(
            f"Retrieval deadline exceeded during stage={stage}", trace
        ) from exc
    except asyncio.CancelledError:
        trace.cancelled = True
        trace.error_stage = stage
        raise
    finally:
        trace.stage_ms[stage] = (time.perf_counter() - started) * 1000.0


async def _run_blocking(executor: Executor, func: Any, *args: Any, **kwargs: Any) -> Any:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(executor, partial(func, *args, **kwargs))


class DenseCandidateRetriever:
    name = "dense"

    def __init__(self, repo: Any, vector_store: Any, embedder: Embedder, executor: Executor) -> None:
        self._repo = repo
        self._vector_store = vector_store
        self._embedder = embedder
        self._executor = executor

    async def retrieve(
        self,
        request: RetrievalRequest,
        *,
        pool_size: int,
        budget: QueryBudget,
        trace: RetrievalTrace,
    ) -> List[Candidate]:
        if isinstance(self._embedder, HashEmbedder) and contains_cjk(request.query):
            trace.candidate_counts[self.name] = 0
            return []

        active_points = getattr(self._repo, "active_vector_points", None)
        generation_visibility = callable(active_points)
        vector_limit = min(1000, max(pool_size, pool_size * 4)) if generation_visibility else pool_size

        async def embed() -> List[float]:
            method = getattr(self._embedder, "aembed_query", None)
            if callable(method):
                return list(await method(request.query))
            sync = getattr(self._embedder, "embed_query", None)
            if callable(sync):
                return list(await _run_blocking(self._executor, sync, request.query))
            return list(await _run_blocking(self._executor, self._embedder.embed, request.query))

        vector = await _await_stage(
            "dense.embed", embed(), budget=budget, trace=trace
        )
        hits = await _await_stage(
            "dense.search",
            _run_blocking(
                self._executor,
                self._vector_store.search,
                vector,
                request.filters,
                vector_limit,
            ),
            budget=budget,
            trace=trace,
        )

        if generation_visibility and hits:
            logical_ids = [
                str(payload.get("chunk_id") or point_id)
                for point_id, _score, payload in hits
            ]
            visible_points = await _await_stage(
                "dense.visibility",
                _run_blocking(self._executor, active_points, logical_ids),
                budget=budget,
                trace=trace,
            )
            visible = []
            for point_id, score, payload in hits:
                logical_id = str(payload.get("chunk_id") or point_id)
                expected = visible_points.get(logical_id)
                if expected is None or str(point_id) != str(expected):
                    continue
                visible.append((point_id, score, payload))
                if len(visible) >= pool_size:
                    break
            hits = visible
        else:
            hits = list(hits[:pool_size])

        candidates = [
            Candidate(
                chunk_id=str(payload.get("chunk_id") or point_id),
                score=float(score),
                source=self.name,
                rank=rank,
                payload={**dict(payload or {}), "_physical_point_id": str(point_id)},
                raw=(point_id, score, payload),
            )
            for rank, (point_id, score, payload) in enumerate(hits, 1)
        ]
        trace.candidate_counts[self.name] = len(candidates)
        return candidates


class LexicalCandidateRetriever:
    name = "lexical"

    def __init__(self, repo: Any, executor: Executor) -> None:
        self._repo = repo
        self._executor = executor

    async def retrieve(
        self,
        request: RetrievalRequest,
        *,
        pool_size: int,
        budget: QueryBudget,
        trace: RetrievalTrace,
    ) -> List[Candidate]:
        hits = await _await_stage(
            "lexical.search",
            _run_blocking(
                self._executor,
                fts_search,
                self._repo,
                request.query,
                request.filters,
                pool_size,
            ),
            budget=budget,
            trace=trace,
        )
        candidates = [
            Candidate(
                chunk_id=chunk.chunk_id,
                score=float(score),
                source=self.name,
                rank=rank,
                raw=chunk,
            )
            for rank, (chunk, score) in enumerate(hits, 1)
        ]
        trace.candidate_counts[self.name] = len(candidates)
        return candidates


class QdrantDenseSparseCandidateRetriever:
    """Experimental native Qdrant hybrid port.

    #57 Phase 1 freezes the plan identifier and backend contract without silently
    promoting a sparse representation. The active IndexVersion must explicitly
    provide a backend `native_hybrid_search()` implementation backed by named
    dense+sparse vectors before this plan can execute.
    """

    name = "qdrant_dense_sparse"

    def __init__(self, vector_store: Any, embedder: Embedder, executor: Executor) -> None:
        self._vector_store = vector_store
        self._embedder = embedder
        self._executor = executor

    async def retrieve(
        self,
        request: RetrievalRequest,
        *,
        pool_size: int,
        budget: QueryBudget,
        trace: RetrievalTrace,
    ) -> List[Candidate]:
        native = getattr(self._vector_store, "native_hybrid_search", None)
        if not callable(native):
            raise UnsupportedRetrievalPlan(
                "qdrant_dense_sparse requires an IndexVersion with named dense+sparse "
                "vectors and a vector backend exposing native_hybrid_search(); the "
                "current dense index remains the control plan"
            )
        method = getattr(self._embedder, "aembed_query", None)
        if callable(method):
            dense = await _await_stage(
                "qdrant_dense_sparse.embed", method(request.query), budget=budget, trace=trace
            )
        else:
            dense = await _await_stage(
                "qdrant_dense_sparse.embed",
                _run_blocking(self._executor, self._embedder.embed_query, request.query),
                budget=budget,
                trace=trace,
            )
        hits = await _await_stage(
            "qdrant_dense_sparse.search",
            _run_blocking(
                self._executor,
                native,
                request.query,
                list(dense),
                request.filters,
                pool_size,
            ),
            budget=budget,
            trace=trace,
        )
        candidates = [
            Candidate(
                chunk_id=str(payload.get("chunk_id") or point_id),
                score=float(score),
                source=self.name,
                rank=rank,
                payload={**dict(payload or {}), "_physical_point_id": str(point_id)},
                raw=(point_id, score, payload),
            )
            for rank, (point_id, score, payload) in enumerate(hits, 1)
        ]
        trace.candidate_counts[self.name] = len(candidates)
        return candidates


class AsyncRetrievalEngine:
    def __init__(
        self,
        repo: Any,
        vector_store: Any,
        embedder: Embedder,
        reranker: Any,
        executor: Executor,
    ) -> None:
        self._repo = repo
        self._vector_store = vector_store
        self._embedder = embedder
        self._reranker = reranker
        self._executor = executor
        self._dense = DenseCandidateRetriever(repo, vector_store, embedder, executor)
        self._lexical = LexicalCandidateRetriever(repo, executor)
        self._native = QdrantDenseSparseCandidateRetriever(vector_store, embedder, executor)

    async def execute(self, request: RetrievalRequest) -> RetrievalResponse:
        pool_size = resolve_candidate_pool(request.top_k, request.candidate_pool)
        budget = QueryBudget(request.deadline_ms)
        trace = RetrievalTrace(
            plan=request.plan.value,
            deadline_ms=request.deadline_ms,
            candidate_pool=pool_size,
            started_monotonic=budget.started,
            reranker_requested=bool(request.rerank),
            diversity_enabled=bool(request.diversity),
        )

        if request.plan is RetrievalPlan.HYBRID_RRF:
            dense_task = asyncio.create_task(
                self._dense.retrieve(request, pool_size=pool_size, budget=budget, trace=trace)
            )
            lexical_task = asyncio.create_task(
                self._lexical.retrieve(request, pool_size=pool_size, budget=budget, trace=trace)
            )
            try:
                dense, lexical = await _await_stage(
                    "candidates.parallel",
                    asyncio.gather(dense_task, lexical_task),
                    budget=budget,
                    trace=trace,
                )
            except BaseException:
                for task in (dense_task, lexical_task):
                    if not task.done():
                        task.cancel()
                await asyncio.gather(dense_task, lexical_task, return_exceptions=True)
                raise
        elif request.plan is RetrievalPlan.DENSE:
            dense = await self._dense.retrieve(
                request, pool_size=pool_size, budget=budget, trace=trace
            )
            lexical = []
        elif request.plan is RetrievalPlan.LEXICAL:
            dense = []
            lexical = await self._lexical.retrieve(
                request, pool_size=pool_size, budget=budget, trace=trace
            )
        elif request.plan is RetrievalPlan.QDRANT_DENSE_SPARSE:
            dense = await self._native.retrieve(
                request, pool_size=pool_size, budget=budget, trace=trace
            )
            lexical = []
        else:  # pragma: no cover - enum exhaustiveness
            raise UnsupportedRetrievalPlan(str(request.plan))

        ranked, source_trace, payload_map, chunk_map = await self._fuse(
            request, dense, lexical, trace=trace
        )
        ranked, rerank_scores = await self._rerank(
            request, ranked, payload_map, chunk_map, pool_size=pool_size, budget=budget, trace=trace
        )
        if request.diversity:
            started = time.perf_counter()
            ranked = _diversify(ranked, payload_map, chunk_map)
            trace.stage_ms["diversity"] = (time.perf_counter() - started) * 1000.0

        chunks = _materialize_results(
            ranked[: request.top_k],
            source_trace=source_trace,
            payload_map=payload_map,
            chunk_map=chunk_map,
            rerank_scores=rerank_scores,
            embedder=self._embedder,
            trace=trace,
        )
        return RetrievalResponse(chunks=chunks, trace=trace)

    async def _fuse(
        self,
        request: RetrievalRequest,
        dense: Sequence[Candidate],
        lexical: Sequence[Candidate],
        *,
        trace: RetrievalTrace,
    ) -> tuple[
        List[tuple[str, float]],
        Dict[str, Dict[str, Any]],
        Dict[str, Dict[str, Any]],
        Dict[str, Chunk],
    ]:
        started = time.perf_counter()
        source_trace: Dict[str, Dict[str, Any]] = {}
        payload_map: Dict[str, Dict[str, Any]] = {}
        chunk_map: Dict[str, Chunk] = {}

        dense_ranked = []
        for candidate in dense:
            dense_ranked.append((candidate.chunk_id, candidate.score))
            payload_map[candidate.chunk_id] = dict(candidate.payload)
            source_trace.setdefault(candidate.chunk_id, {})[candidate.source] = {
                "rank": candidate.rank,
                "score": candidate.score,
                "raw_score": candidate.score,
                "physical_point_id": candidate.payload.get("_physical_point_id"),
                "generation_id": candidate.payload.get("generation_id"),
            }

        lexical_ranked = []
        for candidate in lexical:
            lexical_ranked.append((candidate.chunk_id, candidate.score))
            if isinstance(candidate.raw, Chunk):
                chunk_map[candidate.chunk_id] = candidate.raw
            source_trace.setdefault(candidate.chunk_id, {})[candidate.source] = {
                "rank": candidate.rank,
                "score": candidate.score,
                "raw_score": candidate.score,
            }

        if request.plan is RetrievalPlan.HYBRID_RRF:
            qdrant_hits = [candidate.raw for candidate in dense if candidate.raw is not None]
            fts_hits = [
                (candidate.raw, candidate.score)
                for candidate in lexical
                if isinstance(candidate.raw, Chunk)
            ]
            policy = adaptive_fusion_policy(
                request.query,
                qdrant_hits,
                fts_hits,
                hash_fallback=isinstance(self._embedder, HashEmbedder),
            )
            ranked = rrf_fuse(
                dense_ranked,
                lexical_ranked,
                weight_primary=policy.vector_weight,
                weight_secondary=policy.lexical_weight,
            )
            trace.fusion_method = "adaptive-rrf"
            trace.fusion_policy = policy.as_dict()
        elif request.plan is RetrievalPlan.LEXICAL:
            ranked = list(lexical_ranked)
            trace.fusion_method = "lexical-only"
            trace.fusion_policy = {"vector_weight": 0.0, "lexical_weight": 1.0}
        elif request.plan is RetrievalPlan.DENSE:
            ranked = list(dense_ranked)
            trace.fusion_method = "dense-only"
            trace.fusion_policy = {"vector_weight": 1.0, "lexical_weight": 0.0}
        else:
            ranked = list(dense_ranked)
            trace.fusion_method = "qdrant-native"
            trace.fusion_policy = {"backend": "qdrant", "native_hybrid": True}

        for chunk_id, score in ranked:
            source_trace.setdefault(chunk_id, {})["fusion_score"] = float(score)
        trace.stage_ms["fusion"] = (time.perf_counter() - started) * 1000.0
        return ranked, source_trace, payload_map, chunk_map

    async def _rerank(
        self,
        request: RetrievalRequest,
        ranked: List[tuple[str, float]],
        payload_map: Dict[str, Dict[str, Any]],
        chunk_map: Dict[str, Chunk],
        *,
        pool_size: int,
        budget: QueryBudget,
        trace: RetrievalTrace,
    ) -> tuple[List[tuple[str, float]], Dict[str, float]]:
        configured = bool(
            self._reranker
            and hasattr(self._reranker, "enabled")
            and self._reranker.enabled
        )
        trace.reranker_configured = configured
        trace.reranker_enabled = bool(request.rerank and configured)
        rerank_scores: Dict[str, float] = {}
        if not trace.reranker_enabled or not ranked:
            return ranked, rerank_scores

        candidates = ranked[:pool_size]
        trace.reranker_candidate_count = len(candidates)
        texts = [
            chunk_map[cid].text
            if cid in chunk_map
            else str(payload_map.get(cid, {}).get("text", ""))
            for cid, _score in candidates
        ]
        try:
            reranked = await _await_stage(
                "rerank",
                _run_blocking(
                    self._executor,
                    self._reranker.rerank,
                    request.query,
                    texts,
                    top_k=request.top_k,
                ),
                budget=budget,
                trace=trace,
            )
        except RetrievalDeadlineExceeded:
            raise
        except Exception:
            logger.exception("Optional reranker failed; falling back to fused ordering")
            return ranked, rerank_scores

        valid: List[tuple[str, float]] = []
        for index, score in reranked:
            if isinstance(index, int) and 0 <= index < len(candidates):
                chunk_id = candidates[index][0]
                rerank_scores[chunk_id] = float(score)
                valid.append((chunk_id, float(score)))
        return (valid or ranked), rerank_scores


def _materialize_results(
    ranked: Sequence[tuple[str, float]],
    *,
    source_trace: Dict[str, Dict[str, Any]],
    payload_map: Dict[str, Dict[str, Any]],
    chunk_map: Dict[str, Chunk],
    rerank_scores: Dict[str, float],
    embedder: Embedder,
    trace: RetrievalTrace,
) -> List[RetrievalChunk]:
    results: List[RetrievalChunk] = []
    context = trace.as_dict()
    for final_rank, (chunk_id, final_score) in enumerate(ranked, 1):
        candidate_trace = dict(source_trace.get(chunk_id, {}))
        candidate_trace.update(
            {
                "final_rank": final_rank,
                "final_score": float(final_score),
                "rerank_score": rerank_scores.get(chunk_id),
                "embedding_model": embedder.model_name,
                "fusion_mode": trace.fusion_method,
                "context": context,
            }
        )
        chunk = chunk_map.get(chunk_id)
        if chunk is not None:
            metadata = dict(chunk.metadata or {})
            metadata.update(
                {
                    "doc_id": chunk.doc_id,
                    "chunk_index": chunk.chunk_index,
                    "path": chunk.path,
                    "url": chunk.url,
                    "page": chunk.page,
                    "section": chunk.section,
                    "qdrant_point_id": chunk.qdrant_point_id,
                    "source_id": chunk.source_id,
                    "generation_id": chunk.generation_id,
                    "embedding_model": embedder.model_name,
                    "_retrieval": candidate_trace,
                }
            )
            results.append(
                RetrievalChunk(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    text=chunk.text,
                    score=float(final_score),
                    citations=[_chunk_citation(chunk)],
                    metadata=metadata,
                )
            )
            continue

        payload = dict(payload_map.get(chunk_id, {}))
        if not payload:
            continue
        payload.pop("_physical_point_id", None)
        payload["_retrieval"] = candidate_trace
        logical_id = str(payload.get("chunk_id") or chunk_id)
        results.append(
            RetrievalChunk(
                chunk_id=logical_id,
                doc_id=str(payload.get("doc_id") or "unknown"),
                text=str(payload.get("text") or ""),
                score=float(final_score),
                citations=[_payload_citation(payload, logical_id)],
                metadata=payload,
            )
        )
    return results


def _diversify(
    ranked: Sequence[tuple[str, float]],
    payload_map: Dict[str, Dict[str, Any]],
    chunk_map: Dict[str, Chunk],
    *,
    threshold: float = 0.92,
) -> List[tuple[str, float]]:
    """Cheap optional near-duplicate suppression, intentionally not default MMR.

    This stage only establishes the diversity port for Phase 1. Semantic MMR is
    benchmark-gated separately because it changes relevance/cost characteristics.
    """
    selected: List[tuple[str, float]] = []
    token_sets: List[set[str]] = []
    for chunk_id, score in ranked:
        text = (
            chunk_map[chunk_id].text
            if chunk_id in chunk_map
            else str(payload_map.get(chunk_id, {}).get("text", ""))
        )
        tokens = set(text.lower().split())
        duplicate = False
        if tokens:
            for other in token_sets:
                union = tokens | other
                if union and (len(tokens & other) / len(union)) >= threshold:
                    duplicate = True
                    break
        if duplicate:
            continue
        selected.append((chunk_id, score))
        token_sets.append(tokens)
    return selected


def _chunk_citation(chunk: Chunk) -> str:
    location = chunk.path or chunk.url
    if location and chunk.page is not None:
        return f"{chunk.doc_id}:{location}:page={chunk.page}:chunk={chunk.chunk_index}"
    if chunk.path:
        return f"{chunk.doc_id}:{chunk.path}:{chunk.chunk_index}"
    if chunk.url:
        return f"{chunk.doc_id}:{chunk.url}:{chunk.chunk_index}"
    return f"{chunk.doc_id}:{chunk.chunk_index}"


def _payload_citation(payload: Dict[str, Any], logical_id: str) -> str:
    doc_id = str(payload.get("doc_id") or "unknown")
    location = payload.get("path") or payload.get("url")
    page = payload.get("page")
    chunk_index = payload.get("chunk_index")
    if location and page is not None:
        return f"{doc_id}:{location}:page={page}:chunk={chunk_index}"
    if location:
        return f"{doc_id}:{location}:{chunk_index}"
    return logical_id
