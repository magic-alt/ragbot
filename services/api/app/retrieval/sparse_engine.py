from __future__ import annotations

import asyncio
from concurrent.futures import Executor
from typing import Any, Optional

from .async_engine import (
    AsyncRetrievalEngine,
    QueryBudget,
    _await_stage,
    _diversify,
    _materialize_results,
    _run_blocking,
)
from .contracts import (
    Candidate,
    RetrievalPlan,
    RetrievalRequest,
    RetrievalResponse,
    RetrievalTrace,
    UnsupportedRetrievalPlan,
)
from .embedding_contract import embedding_contract_id
from .policy import resolve_candidate_pool
from .sparse import SparseEncoder, sparse_contract_from_index


class SparseAwareRetrievalEngine(AsyncRetrievalEngine):
    """Phase-2 engine extension for controlled Qdrant dense+sparse candidates.

    All existing plans delegate to the Phase-1 engine unchanged. The native
    dense+sparse path may target a validating IndexVersion by ID so it can be
    benchmarked before alias activation. Weighted RRF is a query-time fusion
    contract and therefore reuses the same physical IndexVersion.
    """

    def __init__(
        self,
        repo: Any,
        vector_store: Any,
        embedder: Any,
        reranker: Any,
        executor: Executor,
        *,
        sparse_encoder: Optional[SparseEncoder] = None,
    ) -> None:
        super().__init__(repo, vector_store, embedder, reranker, executor)
        self._sparse_repo = repo
        self._sparse_vector_store = vector_store
        self._sparse_embedder = embedder
        self._sparse_encoder = sparse_encoder
        self._sparse_executor = executor

    async def execute(self, request: RetrievalRequest) -> RetrievalResponse:
        if request.plan is not RetrievalPlan.QDRANT_DENSE_SPARSE:
            return await super().execute(request)
        return await self._execute_dense_sparse(request)

    async def _execute_dense_sparse(self, request: RetrievalRequest) -> RetrievalResponse:
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
        version, dense_name, sparse_name, sparse_contract = self._resolve_index(request)
        trace.index_version_id = version.index_version_id
        trace.representation_contracts = {
            "dense": version.embedding_contract_id,
            "sparse": sparse_contract,
        }

        async def dense_embed() -> list[float]:
            method = getattr(self._sparse_embedder, "aembed_query", None)
            if callable(method):
                return list(await method(request.query))
            sync = getattr(self._sparse_embedder, "embed_query", None)
            if callable(sync):
                return list(
                    await _run_blocking(self._sparse_executor, sync, request.query)
                )
            return list(
                await _run_blocking(
                    self._sparse_executor, self._sparse_embedder.embed, request.query
                )
            )

        dense = await _await_stage(
            "qdrant_dense_sparse.dense_embed",
            dense_embed(),
            budget=budget,
            trace=trace,
        )
        sparse = await _await_stage(
            "qdrant_dense_sparse.sparse_embed",
            _run_blocking(
                self._sparse_executor,
                self._sparse_encoder.embed_query,  # type: ignore[union-attr]
                request.query,
            ),
            budget=budget,
            trace=trace,
        )
        native = getattr(self._sparse_vector_store, "native_hybrid_search", None)
        if not callable(native):
            raise UnsupportedRetrievalPlan(
                "qdrant_dense_sparse requires a Qdrant backend exposing native_hybrid_search()"
            )
        fusion_spec = request.fusion_spec
        native_kwargs = {
            "collection_name": version.physical_collection,
            "dense_name": dense_name,
            "sparse_name": sparse_name,
            "prefetch_limit": max(pool_size, pool_size * 4),
        }
        # Keep the Phase-2 backend seam byte-for-byte compatible when no
        # explicit fusion experiment is selected. Third-party/fake backends
        # written against the original native_hybrid_search signature therefore
        # never receive new weighted-RRF keyword arguments unless requested.
        if fusion_spec is not None:
            native_kwargs.update(
                {
                    "rrf_weights": fusion_spec.weights,
                    "rrf_k": fusion_spec.k,
                }
            )
        hits = await _await_stage(
            "qdrant_dense_sparse.search",
            _run_blocking(
                self._sparse_executor,
                native,
                dense,
                sparse,
                request.filters,
                pool_size,
                **native_kwargs,
            ),
            budget=budget,
            trace=trace,
        )
        hits = await self._apply_generation_visibility(hits, pool_size, budget, trace)
        candidates = [
            Candidate(
                chunk_id=str(payload.get("chunk_id") or point_id),
                score=float(score),
                source="qdrant_dense_sparse",
                rank=rank,
                payload={**dict(payload or {}), "_physical_point_id": str(point_id)},
                raw=(point_id, score, payload),
            )
            for rank, (point_id, score, payload) in enumerate(hits, 1)
        ]
        trace.candidate_counts["qdrant_dense_sparse"] = len(candidates)

        ranked, source_trace, payload_map, chunk_map = await self._fuse(
            request, candidates, [], trace=trace
        )
        # `_fuse()` materializes the already fused Qdrant order and keeps the
        # Phase-2 qdrant-native compatibility trace. An explicit fusion spec
        # overrides only the trace contract, not ranking a second time.
        if fusion_spec is not None:
            trace.fusion_method = (
                "qdrant-weighted-rrf" if fusion_spec.weighted else "qdrant-rrf"
            )
            trace.fusion_policy = fusion_spec.as_dict()
        ranked, rerank_scores = await self._rerank(
            request,
            ranked,
            payload_map,
            chunk_map,
            pool_size=pool_size,
            budget=budget,
            trace=trace,
        )
        if request.diversity:
            import time

            started = time.perf_counter()
            ranked = _diversify(ranked, payload_map, chunk_map)
            trace.stage_ms["diversity"] = (time.perf_counter() - started) * 1000.0
        chunks = _materialize_results(
            ranked[: request.top_k],
            source_trace=source_trace,
            payload_map=payload_map,
            chunk_map=chunk_map,
            rerank_scores=rerank_scores,
            embedder=self._sparse_embedder,
            trace=trace,
        )
        return RetrievalResponse(chunks=chunks, trace=trace)

    def _resolve_index(self, request: RetrievalRequest):
        if self._sparse_encoder is None:
            raise UnsupportedRetrievalPlan(
                "qdrant_dense_sparse requires named dense+sparse index capability and a configured "
                "sparse encoder; set RAGBOT_SPARSE_ENABLED=true and build a sparse IndexVersion"
            )
        alias = getattr(self._sparse_vector_store, "alias_name", None)
        getter = getattr(self._sparse_repo, "get_index_version", None)
        active_getter = getattr(self._sparse_repo, "get_active_index_version", None)
        if request.index_version_id:
            version = getter(request.index_version_id) if callable(getter) else None
        elif alias and callable(active_getter):
            version = active_getter(alias)
        else:
            version = None
        if version is None:
            raise UnsupportedRetrievalPlan(
                "qdrant_dense_sparse requires a registered IndexVersion"
            )
        if getattr(version, "status", None) not in {"validating", "ready", "active"}:
            raise UnsupportedRetrievalPlan(
                f"IndexVersion is not queryable for evaluation: {getattr(version, 'status', None)}"
            )
        current_dense_contract = embedding_contract_id(self._sparse_embedder)
        if str(version.embedding_contract_id) != str(current_dense_contract):
            raise UnsupportedRetrievalPlan(
                "Phase-2 dense+sparse evaluation keeps the active dense embedding contract fixed; "
                f"index={version.embedding_contract_id} runtime={current_dense_contract}"
            )
        sparse_schema = sparse_contract_from_index(version)
        if not sparse_schema:
            raise UnsupportedRetrievalPlan(
                "Selected IndexVersion has no sparse vector contract"
            )
        sparse_contract = str(sparse_schema.get("contract_id") or "")
        if sparse_contract != self._sparse_encoder.contract_id:
            raise UnsupportedRetrievalPlan(
                "Configured sparse encoder does not match IndexVersion: "
                f"index={sparse_contract} runtime={self._sparse_encoder.contract_id}"
            )
        dense_schema = dict((version.vector_schema or {}).get("dense") or {})
        dense_name = str(dense_schema.get("name") or "dense")
        sparse_name = str(sparse_schema.get("name") or sparse_schema.get("vector_name") or "sparse")
        return version, dense_name, sparse_name, sparse_contract

    async def _apply_generation_visibility(
        self,
        hits: Any,
        pool_size: int,
        budget: QueryBudget,
        trace: RetrievalTrace,
    ) -> list[Any]:
        values = list(hits or [])
        active_points = getattr(self._sparse_repo, "active_vector_points", None)
        if not callable(active_points) or not values:
            return values[:pool_size]
        logical_ids = [
            str(payload.get("chunk_id") or point_id)
            for point_id, _score, payload in values
        ]
        visible_points = await _await_stage(
            "qdrant_dense_sparse.visibility",
            _run_blocking(self._sparse_executor, active_points, logical_ids),
            budget=budget,
            trace=trace,
        )
        visible = []
        for point_id, score, payload in values:
            logical_id = str(payload.get("chunk_id") or point_id)
            expected = visible_points.get(logical_id)
            if expected is None or str(point_id) != str(expected):
                continue
            visible.append((point_id, score, payload))
            if len(visible) >= pool_size:
                break
        return visible
