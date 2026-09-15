from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from benchmarks.weighted_rrf_promotion import (
    _scoped_dataset,
    evaluate_weighted_rrf_grid,
    parse_weight_pairs,
)
from contracts.types import RetrievalChunk
from services.api.app.quality.contracts import PromotionPolicy
from services.api.app.retrieval.contracts import (
    QdrantRrfFusionSpec,
    RetrievalPlan,
    RetrievalRequest,
    RetrievalResponse,
    RetrievalTrace,
)
from services.api.app.retrieval.embedder import HashEmbedder
from services.api.app.retrieval.sparse import DeterministicSparseEncoder
from services.api.app.retrieval.sparse_engine import SparseAwareRetrievalEngine
from services.api.app.storage.index_support import ensure_index_repository
from services.api.app.storage.models import IndexVersion
from services.api.app.storage.quality_support import ensure_quality_repository
from services.api.app.storage.repo import InMemoryRepo


def test_fusion_contract_is_content_derived_and_validated() -> None:
    equal = QdrantRrfFusionSpec()
    dense_first = QdrantRrfFusionSpec(dense_weight=3, sparse_weight=1)
    assert equal.contract_id.startswith("fusion-")
    assert equal.contract_id != dense_first.contract_id
    assert dense_first.as_dict()["weights"] == [3.0, 1.0]
    assert dense_first.as_dict()["prefetch_order"] == ["dense", "sparse"]
    with pytest.raises(ValueError, match="weights must be > 0"):
        QdrantRrfFusionSpec(dense_weight=0, sparse_weight=1)
    with pytest.raises(ValueError, match="k must be > 0"):
        QdrantRrfFusionSpec(k=0)


def test_weight_grid_parser_deduplicates_pairs() -> None:
    specs = parse_weight_pairs("1:1,2:1,3:1,3:1,5:1")
    assert [spec.weights for spec in specs] == [
        (1.0, 1.0),
        (2.0, 1.0),
        (3.0, 1.0),
        (5.0, 1.0),
    ]


def test_fusion_spec_is_internal_to_qdrant_dense_sparse_plan() -> None:
    with pytest.raises(ValueError, match="supported only by qdrant_dense_sparse"):
        RetrievalRequest(
            query="servo",
            filters={},
            plan=RetrievalPlan.DENSE,
            fusion_spec=QdrantRrfFusionSpec(3, 1),
        )


class _CaptureVectorStore:
    alias_name = "rag-active"

    def __init__(self) -> None:
        self.calls = []

    def native_hybrid_search(self, dense, sparse, filters, top_k, **kwargs):
        self.calls.append({"filters": filters, "top_k": top_k, **kwargs})
        return [
            (
                "point-1",
                0.9,
                {
                    "chunk_id": "chunk-1",
                    "doc_id": "doc-1",
                    "text": "servo evidence",
                    "tenant_id": "tenant-a",
                },
            )
        ]


class _IndexRepo:
    def __init__(self, version: IndexVersion) -> None:
        self.version = version

    def get_index_version(self, index_version_id: str):
        return self.version if index_version_id == self.version.index_version_id else None

    def get_active_index_version(self, alias_name: str):
        return None


def test_sparse_engine_passes_weighted_rrf_to_backend_and_trace() -> None:
    dense = HashEmbedder(dim=4)
    sparse = DeterministicSparseEncoder()
    version = IndexVersion(
        index_version_id="idx-candidate",
        alias_name="rag-active",
        physical_collection="candidate",
        embedding_contract_id=dense.contract_id,
        embedding_spec={"dimension": 4},
        vector_schema={
            "dense": {"name": "dense", "dimension": 4, "distance": "cosine"},
            "sparse": {"name": "sparse", "contract_id": sparse.contract_id},
        },
        status="validating",
    )
    store = _CaptureVectorStore()
    executor = ThreadPoolExecutor(max_workers=2)
    engine = SparseAwareRetrievalEngine(
        _IndexRepo(version), store, dense, None, executor, sparse_encoder=sparse
    )
    spec = QdrantRrfFusionSpec(dense_weight=3, sparse_weight=1, k=2)
    try:
        response = asyncio.run(
            engine.execute(
                RetrievalRequest(
                    query="servo tuning",
                    filters={"tenant_id": "tenant-a"},
                    top_k=1,
                    plan=RetrievalPlan.QDRANT_DENSE_SPARSE,
                    rerank=False,
                    index_version_id=version.index_version_id,
                    fusion_spec=spec,
                )
            )
        )
    finally:
        executor.shutdown(wait=True)
    assert store.calls[0]["rrf_weights"] == (3.0, 1.0)
    assert store.calls[0]["rrf_k"] == 2
    assert response.trace.fusion_method == "qdrant-weighted-rrf"
    assert response.trace.fusion_policy["contract_id"] == spec.contract_id
    assert response.trace.fusion_policy["weights"] == [3.0, 1.0]


class _GridRetriever:
    async def query(self, request: RetrievalRequest) -> RetrievalResponse:
        case = request.query.split()[0]
        fusion = request.fusion_spec
        trace = RetrievalTrace(
            plan=request.plan.value,
            deadline_ms=request.deadline_ms,
            candidate_pool=10,
            started_monotonic=0.0,
            index_version_id=request.index_version_id,
            representation_contracts={"dense": "emb-a", "sparse": "sparse-a"}
            if request.plan is RetrievalPlan.QDRANT_DENSE_SPARSE
            else {"dense": "emb-a"},
        )
        if fusion is not None:
            trace.fusion_method = "qdrant-weighted-rrf" if fusion.weighted else "qdrant-rrf"
            trace.fusion_policy = fusion.as_dict()
        else:
            trace.fusion_method = "adaptive-rrf"
        trace.stage_ms["search"] = 1.0
        return RetrievalResponse(
            chunks=[
                RetrievalChunk(
                    chunk_id=f"chunk-{case}",
                    doc_id="deepseek-doc",
                    text=f"{case} evidence",
                    score=1.0,
                    citations=[f"deepseek-doc:{case}"],
                    metadata={"_retrieval": {"final_rank": 1, "context": trace.as_dict()}},
                )
            ],
            trace=trace,
        )


def test_weighted_grid_reuses_one_candidate_and_persists_distinct_fusion_runs() -> None:
    repo = ensure_quality_repository(ensure_index_repository(InMemoryRepo()))
    baseline = IndexVersion(
        index_version_id="idx-base",
        alias_name="rag-active",
        physical_collection="base",
        embedding_contract_id="emb-a",
        embedding_spec={"dimension": 4},
        vector_schema={"dense": {"dimension": 4, "distance": "cosine"}},
        status="active",
    )
    candidate = IndexVersion(
        index_version_id="idx-candidate",
        alias_name="rag-active",
        physical_collection="candidate",
        embedding_contract_id="emb-a",
        embedding_spec={"dimension": 4},
        vector_schema={
            "dense": {"name": "dense", "dimension": 4, "distance": "cosine"},
            "sparse": {"name": "sparse", "contract_id": "sparse-a"},
        },
        status="validating",
    )
    repo.add_index_version(baseline)
    repo.add_index_version(candidate)
    lifecycle = SimpleNamespace(repo=repo, alias_name="rag-active")
    services = SimpleNamespace(repo=repo, retriever=_GridRetriever(), index_lifecycle=lifecycle)
    dataset = {
        "name": "weighted-grid-smoke",
        "defaults": {"top_k": 10},
        "cases": [
            {"id": "alpha", "category": "exact", "query": "alpha query", "relevance": {"doc_ids": ["deepseek-doc"]}},
            {"id": "beta", "category": "cross-lingual", "query": "beta query", "relevance": {"doc_ids": ["deepseek-doc"]}},
        ],
    }
    specs = [
        QdrantRrfFusionSpec(1, 1),
        QdrantRrfFusionSpec(3, 1),
        QdrantRrfFusionSpec(5, 1),
    ]
    result = asyncio.run(
        evaluate_weighted_rrf_grid(
            services=services,
            dataset=dataset,
            candidate_index_version_id=candidate.index_version_id,
            tenant_id="tenant-a",
            fusion_specs=specs,
            rerank=False,
            repetitions=1,
            policy=PromotionPolicy(max_p95_latency_increase_ratio=10.0),
        )
    )
    assert result["accepted_count"] == 3
    assert result["activation_performed"] is False
    fusion_ids = {
        item["evaluation"]["runtime_contracts"]["fusion_contract_id"]
        for item in result["candidates"]
    }
    assert fusion_ids == {spec.contract_id for spec in specs}
    evaluations = repo.list_evaluation_runs(limit=10)
    assert len(evaluations) == 4  # one shared baseline + three fusion candidates


def test_scope_doc_id_becomes_part_of_scoped_dataset() -> None:
    scoped = _scoped_dataset(
        {"name": "x", "defaults": {"top_k": 10}, "cases": []},
        "deepseek-doc",
    )
    assert scoped["defaults"]["filters"]["doc_ids"] == ["deepseek-doc"]
