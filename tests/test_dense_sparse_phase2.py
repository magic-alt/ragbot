from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from benchmarks.retrieval_plan_promotion import evaluate_dense_sparse_candidate
from contracts.types import RetrievalChunk
from services.api.app.quality.contracts import PromotionPolicy, retrieval_contract_id
from services.api.app.retrieval.contracts import RetrievalPlan, RetrievalResponse, RetrievalTrace
from services.api.app.retrieval.embedder import HashEmbedder
from services.api.app.retrieval.embedding_router import EmbeddingRouter
from services.api.app.retrieval.service import Retriever
from services.api.app.retrieval.sparse import DeterministicSparseEncoder, SparseEmbeddingSpec
from services.api.app.retrieval.sparse_index_lifecycle import SparseIndexLifecycleService
from services.api.app.storage.index_support import ensure_index_repository
from services.api.app.storage.models import Chunk, IndexVersion
from services.api.app.storage.quality_support import ensure_quality_repository
from services.api.app.storage.repo import InMemoryRepo


class FakeHybridStore:
    alias_name = "rag_chunks_active"

    def __init__(self, dim: int = 4) -> None:
        self.alias_target = "rag_chunks"
        self.dimensions = {"rag_chunks": dim}
        self.schemas = {"rag_chunks": {"dense": {"": {"dimension": dim}}, "sparse": {}}}
        self.collections = {"rag_chunks": {}}
        self.native_calls = []

    @property
    def dim(self) -> int:
        return self.collection_dimension(self.alias_target)

    def active_collection_name(self) -> str:
        return self.alias_target

    def collection_dimension(self, collection_name: str) -> int:
        return self.dimensions[collection_name]

    def create_physical_collection(self, collection_name: str, *, dim: int, distance: str = "cosine") -> None:
        self.dimensions[collection_name] = dim
        self.schemas[collection_name] = {"dense": {"": {"dimension": dim}}, "sparse": {}}
        self.collections[collection_name] = {}

    def create_hybrid_collection(
        self,
        collection_name: str,
        *,
        dim: int,
        distance: str,
        dense_name: str = "dense",
        sparse_name: str = "sparse",
        sparse_modifier: str = "idf",
    ) -> None:
        self.dimensions[collection_name] = dim
        self.schemas[collection_name] = {
            "dense": {dense_name: {"dimension": dim, "distance": distance}},
            "sparse": {sparse_name: {"modifier": sparse_modifier}},
        }
        self.collections[collection_name] = {}

    def collection_vector_schema(self, collection_name: str):
        return self.schemas[collection_name]

    def upsert_to_collection(self, collection_name: str, points) -> None:
        for point_id, dense, payload in points:
            self.collections[collection_name][str(point_id)] = {
                "dense": list(dense),
                "payload": dict(payload),
            }

    def upsert_hybrid_to_collection(
        self,
        collection_name: str,
        points,
        *,
        dense_name: str = "dense",
        sparse_name: str = "sparse",
    ) -> None:
        for point_id, dense, sparse, payload in points:
            self.collections[collection_name][str(point_id)] = {
                "dense": list(dense),
                "sparse": sparse,
                "payload": dict(payload),
            }

    def search_collection(self, collection_name: str, query_vector, filters, top_k: int):
        rows = []
        for point_id, value in self.collections[collection_name].items():
            payload = value["payload"]
            if filters.get("tenant_id") and payload.get("tenant_id") != filters["tenant_id"]:
                continue
            score = sum(a * b for a, b in zip(query_vector, value["dense"]))
            rows.append((point_id, score, dict(payload)))
        rows.sort(key=lambda item: item[1], reverse=True)
        return rows[:top_k]

    def native_hybrid_search(
        self,
        dense_vector,
        sparse_vector,
        filters,
        top_k,
        *,
        collection_name=None,
        dense_name="dense",
        sparse_name="sparse",
        prefetch_limit=None,
    ):
        target = collection_name or self.alias_target
        self.native_calls.append(
            {
                "collection": target,
                "dense_name": dense_name,
                "sparse_name": sparse_name,
                "prefetch_limit": prefetch_limit,
            }
        )
        rows = []
        for point_id, value in self.collections[target].items():
            payload = value["payload"]
            if filters.get("tenant_id") and payload.get("tenant_id") != filters["tenant_id"]:
                continue
            # The fake score is deterministic; real Qdrant RRF is covered by
            # the dedicated integration workflow.
            dense_score = sum(a * b for a, b in zip(dense_vector, value["dense"]))
            sparse_score = float(len(set(sparse_vector.indices).intersection(value["sparse"].indices)))
            rows.append((point_id, dense_score + sparse_score, dict(payload)))
        rows.sort(key=lambda item: item[1], reverse=True)
        return rows[:top_k]

    def switch_alias(self, collection_name: str):
        previous = self.alias_target
        self.alias_target = collection_name
        return previous

    def delete_collection(self, collection_name: str) -> bool:
        existed = collection_name in self.collections
        self.collections.pop(collection_name, None)
        self.dimensions.pop(collection_name, None)
        self.schemas.pop(collection_name, None)
        return existed

    def count_collection(self, collection_name: str) -> int:
        return len(self.collections[collection_name])


def _chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=f"doc-{chunk_id}",
        tenant_id="tenant-a",
        chunk_index=0,
        text=text,
        checksum=f"sum-{chunk_id}",
        metadata={"source_type": "pdf", "acl_hash": "public"},
    )


def _fixture():
    repo = ensure_index_repository(InMemoryRepo())
    dense = HashEmbedder(dim=4)
    router = EmbeddingRouter([dense])
    sparse = DeterministicSparseEncoder(vector_name="sparse")
    store = FakeHybridStore(dim=4)
    chunks = [_chunk("a", "alpha servo motor"), _chunk("b", "beta ethercat fieldbus")]
    repo.add_chunks(chunks)
    service = SparseIndexLifecycleService(
        repo,
        store,
        router,
        alias_name=store.alias_name,
        sparse_encoder=sparse,
    )
    baseline = service.bootstrap_current(dense)
    return repo, dense, sparse, store, service, baseline, chunks


def test_sparse_contract_is_content_derived_and_validates_modifier() -> None:
    first = SparseEmbeddingSpec(provider_id="fastembed", model="Qdrant/bm25", modifier="idf")
    second = SparseEmbeddingSpec(provider_id="fastembed", model="Qdrant/bm25", modifier="idf")
    changed = SparseEmbeddingSpec(provider_id="fastembed", model="Qdrant/bm25", modifier="none")
    assert first.contract_id == second.contract_id
    assert first.contract_id != changed.contract_id
    with pytest.raises(ValueError, match="modifier"):
        SparseEmbeddingSpec(provider_id="fastembed", model="x", modifier="magic")


def test_retrieval_contract_changes_when_sparse_representation_changes() -> None:
    baseline = retrieval_contract_id(
        plan="qdrant_dense_sparse",
        top_k=10,
        candidate_pool=40,
        rerank=False,
        diversity=False,
        representation_contracts={"dense": "emb-a", "sparse": "sparse-a"},
        index_version_id="idx-candidate",
    )
    changed = retrieval_contract_id(
        plan="qdrant_dense_sparse",
        top_k=10,
        candidate_pool=40,
        rerank=False,
        diversity=False,
        representation_contracts={"dense": "emb-a", "sparse": "sparse-b"},
        index_version_id="idx-candidate",
    )
    assert baseline != changed


def test_sparse_candidate_preserves_dense_contract_and_builds_named_vectors() -> None:
    repo, dense, sparse, store, service, baseline, chunks = _fixture()
    candidate = service.create_candidate(
        dense.contract_id,
        sparse_contract_id=sparse.contract_id,
    )
    assert candidate.vector_schema["dense"]["name"] == "dense"
    assert candidate.vector_schema["sparse"]["contract_id"] == sparse.contract_id
    assert candidate.build_stats["baseline_index_version_id"] == baseline.index_version_id

    built = service.build(candidate.index_version_id, batch_size=1)
    assert built.status == "validating"
    values = store.collections[candidate.physical_collection]
    assert len(values) == len(chunks)
    assert all("sparse" in value for value in values.values())
    assert {value["payload"]["sparse_contract_id"] for value in values.values()} == {
        sparse.contract_id
    }


def test_sparse_candidate_rejects_dense_contract_change() -> None:
    _repo, _dense, sparse, _store, service, _baseline, _chunks = _fixture()
    with pytest.raises(ValueError, match="preserve the active dense embedding contract"):
        service.create_candidate(
            "emb-different",
            sparse_contract_id=sparse.contract_id,
        )


def test_candidate_query_targets_unactivated_index_and_emits_representation_lineage() -> None:
    repo, dense, sparse, store, service, baseline, _chunks = _fixture()
    candidate = service.create_candidate(
        dense.contract_id,
        sparse_contract_id=sparse.contract_id,
    )
    service.build(candidate.index_version_id)
    retriever = Retriever(
        repo,
        store,
        embedder=dense,
        reranker=None,
        sparse_encoder=sparse,
    )
    try:
        chunks = asyncio.run(
            retriever.aretrieve(
                "ethercat fieldbus",
                {"tenant_id": "tenant-a"},
                top_k=2,
                plan="qdrant_dense_sparse",
                rerank=False,
                index_version_id=candidate.index_version_id,
            )
        )
        assert chunks
        context = chunks[0].metadata["_retrieval"]["context"]
        assert context["index_version_id"] == candidate.index_version_id
        assert context["representation_contracts"] == {
            "dense": dense.contract_id,
            "sparse": sparse.contract_id,
        }
        assert context["fusion_method"] == "qdrant-native"
        assert store.native_calls[-1]["collection"] == candidate.physical_collection
        assert store.active_collection_name() == baseline.physical_collection
    finally:
        retriever.close()


class _PlanFixtureRetriever:
    def __init__(self, dense_contract: str, sparse_contract: str, candidate_id: str) -> None:
        self.dense_contract = dense_contract
        self.sparse_contract = sparse_contract
        self.candidate_id = candidate_id

    async def query(self, request):
        candidate = request.plan is RetrievalPlan.QDRANT_DENSE_SPARSE
        query = request.query
        relevant = "alpha" if "alpha" in query else "beta"
        if candidate:
            ordered = [relevant, "noise"]
        else:
            ordered = ["noise", relevant]
        chunks = []
        for rank, token in enumerate(ordered, 1):
            chunks.append(
                RetrievalChunk(
                    chunk_id=f"chunk-{token}",
                    doc_id=f"doc-{token}",
                    text=f"{token} evidence",
                    score=1.0 / rank,
                    citations=[f"doc-{token}:0"],
                    metadata={
                        "path": f"{token}.txt",
                        "_retrieval": {"final_rank": rank},
                    },
                )
            )
        trace = RetrievalTrace(
            plan=request.plan.value,
            deadline_ms=request.deadline_ms,
            candidate_pool=10,
            started_monotonic=0.0,
            fusion_method="qdrant-native" if candidate else "adaptive-rrf",
            representation_contracts=(
                {"dense": self.dense_contract, "sparse": self.sparse_contract}
                if candidate
                else {"dense": self.dense_contract}
            ),
            index_version_id=self.candidate_id if candidate else None,
        )
        trace.stage_ms["search"] = 1.0 if candidate else 2.0
        return RetrievalResponse(chunks=chunks, trace=trace)


def test_golden_comparison_persists_evaluation_runs_and_promotion_decision() -> None:
    repo = ensure_quality_repository(ensure_index_repository(InMemoryRepo()))
    dense_contract = "emb-control"
    sparse_contract = "sparse-candidate"
    baseline = IndexVersion(
        index_version_id="idx-base",
        alias_name="rag-active",
        physical_collection="base",
        embedding_contract_id=dense_contract,
        embedding_spec={"dimension": 4},
        vector_schema={"dense": {"dimension": 4, "distance": "cosine"}},
        status="active",
    )
    candidate = IndexVersion(
        index_version_id="idx-candidate",
        alias_name="rag-active",
        physical_collection="candidate",
        embedding_contract_id=dense_contract,
        embedding_spec={"dimension": 4},
        vector_schema={
            "dense": {"name": "dense", "dimension": 4, "distance": "cosine"},
            "sparse": {"name": "sparse", "contract_id": sparse_contract},
        },
        status="validating",
    )
    repo.add_index_version(baseline)
    repo.add_index_version(candidate)
    lifecycle = SimpleNamespace(
        repo=repo,
        alias_name="rag-active",
        mark_ready=lambda *args, **kwargs: None,
    )
    services = SimpleNamespace(
        repo=repo,
        retriever=_PlanFixtureRetriever(dense_contract, sparse_contract, candidate.index_version_id),
        index_lifecycle=lifecycle,
    )
    dataset = {
        "schema_version": 1,
        "name": "promotion-smoke",
        "defaults": {"top_k": 10},
        "cases": [
            {
                "id": "alpha",
                "category": "exact",
                "query": "alpha query",
                "relevance": {"doc_ids": ["doc-alpha"], "max_rank": 5},
            },
            {
                "id": "beta",
                "category": "paraphrase",
                "query": "beta query",
                "relevance": {"doc_ids": ["doc-beta"], "max_rank": 5},
            },
        ],
    }
    result = asyncio.run(
        evaluate_dense_sparse_candidate(
            services=services,
            dataset=dataset,
            candidate_index_version_id=candidate.index_version_id,
            tenant_id="tenant-a",
            rerank=False,
            repetitions=1,
            code_revision="test-sha",
            policy=PromotionPolicy(
                max_recall_drop=0.0,
                max_mrr_drop=0.0,
                max_ndcg_drop=0.0,
                max_p95_latency_increase_ratio=10.0,
                max_cost_increase_ratio=0.0,
            ),
        )
    )
    assert result["promotion"]["decision"] == "accept"
    assert result["candidate"]["runtime_contracts"]["sparse_contract_id"] == sparse_contract
    assert result["candidate"]["metrics"]["mrr"] > result["baseline"]["metrics"]["mrr"]
    assert repo.get_evaluation_run(result["candidate"]["evaluation_run_id"]) is not None
    decisions = repo.list_promotion_decisions(limit=10)
    assert decisions and decisions[0].candidate_evaluation_id == result["candidate"]["evaluation_run_id"]
