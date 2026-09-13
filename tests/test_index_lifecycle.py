from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from services.api.app.retrieval.embedder import HashEmbedder
from services.api.app.retrieval.embedding_router import ActiveIndexEmbedder, EmbeddingRouter
from services.api.app.retrieval.index_lifecycle import IndexLifecycleService
from services.api.app.storage.index_support import ensure_index_repository
from services.api.app.storage.models import Chunk
from services.api.app.storage.repo import InMemoryRepo
from services.worker.jobs.embed_and_upsert import _build_payload
from services.api.app.retrieval.qdrant import point_id_for_chunk


class FakeVersionedVectorStore:
    alias_name = "rag_chunks_active"

    def __init__(self, legacy: str = "rag_chunks", dim: int = 4) -> None:
        self.alias_target = legacy
        self.dimensions = {legacy: dim}
        self.collections = {legacy: {}}

    @property
    def dim(self) -> int:
        return self.collection_dimension(self.alias_target)

    def active_collection_name(self) -> str:
        return self.alias_target

    def collection_dimension(self, collection_name: str) -> int:
        return self.dimensions[collection_name]

    def create_physical_collection(self, collection_name: str, *, dim: int, distance: str = "cosine") -> None:
        self.dimensions.setdefault(collection_name, dim)
        self.collections.setdefault(collection_name, {})
        assert self.dimensions[collection_name] == dim

    def upsert_to_collection(self, collection_name: str, points) -> None:
        expected = self.dimensions[collection_name]
        for point_id, vector, payload in points:
            assert len(vector) == expected
            self.collections[collection_name][str(point_id)] = (list(vector), dict(payload))

    def search_collection(self, collection_name: str, query_vector, filters, top_k: int):
        assert len(query_vector) == self.dimensions[collection_name]
        rows = []
        for point_id, (vector, payload) in self.collections[collection_name].items():
            if filters.get("tenant_id") and payload.get("tenant_id") != filters["tenant_id"]:
                continue
            score = sum(a * b for a, b in zip(query_vector, vector))
            rows.append((point_id, score, dict(payload)))
        rows.sort(key=lambda item: item[1], reverse=True)
        return rows[:top_k]

    def switch_alias(self, collection_name: str):
        previous = self.alias_target
        self.alias_target = collection_name
        return previous

    def delete_collection(self, collection_name: str) -> bool:
        if collection_name == self.alias_target:
            raise ValueError("cannot delete active")
        existed = collection_name in self.collections
        self.collections.pop(collection_name, None)
        self.dimensions.pop(collection_name, None)
        return existed

    def count_collection(self, collection_name: str) -> int:
        return len(self.collections[collection_name])


def _chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id="doc-1",
        tenant_id="tenant-a",
        chunk_index=int(chunk_id[-1]),
        text=text,
        checksum=f"checksum-{chunk_id}",
        metadata={
            "source_type": "pdf",
            "acl_hash": "public",
            "tags": ["test"],
            "parser_config_hash": "parser-a",
            "parser_provider": "ragbot",
            "parser_strategy": "text",
            "chunker_config_hash": "chunker-a",
            "chunker_provider": "ragbot",
            "chunker_strategy": "fixed",
        },
    )


def _seed_legacy(store, embedder, chunks):
    points = []
    for chunk in chunks:
        vector = embedder.embed(chunk.text)
        point_id = point_id_for_chunk(chunk.chunk_id)
        chunk.qdrant_point_id = point_id
        chunk.metadata["embedding_contract_id"] = embedder.contract_id
        chunk.metadata["embedding_model"] = embedder.model_name
        chunk.metadata["embedding_dimension"] = embedder.dimension
        payload = _build_payload(chunk, embedder.model_name)
        payload["embedding_contract_id"] = embedder.contract_id
        points.append((point_id, vector, payload))
    store.upsert_to_collection(store.active_collection_name(), points)


def _fixture():
    repo = ensure_index_repository(InMemoryRepo())
    old = HashEmbedder(dim=4)
    new = HashEmbedder(dim=6)
    router = EmbeddingRouter([old, new])
    store = FakeVersionedVectorStore(dim=4)
    chunks = [_chunk("chunk-1", "alpha motor control"), _chunk("chunk-2", "beta ethercat servo")]
    repo.add_chunks(chunks)
    _seed_legacy(store, old, chunks)
    service = IndexLifecycleService(repo, store, router)
    baseline = service.bootstrap_current(old)
    return repo, old, new, router, store, service, baseline, chunks


def test_index_build_activate_rollback_and_alias_bound_embedding():
    repo, old, new, router, store, service, baseline, chunks = _fixture()
    candidate = service.create_candidate(new.contract_id)
    candidate = service.build(candidate.index_version_id, batch_size=1)
    assert candidate.status == "validating"
    assert candidate.build_stats["vectors_written"] == 2
    assert candidate.parser_contracts[0]["config_hash"] == "parser-a"
    for chunk in chunks:
        assert chunk.qdrant_point_id in store.collections[candidate.physical_collection]

    evidence = service.shadow_compare(
        candidate.index_version_id,
        [{"query": "ethercat", "expected_chunk_ids": ["chunk-2"]}],
        top_k=2,
    )
    assert evidence["baseline_index_version_id"] == baseline.index_version_id
    assert evidence["candidate_index_version_id"] == candidate.index_version_id
    assert evidence["labeled_cases"] == 1

    service.mark_ready(candidate.index_version_id, evidence, approved=True)
    active_embedder = ActiveIndexEmbedder(repo, router, store.alias_name, old, vector_store=store)
    assert active_embedder.contract_id == old.contract_id
    service.activate(candidate.index_version_id, retention_seconds=3600)
    assert store.active_collection_name() == candidate.physical_collection
    assert active_embedder.contract_id == new.contract_id
    assert active_embedder.dimension == 6
    assert repo.get_active_index_version(store.alias_name).index_version_id == candidate.index_version_id
    assert repo.get_index_version(baseline.index_version_id).status == "retired"
    for chunk in chunks:
        assert chunk.metadata["embedding_contract_id"] == new.contract_id
        assert chunk.metadata["embedding_model"] == new.model_name
        assert chunk.metadata["embedding_dimension"] == new.dimension

    service.rollback(baseline.index_version_id, retention_seconds=3600)
    assert store.active_collection_name() == baseline.physical_collection
    assert active_embedder.contract_id == old.contract_id
    assert repo.get_active_index_version(store.alias_name).index_version_id == baseline.index_version_id
    for chunk in chunks:
        assert chunk.metadata["embedding_contract_id"] == old.contract_id
        assert chunk.metadata["embedding_model"] == old.model_name
        assert chunk.metadata["embedding_dimension"] == old.dimension


def test_alias_is_query_authority_and_reconcile_repairs_postgres_pointer():
    repo, old, new, router, store, service, baseline, chunks = _fixture()
    candidate = service.create_candidate(new.contract_id)
    service.build(candidate.index_version_id)
    service.mark_ready(candidate.index_version_id, {"manual": True}, approved=True)

    store.switch_alias(candidate.physical_collection)
    assert repo.get_active_index_version(store.alias_name).index_version_id == baseline.index_version_id

    active_embedder = ActiveIndexEmbedder(repo, router, store.alias_name, old, vector_store=store)
    assert active_embedder.contract_id == new.contract_id
    result = service.reconcile()
    assert result["action"] == "postgres_pointer_repaired"
    assert repo.get_active_index_version(store.alias_name).index_version_id == candidate.index_version_id
    assert {chunk.metadata["embedding_contract_id"] for chunk in chunks} == {new.contract_id}


def test_failed_postgres_activation_compensates_alias_switch(monkeypatch):
    repo, old, new, router, store, service, baseline, _chunks = _fixture()
    candidate = service.create_candidate(new.contract_id)
    service.build(candidate.index_version_id)
    service.mark_ready(candidate.index_version_id, {"manual": True}, approved=True)

    def explode(*args, **kwargs):
        raise RuntimeError("database commit failed")

    monkeypatch.setattr(repo, "activate_index_version", explode)
    with pytest.raises(RuntimeError, match="database commit failed"):
        service.activate(candidate.index_version_id)
    assert store.active_collection_name() == baseline.physical_collection


def test_retention_prune_never_deletes_active_collection():
    repo, old, new, router, store, service, baseline, _chunks = _fixture()
    candidate = service.create_candidate(new.contract_id)
    service.build(candidate.index_version_id)
    service.mark_ready(candidate.index_version_id, {"manual": True}, approved=True)
    service.activate(candidate.index_version_id, retention_seconds=0)

    old_row = repo.get_index_version(baseline.index_version_id)
    assert old_row.status == "retired"
    repo.update_index_version(
        baseline.index_version_id,
        delete_after=(datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat(),
    )
    result = service.prune_retired()
    assert baseline.index_version_id in result["deleted"]
    assert baseline.physical_collection not in store.collections
    assert candidate.physical_collection in store.collections


def test_mark_ready_requires_explicit_evidence_and_approval():
    _repo, _old, new, _router, _store, service, _baseline, _chunks = _fixture()
    candidate = service.create_candidate(new.contract_id)
    service.build(candidate.index_version_id)
    with pytest.raises(ValueError, match="evidence"):
        service.mark_ready(candidate.index_version_id, {}, approved=True)
    with pytest.raises(ValueError, match="approval"):
        service.mark_ready(candidate.index_version_id, {"ok": True}, approved=False)


def test_activation_and_rollback_reject_stale_catalog_snapshot():
    repo, _old, new, _router, _store, service, baseline, chunks = _fixture()
    candidate = service.create_candidate(new.contract_id)
    service.build(candidate.index_version_id)
    service.mark_ready(candidate.index_version_id, {"manual": True}, approved=True)

    chunks[0].checksum = "changed-after-build"
    with pytest.raises(RuntimeError, match="Knowledge catalog changed"):
        service.activate(candidate.index_version_id)

    chunks[0].checksum = "checksum-chunk-1"
    service.activate(candidate.index_version_id)
    chunks[1].checksum = "changed-after-activation"
    with pytest.raises(RuntimeError, match="Knowledge catalog changed"):
        service.rollback(baseline.index_version_id)
