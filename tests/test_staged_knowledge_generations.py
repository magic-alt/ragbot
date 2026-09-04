from __future__ import annotations

from datetime import datetime, timezone

import pytest

from services.api.app.retrieval.embedder import HashEmbedder
from services.api.app.retrieval.qdrant import InMemoryQdrant
from services.api.app.retrieval.service import Retriever
from services.api.app.storage.generation_support import ensure_generation_repository
from services.api.app.storage.models import Chunk, Source
from services.api.app.storage.repo import InMemoryRepo
from services.worker import main as worker_main
from services.worker import pipeline


def _source() -> Source:
    now = datetime.now(timezone.utc).isoformat()
    return Source(
        source_id="source-generation-test",
        tenant_id="tenant-a",
        source_type="web",
        name="generation test",
        config={"url": "https://example.invalid", "doc_id": "doc-generation-test"},
        created_at=now,
        updated_at=now,
    )


def _candidate(text: str, checksum: str, chunk_id: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id="doc-generation-test",
        tenant_id="tenant-a",
        chunk_index=0,
        text=text,
        checksum=checksum,
        url="https://example.invalid",
        metadata={"acl_hash": "public", "version": "1.0"},
    )


def _install_connector(monkeypatch: pytest.MonkeyPatch, candidate: Chunk) -> None:
    monkeypatch.setattr(
        pipeline,
        "_run_connector",
        lambda _source, _repo, _previous=(): iter([candidate]),
    )


def test_staged_activation_hides_retired_vector_until_outbox_cleanup(monkeypatch: pytest.MonkeyPatch) -> None:
    repo = InMemoryRepo()
    ensure_generation_repository(repo)
    qdrant = InMemoryQdrant(dim=64)
    embedder = HashEmbedder(dim=64)
    source = _source()
    repo.add_source(source)

    first = _candidate("old actuator commissioning note", "checksum-v1", "candidate-v1")
    _install_connector(monkeypatch, first)
    first_job = pipeline.run_ingest_pipeline(source, repo, qdrant, embedder=embedder)
    assert first_job.status == "completed"
    assert first_job.stats["publication_mode"] == "staged-generation"
    first_active = repo.list_chunks("doc-generation-test")
    assert len(first_active) == 1
    old_chunk_id = first_active[0].chunk_id
    old_point_id = first_active[0].qdrant_point_id
    first_generation = first_job.stats["knowledge_generation_id"]
    assert repo.get_active_generation_id(source.source_id) == first_generation
    assert qdrant.count() == 1

    second = _candidate("new actuator commissioning note", "checksum-v2", "candidate-v2")
    _install_connector(monkeypatch, second)
    second_job = pipeline.run_ingest_pipeline(source, repo, qdrant, embedder=embedder)
    assert second_job.status == "completed"
    assert second_job.stats["previous_knowledge_generation_id"] == first_generation
    assert second_job.stats["vector_cleanup_enqueued"] == 1
    assert qdrant.count() == 2  # retired point still exists physically

    active = repo.list_chunks("doc-generation-test")
    assert len(active) == 1
    assert active[0].chunk_id != old_chunk_id
    assert active[0].qdrant_point_id != old_point_id
    assert repo.active_vector_points([old_chunk_id]) == {}

    # Qdrant still contains both generations, but retrieval admits only the
    # physical point referenced by the authoritative active PG manifest.
    retriever = Retriever(repo, qdrant, embedder=embedder)
    hits = retriever.retrieve(
        "actuator commissioning note",
        {"tenant_id": "tenant-a", "security_scope": ["public"]},
        top_k=10,
        mode="vector",
    )
    assert hits
    assert {hit.chunk_id for hit in hits} == {active[0].chunk_id}

    processed = worker_main._drain_publication_outbox(
        repo,
        qdrant,
        worker_id="worker-a",
        lease_seconds=30,
        max_attempts=5,
        retry_base_seconds=0.01,
        retry_max_seconds=1.0,
    )
    assert processed == 1
    assert qdrant.count() == 1


class _FailAfterWriteQdrant(InMemoryQdrant):
    fail_after_write = False

    def upsert(self, points):
        super().upsert(points)
        if self.fail_after_write:
            raise RuntimeError("simulated qdrant failure after physical write")


def test_failed_staging_does_not_replace_active_manifest(monkeypatch: pytest.MonkeyPatch) -> None:
    repo = InMemoryRepo()
    ensure_generation_repository(repo)
    qdrant = _FailAfterWriteQdrant(dim=64)
    embedder = HashEmbedder(dim=64)
    source = _source()
    repo.add_source(source)

    _install_connector(monkeypatch, _candidate("stable active knowledge", "stable", "stable-candidate"))
    first_job = pipeline.run_ingest_pipeline(source, repo, qdrant, embedder=embedder)
    assert first_job.status == "completed"
    active_before = repo.list_chunks("doc-generation-test")
    assert len(active_before) == 1
    active_id = active_before[0].chunk_id
    active_point = active_before[0].qdrant_point_id

    qdrant.fail_after_write = True
    _install_connector(monkeypatch, _candidate("candidate that must not activate", "changed", "changed-candidate"))
    failed_job = pipeline.run_ingest_pipeline(source, repo, qdrant, embedder=embedder)
    assert failed_job.status == "failed"

    active_after = repo.list_chunks("doc-generation-test")
    assert [chunk.chunk_id for chunk in active_after] == [active_id]
    assert repo.active_vector_points([active_id]) == {active_id: active_point}
    assert qdrant.count() == 2  # failed staged point is present but invisible

    qdrant.fail_after_write = False
    processed = worker_main._drain_publication_outbox(
        repo,
        qdrant,
        worker_id="worker-a",
        lease_seconds=30,
        max_attempts=5,
        retry_base_seconds=0.01,
        retry_max_seconds=1.0,
    )
    assert processed == 1
    assert qdrant.count() == 1
    assert repo.list_chunks("doc-generation-test")[0].chunk_id == active_id


def test_publication_outbox_retry_is_durable_and_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    repo = InMemoryRepo()
    ensure_generation_repository(repo)
    qdrant = InMemoryQdrant(dim=64)
    embedder = HashEmbedder(dim=64)
    source = _source()
    repo.add_source(source)

    _install_connector(monkeypatch, _candidate("one", "one", "one"))
    assert pipeline.run_ingest_pipeline(source, repo, qdrant, embedder=embedder).status == "completed"
    _install_connector(monkeypatch, _candidate("two", "two", "two"))
    assert pipeline.run_ingest_pipeline(source, repo, qdrant, embedder=embedder).status == "completed"

    original_delete = qdrant.delete_points
    failures = {"remaining": 1}

    def flaky_delete(point_ids):
        if failures["remaining"]:
            failures["remaining"] -= 1
            raise RuntimeError("transient delete failure")
        return original_delete(point_ids)

    qdrant.delete_points = flaky_delete  # type: ignore[method-assign]
    monkeypatch.setattr(worker_main, "durable_retry_delay", lambda *_args, **_kwargs: 0.0)
    processed = worker_main._drain_publication_outbox(
        repo,
        qdrant,
        worker_id="worker-a",
        lease_seconds=30,
        max_attempts=5,
        retry_base_seconds=0.01,
        retry_max_seconds=1.0,
    )
    assert processed == 0
    assert qdrant.count() == 2

    processed = worker_main._drain_publication_outbox(
        repo,
        qdrant,
        worker_id="worker-a",
        lease_seconds=30,
        max_attempts=5,
        retry_base_seconds=0.01,
        retry_max_seconds=1.0,
    )
    assert processed == 1
    assert qdrant.count() == 1

    # Completed outbox events are not reclaimed.
    assert worker_main._drain_publication_outbox(
        repo,
        qdrant,
        worker_id="worker-a",
        lease_seconds=30,
        max_attempts=5,
        retry_base_seconds=0.01,
        retry_max_seconds=1.0,
    ) == 0
