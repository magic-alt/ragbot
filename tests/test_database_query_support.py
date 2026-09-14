from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from services.api.app.storage.generation_support import ensure_generation_repository
from services.api.app.storage.models import Document, IngestionJob, KnowledgeGeneration, Source
from services.api.app.storage.query_support import ensure_query_repository
from services.api.app.storage.repo import InMemoryRepo


def _iso(offset: int) -> str:
    return (datetime(2026, 9, 14, 0, 0, tzinfo=timezone.utc) + timedelta(seconds=offset)).isoformat()


def _repo() -> InMemoryRepo:
    repo = InMemoryRepo()
    ensure_generation_repository(repo)
    ensure_query_repository(repo)
    return repo


def test_keyset_source_pagination_is_tenant_scoped_and_stable() -> None:
    repo = _repo()
    for index in range(5):
        repo.add_source(
            Source(
                source_id=f"s-{index}",
                tenant_id="tenant-a" if index < 4 else "tenant-b",
                source_type="local_fs",
                name=f"Source {index}",
                created_at=_iso(index),
                updated_at=_iso(index),
            )
        )

    first = repo.page_sources(tenant_ids={"tenant-a"}, limit=2)
    assert first.total == 4
    assert [item.source_id for item in first.items] == ["s-3", "s-2"]
    assert first.next_cursor

    second = repo.page_sources(
        tenant_ids={"tenant-a"}, limit=2, cursor=first.next_cursor
    )
    assert second.total == 4
    assert [item.source_id for item in second.items] == ["s-1", "s-0"]
    assert second.next_cursor is None

    with pytest.raises(ValueError, match="does not match"):
        repo.page_jobs(cursor=first.next_cursor)


def test_document_and_generation_pages_are_bounded() -> None:
    repo = _repo()
    source = Source(
        source_id="source-a",
        tenant_id="tenant-a",
        source_type="pdf",
        name="A",
        created_at=_iso(0),
        updated_at=_iso(0),
    )
    repo.add_source(source)
    for index in range(3):
        repo.add_document(
            Document(
                doc_id=f"doc-{index}",
                tenant_id="tenant-a",
                source_type="pdf",
                title=f"D{index}",
                uri=f"source://source-a/{index}",
                version="1",
                doc_updated_at=_iso(index),
                ingested_at=_iso(index),
                source_id=source.source_id,
            )
        )
        repo.begin_knowledge_generation(
            KnowledgeGeneration(
                generation_id=f"gen-{index}",
                source_id=source.source_id,
                tenant_id=source.tenant_id,
                created_at=_iso(index),
            )
        )

    documents = repo.page_documents(
        tenant_ids={"tenant-a"}, source_id=source.source_id, limit=2
    )
    generations = repo.page_generations(
        tenant_ids={"tenant-a"}, source_id=source.source_id, limit=2
    )
    assert documents.total == 3
    assert len(documents.items) == 2
    assert documents.next_cursor
    assert generations.total == 3
    assert len(generations.items) == 2
    assert generations.next_cursor


def test_documents_for_source_uses_explicit_ownership_and_legacy_fallback() -> None:
    repo = _repo()
    repo.add_document(
        Document(
            doc_id="doc-owned",
            tenant_id="tenant-a",
            source_type="pdf",
            title="owned",
            uri="file:///owned.pdf",
            version="1",
            doc_updated_at=_iso(1),
            ingested_at=_iso(1),
            source_id="source-a",
        )
    )
    repo.add_document(
        Document(
            doc_id="doc-source-b:legacy",
            tenant_id="tenant-a",
            source_type="pdf",
            title="legacy",
            uri="source://source-b/legacy",
            version="1",
            doc_updated_at=_iso(2),
            ingested_at=_iso(2),
        )
    )
    assert [item.doc_id for item in repo.documents_for_source("source-a", "tenant-a")] == [
        "doc-owned"
    ]
    assert [
        item.doc_id
        for item in repo.documents_for_source(
            "source-b", "tenant-a", base_doc_id="doc-source-b"
        )
    ] == ["doc-source-b:legacy"]


def test_latest_active_job_and_overview_do_not_change_product_semantics() -> None:
    repo = _repo()
    source = Source(
        source_id="source-a",
        tenant_id="tenant-a",
        source_type="pdf",
        name="A",
        sync_enabled=True,
        sync_interval_seconds=300,
        sync_next_at=_iso(100),
        created_at=_iso(0),
        updated_at=_iso(0),
    )
    repo.add_source(source)
    repo.add_job(
        IngestionJob(
            job_id="completed",
            tenant_id="tenant-a",
            source_id=source.source_id,
            source_type="pdf",
            source_config={},
            status="completed",
            doc_count=2,
            chunk_count=3,
            stats={"chunks_total": 11},
            created_at=_iso(1),
            completed_at=_iso(2),
        )
    )
    repo.add_job(
        IngestionJob(
            job_id="pending",
            tenant_id="tenant-a",
            source_id=source.source_id,
            source_type="pdf",
            source_config={},
            status="pending",
            created_at=_iso(3),
        )
    )
    repo.add_job(
        IngestionJob(
            job_id="running",
            tenant_id="tenant-a",
            source_id=source.source_id,
            source_type="pdf",
            source_config={},
            status="running",
            created_at=_iso(4),
        )
    )

    assert repo.latest_active_job("tenant-a", source.source_id).job_id == "running"
    overview = repo.control_plane_overview({"tenant-a"})
    assert overview["sources"]["total"] == 1
    assert overview["sources"]["scheduled"] == 1
    assert overview["queue"]["pending"] == 1
    assert overview["queue"]["running"] == 1
    assert overview["knowledge"] == {"documents": 2, "chunks": 11}
