from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone

import pytest

from services.api.app.storage.generation_support import ensure_generation_repository
from services.api.app.storage.models import Chunk, Document, KnowledgeGeneration, Source
from services.api.app.storage.pg_repo import PostgresRepo

pytestmark = pytest.mark.skipif(
    not os.getenv("POSTGRES_TEST_DSN"),
    reason="POSTGRES_TEST_DSN not configured",
)


def _document(source: Source, generation_id: str, suffix: str) -> Document:
    now = datetime.now(timezone.utc).isoformat()
    return Document(
        doc_id=f"doc-{source.source_id}-{suffix}",
        tenant_id=source.tenant_id,
        source_type=source.source_type,
        title=f"generation {suffix}",
        uri=f"source://{source.source_id}/{suffix}",
        version="1.0",
        doc_updated_at=now,
        ingested_at=now,
        tags=["generation-smoke"],
        source_id=source.source_id,
        generation_id=generation_id,
    )


def _chunk(source: Source, document: Document, generation_id: str, suffix: str) -> Chunk:
    point_id = str(uuid.uuid4())
    return Chunk(
        chunk_id=f"chunk-{source.source_id}-{suffix}",
        doc_id=document.doc_id,
        tenant_id=source.tenant_id,
        chunk_index=0,
        text=f"staged generation {suffix} searchable knowledge",
        checksum=f"checksum-{suffix}",
        qdrant_point_id=point_id,
        metadata={
            "source_type": source.source_type,
            "source_id": source.source_id,
            "generation_id": generation_id,
            "acl_hash": "public",
            "version": "1.0",
        },
        source_id=source.source_id,
        generation_id=generation_id,
    )


def test_postgres_generation_activation_is_atomic_and_queues_cleanup() -> None:
    dsn = os.environ["POSTGRES_TEST_DSN"]
    repo = PostgresRepo(dsn, pool_min=1, pool_max=2)
    ensure_generation_repository(repo)
    try:
        suffix = uuid.uuid4().hex
        source = Source(
            source_id=f"source-generation-{suffix}",
            tenant_id=f"tenant-generation-{suffix}",
            source_type="local_fs",
            name="generation postgres smoke",
            config={"path": "/tmp"},
            created_at=datetime.now(timezone.utc).isoformat(),
            updated_at=datetime.now(timezone.utc).isoformat(),
        )
        repo.add_source(source)

        generation_one = f"gen-one-{suffix}"
        repo.begin_knowledge_generation(
            KnowledgeGeneration(
                generation_id=generation_one,
                source_id=source.source_id,
                tenant_id=source.tenant_id,
                job_id=f"job-one-{suffix}",
            )
        )
        doc_one = _document(source, generation_one, "one")
        chunk_one = _chunk(source, doc_one, generation_one, "one")
        assert repo.stage_knowledge_generation(generation_one, [doc_one], [chunk_one]) == {
            "documents": 1,
            "chunks": 1,
        }
        repo.mark_knowledge_generation_prepared(generation_one, {"smoke": 1})
        assert repo.activate_knowledge_generation(source.source_id, generation_one) is None
        assert repo.get_active_generation_id(source.source_id) == generation_one
        assert repo.active_vector_points([chunk_one.chunk_id]) == {
            chunk_one.chunk_id: chunk_one.qdrant_point_id,
        }
        stored_one = repo.get_document(doc_one.doc_id)
        assert stored_one is not None
        assert repo.list_chunks(doc_one.doc_id)[0].chunk_id == chunk_one.chunk_id

        generation_two = f"gen-two-{suffix}"
        repo.begin_knowledge_generation(
            KnowledgeGeneration(
                generation_id=generation_two,
                source_id=source.source_id,
                tenant_id=source.tenant_id,
                job_id=f"job-two-{suffix}",
            )
        )
        doc_two = _document(source, generation_two, "two")
        chunk_two = _chunk(source, doc_two, generation_two, "two")
        repo.stage_knowledge_generation(generation_two, [doc_two], [chunk_two])
        repo.mark_knowledge_generation_prepared(generation_two, {"smoke": 2})
        retired = repo.activate_knowledge_generation(
            source.source_id,
            generation_two,
            cleanup_point_ids=[chunk_one.qdrant_point_id],
            previous_doc_ids=[doc_one.doc_id],
        )
        assert retired == generation_one
        assert repo.get_active_generation_id(source.source_id) == generation_two
        assert repo.get_document(doc_one.doc_id) is None
        assert repo.get_document(doc_two.doc_id) is not None
        assert repo.active_vector_points([chunk_one.chunk_id, chunk_two.chunk_id]) == {
            chunk_two.chunk_id: chunk_two.qdrant_point_id,
        }

        with repo._pool.connection() as conn:
            rows = conn.execute(
                "SELECT generation_id, status FROM knowledge_generations WHERE generation_id = ANY(%s)",
                ([generation_one, generation_two],),
            ).fetchall()
            statuses = {str(row["generation_id"]): str(row["status"]) for row in rows}
        assert statuses == {generation_one: "retired", generation_two: "active"}

        events = repo.claim_publication_outbox("worker-smoke", lease_seconds=30, limit=10)
        matching = [event for event in events if event.source_id == source.source_id]
        assert len(matching) == 1
        event = matching[0]
        assert event.event_type == "delete_qdrant_points"
        assert event.payload["point_ids"] == [chunk_one.qdrant_point_id]
        assert repo.complete_publication_outbox(event.outbox_id, "worker-smoke")
        assert not [
            event
            for event in repo.claim_publication_outbox("worker-smoke", lease_seconds=30, limit=10)
            if event.source_id == source.source_id
        ]
    finally:
        repo.close()
