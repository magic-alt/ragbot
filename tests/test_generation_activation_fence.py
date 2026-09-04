from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from services.api.app.storage.generation_support import ensure_generation_repository
from services.api.app.storage.models import Chunk, Document, KnowledgeGeneration, Source
from services.api.app.storage.repo import InMemoryRepo
from services.worker.source_fence import source_generation


def test_activation_rejects_source_mutation_without_replacing_active_manifest() -> None:
    repo = InMemoryRepo()
    ensure_generation_repository(repo)
    now = datetime.now(timezone.utc)
    source = Source(
        source_id="source-fence",
        tenant_id="tenant-fence",
        source_type="web",
        name="fence",
        config={"url": "https://example.invalid"},
        created_at=now.isoformat(),
        updated_at=now.isoformat(),
    )
    repo.add_source(source)
    expected = source_generation(source)

    generation_id = "generation-fence"
    repo.begin_knowledge_generation(
        KnowledgeGeneration(
            generation_id=generation_id,
            source_id=source.source_id,
            tenant_id=source.tenant_id,
            stats={"source_generation": expected},
        )
    )
    document = Document(
        doc_id="doc-fence",
        tenant_id=source.tenant_id,
        source_type=source.source_type,
        title="fence",
        uri="source://source-fence",
        version="1.0",
        doc_updated_at=now.isoformat(),
        ingested_at=now.isoformat(),
        source_id=source.source_id,
        generation_id=generation_id,
    )
    chunk = Chunk(
        chunk_id="chunk-fence",
        doc_id=document.doc_id,
        tenant_id=source.tenant_id,
        chunk_index=0,
        text="candidate knowledge",
        qdrant_point_id="00000000-0000-0000-0000-000000000001",
        source_id=source.source_id,
        generation_id=generation_id,
    )
    repo.stage_knowledge_generation(generation_id, [document], [chunk])
    repo.mark_knowledge_generation_prepared(generation_id, {"prepared": True})

    # Mutating the Source after prepare represents an edit/delete/pause lifecycle
    # boundary. The activation lock must observe this newer token and refuse the
    # candidate generation before changing the active manifest/pointer.
    repo.update_source(
        source.source_id,
        updated_at=(now + timedelta(seconds=1)).isoformat(),
    )

    with pytest.raises(RuntimeError, match="Source lifecycle generation changed during ingestion"):
        repo.activate_knowledge_generation(source.source_id, generation_id)

    assert repo.get_active_generation_id(source.source_id) is None
    assert repo.get_document(document.doc_id) is None
    assert repo.list_chunks(document.doc_id) == []
