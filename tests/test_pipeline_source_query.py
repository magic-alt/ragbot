from __future__ import annotations

from services.api.app.storage.models import Document, Source
from services.worker.pipeline import source_documents


class BoundedSourceRepo:
    def __init__(self) -> None:
        self.calls = []
        self.document = Document(
            doc_id="doc-source-a:item",
            tenant_id="tenant-a",
            source_type="local_fs",
            title="owned",
            uri="source://source-a/item",
            version="1",
            doc_updated_at="2026-09-14T00:00:00+00:00",
            ingested_at="2026-09-14T00:00:00+00:00",
            source_id="source-a",
        )

    def documents_for_source(self, source_id: str, tenant_id: str, *, base_doc_id: str):
        self.calls.append((source_id, tenant_id, base_doc_id))
        return [self.document]

    def list_documents(self, *args, **kwargs):  # pragma: no cover - regression sentinel
        raise AssertionError("pipeline must not tenant-scan when documents_for_source is available")


def test_source_documents_prefers_bounded_repository_query() -> None:
    source = Source(
        source_id="source-a",
        tenant_id="tenant-a",
        source_type="local_fs",
        name="A",
        config={"path": "/tmp/a"},
    )
    repo = BoundedSourceRepo()

    result = source_documents(source, repo)  # type: ignore[arg-type]

    assert result == [repo.document]
    assert repo.calls == [("source-a", "tenant-a", "doc-source-a")]
