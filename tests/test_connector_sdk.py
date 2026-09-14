from __future__ import annotations

import threading

import pytest

from services.api.app.storage.connector_state_support import ensure_connector_state_repository
from services.api.app.storage.models import Chunk, Source
from services.api.app.storage.repo import InMemoryRepo
from services.worker.connectors.registry import (
    ConnectorCapabilities,
    ConnectorRegistry,
    ConnectorSpec,
)
from services.worker.connectors.sdk import (
    ConnectorCheckpoint,
    ConnectorContext,
    ConnectorSync,
    PreparedConnectorRun,
    SourceChange,
    SourceRecord,
    make_sdk_runner,
)


class _ActivationRepo:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.activations: list[tuple[str, str]] = []

    def activate_knowledge_generation(self, source_id: str, generation_id: str, *args, **kwargs):
        self.activations.append((source_id, generation_id))
        return "old-generation"


class _Connector:
    def __init__(self, contexts: list[ConnectorContext], changes, checkpoint) -> None:
        self._contexts = contexts
        self._changes = changes
        self._checkpoint = checkpoint

    def validate_config(self, config):
        assert "password" not in config

    def health(self, config):
        return {"ok": True}

    def sync(self, context: ConnectorContext) -> ConnectorSync:
        self._contexts.append(context)
        return ConnectorSync(
            changes=self._changes,
            checkpoint=self._checkpoint,
            full_resync=False,
            diagnostics={"page_count": 2},
        )


def _source() -> Source:
    return Source(
        source_id="external-source",
        tenant_id="tenant-a",
        source_type="example_sdk",
        name="External SDK",
        config={},
    )


def _previous(external_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=f"old-{external_id}",
        doc_id=f"doc-{external_id}",
        tenant_id="tenant-a",
        chunk_index=0,
        text=text,
        metadata={"external_id": external_id, "remote_version": "old"},
        source_id="external-source",
    )


def _transform(record: SourceRecord, source: Source, repo) -> list[Chunk]:
    return [
        Chunk(
            chunk_id=f"new-{record.external_id}",
            doc_id=f"doc-{record.external_id}",
            tenant_id=source.tenant_id,
            chunk_index=0,
            text=str(record.content),
            metadata={},
        )
    ]


def test_checkpoint_promotes_only_after_generation_activation() -> None:
    repo = ensure_connector_state_repository(_ActivationRepo())
    repo.stage_connector_checkpoint(
        "source-1", "tenant-a", ConnectorCheckpoint(delta_token="next").as_dict()
    )
    assert repo.get_connector_checkpoint("source-1") is None

    previous = repo.activate_knowledge_generation("source-1", "gen-2")

    assert previous == "old-generation"
    assert repo.get_connector_checkpoint("source-1")["delta_token"] == "next"
    assert repo.activations == [("source-1", "gen-2")]


def test_prepared_run_stages_only_after_full_stream_consumption() -> None:
    repo = ensure_connector_state_repository(InMemoryRepo())
    checkpoint = ConnectorCheckpoint(cursor="cursor-2")
    run = PreparedConnectorRun(
        chunks=[_previous("a", "a"), _previous("b", "b")],
        source_id="source-1",
        tenant_id="tenant-a",
        repo=repo,
        checkpoint=checkpoint,
    )
    iterator = iter(run)
    assert next(iterator).chunk_id == "old-a"
    iterator.close()
    assert repo.get_connector_checkpoint("source-1") is None
    assert repo._connector_checkpoints.get("source-1") is None

    complete = PreparedConnectorRun(
        chunks=[_previous("a", "a")],
        source_id="source-1",
        tenant_id="tenant-a",
        repo=repo,
        checkpoint=checkpoint,
    )
    assert len(list(complete)) == 1
    # Only candidate/pending state exists until publication activation.
    assert repo.get_connector_checkpoint("source-1") is None
    assert repo._connector_checkpoints["source-1"]["pending_checkpoint"]["cursor"] == "cursor-2"


def test_external_connector_registers_without_core_dispatch_and_applies_deltas() -> None:
    repo = ensure_connector_state_repository(InMemoryRepo())
    source = _source()
    previous = [_previous("keep", "unchanged"), _previous("delete", "remove")]
    contexts: list[ConnectorContext] = []
    checkpoint = ConnectorCheckpoint(delta_token="delta-2")
    changes = [
        SourceChange(kind="delete", external_id="delete"),
        SourceChange(
            kind="upsert",
            external_id="update",
            record=SourceRecord(
                external_id="update",
                remote_version="v2",
                content="updated body",
                title="Updated",
                uri="example://update",
            ),
        ),
    ]
    runner = make_sdk_runner(
        lambda: _Connector(contexts, changes, checkpoint),
        _transform,
        max_concurrency=3,
    )
    registry = ConnectorRegistry()
    registry.register(
        ConnectorSpec(
            source_type="example_sdk",
            runner=runner,
            build_config=lambda location, extra: dict(extra),
            canonicalize=lambda location: location,
            match_location=lambda location: 100 if location.startswith("example://") else 0,
            source_location=lambda config: "example://root",
            capabilities=ConnectorCapabilities(
                incremental=True,
                remote=True,
                credentials=True,
                multi_document=True,
            ),
        )
    )

    result = list(registry.ingest(source, repo, previous))

    assert [chunk.metadata["external_id"] for chunk in result] == ["keep", "update"]
    updated = next(chunk for chunk in result if chunk.metadata["external_id"] == "update")
    assert updated.metadata["remote_version"] == "v2"
    assert updated.metadata["document_title"] == "Updated"
    assert contexts[0].checkpoint is None
    assert contexts[0].max_concurrency == 3
    assert repo._connector_checkpoints[source.source_id]["pending_checkpoint"]["delta_token"] == "delta-2"


def test_connector_change_stream_failure_does_not_stage_checkpoint() -> None:
    repo = ensure_connector_state_repository(InMemoryRepo())

    def broken_changes():
        yield SourceChange(
            kind="upsert",
            external_id="one",
            record=SourceRecord(external_id="one", remote_version="v1", content="one"),
        )
        raise RuntimeError("provider page failed")

    runner = make_sdk_runner(
        lambda: _Connector([], broken_changes(), ConnectorCheckpoint(cursor="unsafe")),
        _transform,
    )
    with pytest.raises(RuntimeError, match="provider page failed"):
        list(runner(_source(), repo, []))

    assert repo.get_connector_checkpoint("external-source") is None
    assert repo._connector_checkpoints.get("external-source") is None
