from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Literal, Mapping, Optional, Protocol, runtime_checkable

from services.api.app.storage.models import Chunk, Source
from services.api.app.storage.protocol import Repo


@dataclass(frozen=True)
class ConnectorCheckpoint:
    """Opaque provider resume state committed only after publication succeeds."""

    token: Optional[str] = None
    cursor: Optional[str] = None
    delta_token: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "token": self.token,
            "cursor": self.cursor,
            "delta_token": self.delta_token,
            "metadata": dict(self.metadata or {}),
        }

    @classmethod
    def from_dict(cls, value: Optional[Mapping[str, Any]]) -> Optional["ConnectorCheckpoint"]:
        if not value:
            return None
        return cls(
            token=_optional_string(value.get("token")),
            cursor=_optional_string(value.get("cursor")),
            delta_token=_optional_string(value.get("delta_token")),
            metadata=dict(value.get("metadata") or {}),
        )


@dataclass(frozen=True)
class SourceRecord:
    """Normalized provider record before Ragbot parsing/chunking."""

    external_id: str
    remote_version: str
    content: str | bytes
    title: Optional[str] = None
    uri: Optional[str] = None
    media_type: str = "text/plain"
    metadata: Mapping[str, Any] = field(default_factory=dict)
    acl: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.external_id).strip():
            raise ValueError("SourceRecord.external_id must not be empty")


@dataclass(frozen=True)
class SourceChange:
    kind: Literal["upsert", "delete"]
    external_id: str
    record: Optional[SourceRecord] = None

    def __post_init__(self) -> None:
        if not str(self.external_id).strip():
            raise ValueError("SourceChange.external_id must not be empty")
        if self.kind == "upsert" and self.record is None:
            raise ValueError("upsert SourceChange requires record")
        if self.record is not None and self.record.external_id != self.external_id:
            raise ValueError("SourceChange external_id must match record.external_id")


@dataclass(frozen=True)
class ConnectorContext:
    source: Source
    checkpoint: Optional[ConnectorCheckpoint]
    max_concurrency: int = 4


@dataclass(frozen=True)
class ConnectorSync:
    changes: Iterable[SourceChange]
    checkpoint: Optional[ConnectorCheckpoint] = None
    full_resync: bool = False
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class Connector(Protocol):
    """Third-party connector SDK contract.

    Implementations enumerate normalized changes lazily. They must not persist
    provider credentials or advance durable checkpoints themselves.
    """

    def validate_config(self, config: Mapping[str, Any]) -> None: ...

    def health(self, config: Mapping[str, Any]) -> Mapping[str, Any]: ...

    def sync(self, context: ConnectorContext) -> ConnectorSync: ...


RecordTransformer = Any


@dataclass
class PreparedConnectorRun:
    chunks: Iterable[Chunk]
    source_id: str
    tenant_id: str
    checkpoint: Optional[ConnectorCheckpoint] = None
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def commit_checkpoint(self, repo: Repo) -> bool:
        if self.checkpoint is None:
            return False
        setter = getattr(repo, "set_connector_checkpoint", None)
        if not callable(setter):
            raise RuntimeError(
                "Connector produced a checkpoint but repository does not implement connector checkpoint state"
            )
        setter(
            self.source_id,
            self.tenant_id,
            self.checkpoint.as_dict(),
        )
        return True


def apply_changes_to_previous_chunks(
    previous_chunks: Iterable[Chunk],
    changes: Iterable[SourceChange],
    *,
    source: Source,
    repo: Repo,
    transform_record: Any,
    full_resync: bool,
) -> Iterable[Chunk]:
    """Adapt SDK deltas into the pipeline's complete candidate snapshot.

    The generation publisher still consumes one complete candidate snapshot.
    Until the ingestion core itself becomes streaming/delta-native, this adapter
    preserves unchanged external records and replaces/deletes only IDs named by
    the connector. Provider enumeration remains lazy and checkpoint-safe.
    """

    grouped: dict[str, list[Chunk]] = {}
    if not full_resync:
        for chunk in previous_chunks:
            external_id = str((chunk.metadata or {}).get("external_id") or "")
            if external_id:
                grouped.setdefault(external_id, []).append(chunk)

    for change in changes:
        if change.kind == "delete":
            grouped.pop(change.external_id, None)
            continue
        assert change.record is not None
        transformed = list(transform_record(change.record, source, repo))
        for chunk in transformed:
            metadata = dict(chunk.metadata or {})
            metadata["external_id"] = change.record.external_id
            metadata["remote_version"] = change.record.remote_version
            metadata.setdefault("document_title", change.record.title)
            metadata.setdefault("document_uri", change.record.uri)
            metadata.setdefault("media_type", change.record.media_type)
            if change.record.acl:
                metadata.setdefault("source_acl", dict(change.record.acl))
            chunk.metadata = metadata
            chunk.source_id = source.source_id
        grouped[change.external_id] = transformed

    for external_id in sorted(grouped):
        for chunk in grouped[external_id]:
            yield chunk


def _optional_string(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
