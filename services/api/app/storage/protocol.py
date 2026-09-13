"""Focused persistence capability protocols for Ragbot.

Callers should depend on the narrowest capability they need. ``Repo`` remains a
backward-compatible composite façade while the codebase migrates away from the
historical all-in-one repository contract.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Protocol, runtime_checkable

from .models import (
    ACLPolicy,
    Chunk,
    Document,
    IngestionJob,
    KnowledgeGeneration,
    PublicationOutboxEvent,
    Source,
    TableData,
)


@runtime_checkable
class KnowledgeCatalogRepo(Protocol):
    def add_document(self, doc: Document) -> None: ...
    def get_document(self, doc_id: str) -> Optional[Document]: ...
    def list_documents(self, tenant_id: Optional[str] = None) -> List[Document]: ...
    def delete_documents(self, doc_ids: Iterable[str]) -> int: ...
    def delete_documents_by_source(self, source_id: str) -> List[str]: ...
    def add_chunk(self, chunk: Chunk) -> None: ...
    def add_chunks(self, chunks: Iterable[Chunk]) -> int: ...
    def get_chunk(self, chunk_id: str) -> Optional[Chunk]: ...
    def list_chunks(self, doc_id: Optional[str] = None) -> List[Chunk]: ...
    def delete_chunks(self, chunk_ids: Iterable[str]) -> int: ...
    def delete_chunks_by_doc(self, doc_id: str) -> int: ...
    def iter_chunks(self) -> Iterable[Chunk]: ...


@runtime_checkable
class ACLRepo(Protocol):
    def add_policy(self, policy: ACLPolicy) -> None: ...
    def get_policy_hash(self, acl_policy_id: Optional[str] = None) -> Optional[str]: ...
    def list_policies(self, tenant_id: Optional[str] = None) -> List[ACLPolicy]: ...


@runtime_checkable
class SourceRepo(Protocol):
    def add_source(self, source: Source) -> None: ...
    def get_source(self, source_id: str) -> Optional[Source]: ...
    def list_sources(self, tenant_id: Optional[str] = None) -> List[Source]: ...
    def update_source(self, source_id: str, **kwargs: Any) -> Optional[Source]: ...
    def delete_source(self, source_id: str) -> bool: ...


@runtime_checkable
class JobQueueRepo(Protocol):
    def add_job(self, job: IngestionJob) -> None: ...
    def add_job_if_absent(self, job: IngestionJob) -> bool: ...
    def get_job(self, job_id: str) -> Optional[IngestionJob]: ...
    def list_jobs(self, tenant_id: Optional[str] = None, source_id: Optional[str] = None) -> List[IngestionJob]: ...
    def update_job(self, job_id: str, **kwargs: Any) -> Optional[IngestionJob]: ...
    def claim_next_job(self, worker_id: str, lease_seconds: int = 120, max_attempts: int = 3) -> Optional[IngestionJob]: ...
    def heartbeat_job(self, job_id: str, worker_id: str, lease_seconds: int = 120) -> bool: ...
    def release_job_lease(self, job_id: str, worker_id: str) -> bool: ...
    def reconcile_ingestion_jobs(self, max_attempts: int = 3) -> Dict[str, int]: ...


@runtime_checkable
class IngestionRepo(KnowledgeCatalogRepo, ACLRepo, SourceRepo, JobQueueRepo, Protocol):
    """Persistence surface required by the ingestion/worker data plane."""


@runtime_checkable
class ControlPlaneRepo(SourceRepo, JobQueueRepo, ACLRepo, Protocol):
    """Persistence surface required by source/job/RBAC control-plane APIs."""


@runtime_checkable
class DebugStateRepo(Protocol):
    def healthcheck(self) -> bool: ...
    def export_state(self) -> Dict[str, List[dict]]: ...


@runtime_checkable
class DevelopmentTableRepo(Protocol):
    """Test/development-only table helper; not a production storage obligation."""

    def register_table(self, table: TableData) -> None: ...
    def get_table(self, name: str) -> Optional[TableData]: ...


@runtime_checkable
class Repo(
    IngestionRepo,
    DebugStateRepo,
    DevelopmentTableRepo,
    Protocol,
):
    """Backward-compatible composite repository façade."""


@runtime_checkable
class GenerationRepo(Protocol):
    """Optional staged-generation publication capability."""

    def begin_knowledge_generation(self, generation: KnowledgeGeneration) -> None: ...
    def stage_knowledge_generation(
        self,
        generation_id: str,
        documents: Iterable[Document],
        chunks: Iterable[Chunk],
    ) -> Dict[str, int]: ...
    def mark_knowledge_generation_prepared(
        self,
        generation_id: str,
        stats: Optional[Dict[str, Any]] = None,
    ) -> None: ...
    def activate_knowledge_generation(
        self,
        source_id: str,
        generation_id: str,
        cleanup_point_ids: Iterable[str] = (),
        previous_doc_ids: Iterable[str] = (),
        expected_source_generation: Optional[str] = None,
    ) -> Optional[str]: ...
    def fail_knowledge_generation(
        self,
        generation_id: str,
        error: str,
        cleanup_point_ids: Iterable[str] = (),
    ) -> None: ...
    def get_active_generation_id(self, source_id: str) -> Optional[str]: ...
    def active_vector_points(self, chunk_ids: Iterable[str]) -> Dict[str, str]: ...


@runtime_checkable
class PublicationOutboxRepo(Protocol):
    def claim_publication_outbox(
        self,
        worker_id: str,
        lease_seconds: int = 120,
        limit: int = 10,
    ) -> List[PublicationOutboxEvent]: ...
    def complete_publication_outbox(self, outbox_id: int, worker_id: str) -> bool: ...
    def retry_publication_outbox(
        self,
        outbox_id: int,
        worker_id: str,
        error: str,
        delay_seconds: float,
        max_attempts: int = 10,
    ) -> bool: ...
    def reconcile_publication_outbox(self, max_attempts: int = 10) -> Dict[str, int]: ...
