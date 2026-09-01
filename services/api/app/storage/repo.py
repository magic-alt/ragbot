from __future__ import annotations

import threading
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterable, List, Optional

from .models import ACLPolicy, Chunk, Document, IngestionJob, Source, TableData


class InMemoryRepo:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._documents: Dict[str, Document] = {}
        self._chunks: Dict[str, Chunk] = {}
        self._policies: Dict[str, ACLPolicy] = {}
        self._jobs: Dict[str, IngestionJob] = {}
        self._sources: Dict[str, Source] = {}
        self._tables: Dict[str, TableData] = {}

    def add_document(self, doc: Document) -> None:
        with self._lock:
            self._documents[doc.doc_id] = doc

    def get_document(self, doc_id: str) -> Optional[Document]:
        with self._lock:
            return self._documents.get(doc_id)

    def list_documents(self, tenant_id: Optional[str] = None) -> List[Document]:
        with self._lock:
            if tenant_id is None:
                return list(self._documents.values())
            return [doc for doc in self._documents.values() if doc.tenant_id == tenant_id]

    def delete_documents(self, doc_ids: Iterable[str]) -> int:
        ids = set(doc_ids)
        if not ids:
            return 0
        with self._lock:
            deleted = 0
            for doc_id in ids:
                if self._documents.pop(doc_id, None) is not None:
                    deleted += 1
            stale_chunks = [cid for cid, chunk in self._chunks.items() if chunk.doc_id in ids]
            for chunk_id in stale_chunks:
                del self._chunks[chunk_id]
            return deleted

    def delete_documents_by_source(self, source_id: str) -> List[str]:
        with self._lock:
            to_delete = [
                doc_id for doc_id, doc in self._documents.items()
                if (doc.uri and doc.uri.startswith(f"source://{source_id}"))
                or doc_id.startswith(f"doc-{source_id}:")
            ]
            for doc_id in to_delete:
                del self._documents[doc_id]
            stale_chunks = [cid for cid, chunk in self._chunks.items() if chunk.doc_id in set(to_delete)]
            for chunk_id in stale_chunks:
                del self._chunks[chunk_id]
            return to_delete

    def add_chunk(self, chunk: Chunk) -> None:
        self.add_chunks([chunk])

    def add_chunks(self, chunks: Iterable[Chunk]) -> int:
        items = list(chunks)
        if not items:
            return 0
        with self._lock:
            for chunk in items:
                self._chunks[chunk.chunk_id] = chunk
        return len(items)

    def list_chunks(self, doc_id: Optional[str] = None) -> List[Chunk]:
        with self._lock:
            if doc_id is None:
                return list(self._chunks.values())
            return [chunk for chunk in self._chunks.values() if chunk.doc_id == doc_id]

    def get_chunk(self, chunk_id: str) -> Optional[Chunk]:
        with self._lock:
            return self._chunks.get(chunk_id)

    def delete_chunks(self, chunk_ids: Iterable[str]) -> int:
        ids = set(chunk_ids)
        if not ids:
            return 0
        with self._lock:
            deleted = 0
            for chunk_id in ids:
                if self._chunks.pop(chunk_id, None) is not None:
                    deleted += 1
            return deleted

    def delete_chunks_by_doc(self, doc_id: str) -> int:
        with self._lock:
            to_delete = [cid for cid, c in self._chunks.items() if c.doc_id == doc_id]
            for cid in to_delete:
                del self._chunks[cid]
            return len(to_delete)

    def iter_chunks(self) -> Iterable[Chunk]:
        with self._lock:
            return list(self._chunks.values())

    def add_policy(self, policy: ACLPolicy) -> None:
        with self._lock:
            self._policies[policy.acl_policy_id] = policy

    def get_policy_hash(self, acl_policy_id: Optional[str]) -> Optional[str]:
        if not acl_policy_id:
            return None
        with self._lock:
            policy = self._policies.get(acl_policy_id)
            return policy.policy_hash if policy else None

    def list_policies(self, tenant_id: Optional[str] = None) -> List[ACLPolicy]:
        with self._lock:
            if tenant_id is None:
                return list(self._policies.values())
            return [policy for policy in self._policies.values() if policy.tenant_id == tenant_id]

    def add_source(self, source: Source) -> None:
        with self._lock:
            self._sources[source.source_id] = source

    def get_source(self, source_id: str) -> Optional[Source]:
        with self._lock:
            return self._sources.get(source_id)

    def list_sources(self, tenant_id: Optional[str] = None) -> List[Source]:
        with self._lock:
            if tenant_id is None:
                return list(self._sources.values())
            return [s for s in self._sources.values() if s.tenant_id == tenant_id]

    def update_source(self, source_id: str, **kwargs) -> Optional[Source]:
        with self._lock:
            source = self._sources.get(source_id)
            if not source:
                return None
            for key, value in kwargs.items():
                if hasattr(source, key):
                    setattr(source, key, value)
            return source

    def delete_source(self, source_id: str) -> bool:
        with self._lock:
            if source_id in self._sources and self._sources[source_id].status != "deleted":
                self._sources[source_id].status = "deleted"
                return True
            return False

    def add_job(self, job: IngestionJob) -> None:
        with self._lock:
            self._jobs[job.job_id] = job

    def get_job(self, job_id: str) -> Optional[IngestionJob]:
        with self._lock:
            return self._jobs.get(job_id)

    def list_jobs(self, tenant_id: Optional[str] = None, source_id: Optional[str] = None) -> List[IngestionJob]:
        with self._lock:
            jobs = list(self._jobs.values())
            if tenant_id:
                jobs = [j for j in jobs if j.tenant_id == tenant_id]
            if source_id:
                jobs = [j for j in jobs if j.source_id == source_id]
            return jobs

    def update_job(self, job_id: str, **kwargs) -> Optional[IngestionJob]:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return None
            for key, value in kwargs.items():
                if hasattr(job, key):
                    setattr(job, key, value)
            return job

    def claim_next_job(self, worker_id: str, lease_seconds: int = 120, max_attempts: int = 3) -> Optional[IngestionJob]:
        now = datetime.now(timezone.utc)
        with self._lock:
            for job in sorted(self._jobs.values(), key=lambda item: item.created_at or ""):
                if job.status == "running" and _is_expired(job.lease_expires_at, now):
                    if job.attempts >= max_attempts:
                        job.status = "failed"
                        job.error = "Worker lease expired and maximum attempts were exhausted"
                        job.completed_at = now.isoformat()
                        job.lease_owner = None
                        job.lease_expires_at = None
                        continue
                    job.status = "pending"
                    job.lease_owner = None
                    job.lease_expires_at = None
                if job.status != "pending" or job.attempts >= max_attempts:
                    continue
                if job.available_at and _parse_time(job.available_at) > now:
                    continue
                job.status = "running"
                job.attempts += 1
                job.started_at = job.started_at or now.isoformat()
                job.lease_owner = worker_id
                job.heartbeat_at = now.isoformat()
                job.lease_expires_at = (now + timedelta(seconds=lease_seconds)).isoformat()
                return job
        return None

    def heartbeat_job(self, job_id: str, worker_id: str, lease_seconds: int = 120) -> bool:
        now = datetime.now(timezone.utc)
        with self._lock:
            job = self._jobs.get(job_id)
            if not job or job.status != "running" or job.lease_owner != worker_id:
                return False
            job.heartbeat_at = now.isoformat()
            job.lease_expires_at = (now + timedelta(seconds=lease_seconds)).isoformat()
            return True

    def release_job_lease(self, job_id: str, worker_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job or job.lease_owner != worker_id:
                return False
            job.lease_owner = None
            job.lease_expires_at = None
            return True

    def register_table(self, table: TableData) -> None:
        with self._lock:
            self._tables[table.name] = table

    def get_table(self, name: str) -> Optional[TableData]:
        with self._lock:
            return self._tables.get(name)

    def healthcheck(self) -> bool:
        return True

    def export_state(self) -> Dict[str, List[dict]]:
        with self._lock:
            return {
                "documents": [asdict(doc) for doc in self._documents.values()],
                "chunks": [asdict(chunk) for chunk in self._chunks.values()],
                "policies": [asdict(policy) for policy in self._policies.values()],
                "sources": [asdict(s) for s in self._sources.values()],
                "jobs": [asdict(job) for job in self._jobs.values()],
                "tables": [asdict(table) for table in self._tables.values()],
            }


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _is_expired(value: Optional[str], now: datetime) -> bool:
    return bool(value and _parse_time(value) <= now)
