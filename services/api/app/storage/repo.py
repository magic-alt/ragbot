from __future__ import annotations

import threading
from dataclasses import asdict
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

    # ── Documents ──────────────────────────────────────────────────────

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

    def delete_documents_by_source(self, source_id: str) -> List[str]:
        with self._lock:
            to_delete = [
                doc_id for doc_id, doc in self._documents.items()
                if doc.uri and doc.uri.startswith(f"source://{source_id}")
            ]
            for doc_id in to_delete:
                del self._documents[doc_id]
            return to_delete

    # ── Chunks ─────────────────────────────────────────────────────────

    def add_chunk(self, chunk: Chunk) -> None:
        with self._lock:
            self._chunks[chunk.chunk_id] = chunk

    def list_chunks(self, doc_id: Optional[str] = None) -> List[Chunk]:
        with self._lock:
            if doc_id is None:
                return list(self._chunks.values())
            return [chunk for chunk in self._chunks.values() if chunk.doc_id == doc_id]

    def get_chunk(self, chunk_id: str) -> Optional[Chunk]:
        with self._lock:
            return self._chunks.get(chunk_id)

    def delete_chunks_by_doc(self, doc_id: str) -> int:
        with self._lock:
            to_delete = [cid for cid, c in self._chunks.items() if c.doc_id == doc_id]
            for cid in to_delete:
                del self._chunks[cid]
            return len(to_delete)

    def iter_chunks(self) -> Iterable[Chunk]:
        with self._lock:
            return list(self._chunks.values())

    # ── Policies ───────────────────────────────────────────────────────

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

    # ── Sources ────────────────────────────────────────────────────────

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
            if source_id in self._sources:
                self._sources[source_id].status = "deleted"
                return True
            return False

    # ── Jobs ───────────────────────────────────────────────────────────

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

    # ── Tables ─────────────────────────────────────────────────────────

    def register_table(self, table: TableData) -> None:
        with self._lock:
            self._tables[table.name] = table

    def get_table(self, name: str) -> Optional[TableData]:
        with self._lock:
            return self._tables.get(name)

    # ── Export ─────────────────────────────────────────────────────────

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
