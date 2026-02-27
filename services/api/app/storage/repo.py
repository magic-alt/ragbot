from __future__ import annotations

import threading
from dataclasses import asdict
from typing import Dict, Iterable, List, Optional

from .models import ACLPolicy, Chunk, Document, IngestionJob, TableData


class InMemoryRepo:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._documents: Dict[str, Document] = {}
        self._chunks: Dict[str, Chunk] = {}
        self._policies: Dict[str, ACLPolicy] = {}
        self._jobs: Dict[str, IngestionJob] = {}
        self._tables: Dict[str, TableData] = {}

    def add_document(self, doc: Document) -> None:
        with self._lock:
            self._documents[doc.doc_id] = doc

    def add_chunk(self, chunk: Chunk) -> None:
        with self._lock:
            self._chunks[chunk.chunk_id] = chunk

    def add_policy(self, policy: ACLPolicy) -> None:
        with self._lock:
            self._policies[policy.acl_policy_id] = policy

    def add_job(self, job: IngestionJob) -> None:
        with self._lock:
            self._jobs[job.job_id] = job

    def get_document(self, doc_id: str) -> Optional[Document]:
        with self._lock:
            return self._documents.get(doc_id)

    def list_documents(self, tenant_id: Optional[str] = None) -> List[Document]:
        with self._lock:
            if tenant_id is None:
                return list(self._documents.values())
            return [doc for doc in self._documents.values() if doc.tenant_id == tenant_id]

    def list_chunks(self, doc_id: Optional[str] = None) -> List[Chunk]:
        with self._lock:
            if doc_id is None:
                return list(self._chunks.values())
            return [chunk for chunk in self._chunks.values() if chunk.doc_id == doc_id]

    def get_chunk(self, chunk_id: str) -> Optional[Chunk]:
        with self._lock:
            return self._chunks.get(chunk_id)

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

    def register_table(self, table: TableData) -> None:
        with self._lock:
            self._tables[table.name] = table

    def get_table(self, name: str) -> Optional[TableData]:
        with self._lock:
            return self._tables.get(name)

    def export_state(self) -> Dict[str, List[dict]]:
        with self._lock:
            return {
                "documents": [asdict(doc) for doc in self._documents.values()],
                "chunks": [asdict(chunk) for chunk in self._chunks.values()],
                "policies": [asdict(policy) for policy in self._policies.values()],
                "jobs": [asdict(job) for job in self._jobs.values()],
                "tables": [asdict(table) for table in self._tables.values()],
            }

    def iter_chunks(self) -> Iterable[Chunk]:
        with self._lock:
            return list(self._chunks.values())
