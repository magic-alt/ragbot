from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Document:
    doc_id: str
    tenant_id: str
    source_type: str
    title: str
    uri: str
    version: str
    doc_updated_at: str
    ingested_at: str
    tags: List[str] = field(default_factory=list)
    acl_policy_id: Optional[str] = None
    status: str = "active"
    source_id: Optional[str] = None
    generation_id: Optional[str] = None


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    tenant_id: str
    chunk_index: int
    text: str
    path: Optional[str] = None
    url: Optional[str] = None
    page: Optional[int] = None
    section: Optional[str] = None
    checksum: Optional[str] = None
    qdrant_point_id: Optional[str] = None
    created_at: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    source_id: Optional[str] = None
    generation_id: Optional[str] = None


@dataclass
class ACLPolicy:
    acl_policy_id: str
    tenant_id: str
    rules: Dict[str, Any]
    policy_hash: str


@dataclass
class Source:
    source_id: str
    tenant_id: str
    source_type: str  # local_fs, pdf, web, repo, s3, gdrive, notion, confluence
    name: str
    config: Dict[str, Any] = field(default_factory=dict)
    status: str = "active"  # active, paused, deleted
    acl_policy_id: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    sync_enabled: bool = False
    sync_interval_seconds: Optional[int] = None
    sync_next_at: Optional[str] = None
    sync_last_enqueued_at: Optional[str] = None


@dataclass
class IngestionJob:
    job_id: str
    tenant_id: str
    source_id: str
    source_type: str
    source_config: Dict[str, Any]
    status: str = "pending"  # pending, running, completed, failed, dead_lettered
    doc_count: int = 0
    chunk_count: int = 0
    error: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    created_at: Optional[str] = None
    stats: Dict[str, Any] = field(default_factory=dict)
    attempts: int = 0
    available_at: Optional[str] = None
    lease_owner: Optional[str] = None
    lease_expires_at: Optional[str] = None
    heartbeat_at: Optional[str] = None
    failure_class: Optional[str] = None
    dead_lettered_at: Optional[str] = None


@dataclass
class KnowledgeGeneration:
    generation_id: str
    source_id: str
    tenant_id: str
    job_id: Optional[str] = None
    status: str = "staging"
    created_at: Optional[str] = None
    prepared_at: Optional[str] = None
    activated_at: Optional[str] = None
    retired_at: Optional[str] = None
    failed_at: Optional[str] = None
    error: Optional[str] = None
    stats: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PublicationOutboxEvent:
    outbox_id: int
    event_type: str
    source_id: str
    generation_id: Optional[str] = None
    payload: Dict[str, Any] = field(default_factory=dict)
    status: str = "pending"
    attempts: int = 0
    available_at: Optional[str] = None
    lease_owner: Optional[str] = None
    lease_expires_at: Optional[str] = None
    last_error: Optional[str] = None
    created_at: Optional[str] = None
    completed_at: Optional[str] = None


@dataclass
class TableData:
    name: str
    rows: List[Dict[str, Any]]
    columns: List[Dict[str, str]]
