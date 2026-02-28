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
    source_type: str  # local_fs, pdf, web, repo, email, database
    name: str
    config: Dict[str, Any] = field(default_factory=dict)
    status: str = "active"  # active, paused, deleted
    acl_policy_id: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


@dataclass
class IngestionJob:
    job_id: str
    tenant_id: str
    source_id: str
    source_type: str
    source_config: Dict[str, Any]
    status: str = "pending"  # pending, running, completed, failed
    doc_count: int = 0
    chunk_count: int = 0
    error: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    created_at: Optional[str] = None
    stats: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TableData:
    name: str
    rows: List[Dict[str, Any]]
    columns: List[Dict[str, str]]

