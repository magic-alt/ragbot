from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional

from ..retrieval.lexical import contains_cjk, lexicalize
from .models import Chunk, Document, KnowledgeGeneration, PublicationOutboxEvent

_OUTBOX_BATCH_SIZE = 500


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    isoformat = getattr(value, "isoformat", None)
    return isoformat() if callable(isoformat) else str(value)


def _batches(values: Iterable[str], size: int = _OUTBOX_BATCH_SIZE) -> Iterable[list[str]]:
    batch: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = str(value)
        if not item or item in seen:
            continue
        seen.add(item)
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


class PostgresGenerationMixin:
    """Atomic staged-generation publication built on a psycopg connection pool."""

    def begin_knowledge_generation(self, generation: KnowledgeGeneration) -> None:
        params = asdict(generation)
        params["stats"] = self._jsonb(params["stats"])
        with self._pool.connection() as conn:
            conn.execute(
                """
                INSERT INTO knowledge_generations (
                    generation_id, source_id, tenant_id, job_id, status,
                    created_at, prepared_at, activated_at, retired_at,
                    failed_at, error, stats
                ) VALUES (
                    %(generation_id)s, %(source_id)s, %(tenant_id)s, %(job_id)s,
                    %(status)s, COALESCE(%(created_at)s, NOW()), %(prepared_at)s,
                    %(activated_at)s, %(retired_at)s, %(failed_at)s,
                    %(error)s, %(stats)s
                )
                ON CONFLICT (generation_id) DO NOTHING
                """,
                params,
            )

    def stage_knowledge_generation(
        self,
        generation_id: str,
        documents: Iterable[Document],
        chunks: Iterable[Chunk],
    ) -> Dict[str, int]:
        docs = list(documents)
        items = list(chunks)
        with self._pool.connection() as conn:
            with conn.transaction():
                row = conn.execute(
                    "SELECT status FROM knowledge_generations WHERE generation_id = %s FOR UPDATE",
                    (generation_id,),
                ).fetchone()
                if not row:
                    raise ValueError(f"Unknown knowledge generation: {generation_id}")
                status = row["status"] if isinstance(row, dict) else row[0]
                if status not in {"staging", "prepared"}:
                    raise ValueError(
                        f"Generation {generation_id} cannot be staged from status={status}"
                    )
                conn.execute("DELETE FROM staged_chunks WHERE generation_id = %s", (generation_id,))
                conn.execute("DELETE FROM staged_documents WHERE generation_id = %s", (generation_id,))

                if docs:
                    params = []
                    for doc in docs:
                        data = asdict(doc)
                        data["generation_id"] = generation_id
                        data["source_id"] = data.get("source_id") or self._generation_source_id(conn, generation_id)
                        data["tags"] = list(data.get("tags") or [])
                        params.append(data)
                    with conn.cursor() as cur:
                        cur.executemany(
                            """
                            INSERT INTO staged_documents (
                                generation_id, source_id, doc_id, tenant_id, source_type,
                                title, uri, version, doc_updated_at, ingested_at,
                                tags, acl_policy_id, status
                            ) VALUES (
                                %(generation_id)s, %(source_id)s, %(doc_id)s, %(tenant_id)s,
                                %(source_type)s, %(title)s, %(uri)s, %(version)s,
                                %(doc_updated_at)s, %(ingested_at)s, %(tags)s,
                                %(acl_policy_id)s, %(status)s
                            )
                            """,
                            params,
                        )

                if items:
                    source_id = self._generation_source_id(conn, generation_id)
                    params = []
                    for chunk in items:
                        data = asdict(chunk)
                        data["generation_id"] = generation_id
                        data["source_id"] = data.get("source_id") or source_id
                        data["metadata"] = self._jsonb(data.get("metadata") or {})
                        data["fts_text"] = lexicalize(chunk.text) if contains_cjk(chunk.text) else chunk.text
                        params.append(data)
                    with conn.cursor() as cur:
                        cur.executemany(
                            """
                            INSERT INTO staged_chunks (
                                generation_id, source_id, chunk_id, doc_id, tenant_id,
                                chunk_index, text, path, url, page, section, checksum,
                                qdrant_point_id, created_at, metadata, fts_text
                            ) VALUES (
                                %(generation_id)s, %(source_id)s, %(chunk_id)s, %(doc_id)s,
                                %(tenant_id)s, %(chunk_index)s, %(text)s, %(path)s,
                                %(url)s, %(page)s, %(section)s, %(checksum)s,
                                %(qdrant_point_id)s, COALESCE(%(created_at)s, NOW()),
                                %(metadata)s, %(fts_text)s
                            )
                            """,
                            params,
                        )
        return {"documents": len(docs), "chunks": len(items)}

    def _generation_source_id(self, conn: Any, generation_id: str) -> str:
        row = conn.execute(
            "SELECT source_id FROM knowledge_generations WHERE generation_id = %s",
            (generation_id,),
        ).fetchone()
        if not row:
            raise ValueError(f"Unknown knowledge generation: {generation_id}")
        return str(row["source_id"] if isinstance(row, dict) else row[0])

    def mark_knowledge_generation_prepared(
        self,
        generation_id: str,
        stats: Optional[Dict[str, Any]] = None,
    ) -> None:
        with self._pool.connection() as conn:
            result = conn.execute(
                """
                UPDATE knowledge_generations
                SET status = 'prepared', prepared_at = NOW(), stats = %(stats)s
                WHERE generation_id = %(generation_id)s AND status IN ('staging', 'prepared')
                """,
                {
                    "generation_id": generation_id,
                    "stats": self._jsonb(stats or {}),
                },
            )
            if not (result.rowcount or 0):
                raise ValueError(f"Generation is not stageable/preparable: {generation_id}")

    def activate_knowledge_generation(
        self,
        source_id: str,
        generation_id: str,
        cleanup_point_ids: Iterable[str] = (),
        previous_doc_ids: Iterable[str] = (),
    ) -> Optional[str]:
        previous_ids = list(dict.fromkeys(str(item) for item in previous_doc_ids if item))
        cleanup = list(dict.fromkeys(str(item) for item in cleanup_point_ids if item))
        with self._pool.connection() as conn:
            with conn.transaction():
                generation = conn.execute(
                    """
                    SELECT source_id, status
                    FROM knowledge_generations
                    WHERE generation_id = %s
                    FOR UPDATE
                    """,
                    (generation_id,),
                ).fetchone()
                if not generation:
                    raise ValueError(f"Unknown knowledge generation: {generation_id}")
                generation_source = str(
                    generation["source_id"] if isinstance(generation, dict) else generation[0]
                )
                generation_status = str(
                    generation["status"] if isinstance(generation, dict) else generation[1]
                )
                if generation_source != source_id:
                    raise ValueError(
                        f"Generation/source mismatch: generation={generation_id} source={source_id}"
                    )
                if generation_status != "prepared":
                    raise ValueError(
                        f"Generation {generation_id} must be prepared before activation; status={generation_status}"
                    )

                conn.execute("SELECT source_id FROM sources WHERE source_id = %s FOR UPDATE", (source_id,))
                active = conn.execute(
                    """
                    SELECT generation_id
                    FROM source_active_generations
                    WHERE source_id = %s
                    FOR UPDATE
                    """,
                    (source_id,),
                ).fetchone()
                old_generation_id = None
                if active:
                    old_generation_id = str(
                        active["generation_id"] if isinstance(active, dict) else active[0]
                    )

                if previous_ids:
                    conn.execute(
                        "DELETE FROM documents WHERE source_id = %s OR doc_id = ANY(%s)",
                        (source_id, previous_ids),
                    )
                else:
                    conn.execute("DELETE FROM documents WHERE source_id = %s", (source_id,))

                conn.execute(
                    """
                    INSERT INTO documents (
                        doc_id, tenant_id, source_type, title, uri, version,
                        doc_updated_at, ingested_at, tags, acl_policy_id, status,
                        source_id, generation_id
                    )
                    SELECT
                        doc_id, tenant_id, source_type, title, uri, version,
                        doc_updated_at, ingested_at, tags, acl_policy_id, status,
                        source_id, generation_id
                    FROM staged_documents
                    WHERE generation_id = %s
                    """,
                    (generation_id,),
                )
                conn.execute(
                    """
                    INSERT INTO chunks (
                        chunk_id, doc_id, tenant_id, chunk_index, text,
                        path, url, page, section, checksum, qdrant_point_id,
                        created_at, metadata, fts_text, source_id, generation_id
                    )
                    SELECT
                        chunk_id, doc_id, tenant_id, chunk_index, text,
                        path, url, page, section, checksum, qdrant_point_id,
                        created_at, metadata, fts_text, source_id, generation_id
                    FROM staged_chunks
                    WHERE generation_id = %s
                    """,
                    (generation_id,),
                )

                conn.execute(
                    """
                    INSERT INTO source_active_generations(source_id, generation_id, updated_at)
                    VALUES (%s, %s, NOW())
                    ON CONFLICT (source_id) DO UPDATE SET
                        generation_id = EXCLUDED.generation_id,
                        updated_at = EXCLUDED.updated_at
                    """,
                    (source_id, generation_id),
                )
                conn.execute(
                    """
                    UPDATE knowledge_generations
                    SET status = 'active', activated_at = NOW(), error = NULL
                    WHERE generation_id = %s
                    """,
                    (generation_id,),
                )
                if old_generation_id and old_generation_id != generation_id:
                    conn.execute(
                        """
                        UPDATE knowledge_generations
                        SET status = 'retired', retired_at = NOW()
                        WHERE generation_id = %s AND status = 'active'
                        """,
                        (old_generation_id,),
                    )

                self._enqueue_cleanup_events(
                    conn,
                    source_id=source_id,
                    generation_id=old_generation_id,
                    point_ids=cleanup,
                )
                conn.execute("DELETE FROM staged_documents WHERE generation_id = %s", (generation_id,))
                return old_generation_id

    def fail_knowledge_generation(
        self,
        generation_id: str,
        error: str,
        cleanup_point_ids: Iterable[str] = (),
    ) -> None:
        cleanup = list(dict.fromkeys(str(item) for item in cleanup_point_ids if item))
        with self._pool.connection() as conn:
            with conn.transaction():
                row = conn.execute(
                    "SELECT source_id FROM knowledge_generations WHERE generation_id = %s FOR UPDATE",
                    (generation_id,),
                ).fetchone()
                if not row:
                    return
                source_id = str(row["source_id"] if isinstance(row, dict) else row[0])
                conn.execute(
                    """
                    UPDATE knowledge_generations
                    SET status = 'failed', failed_at = NOW(), error = %s
                    WHERE generation_id = %s AND status NOT IN ('active', 'retired')
                    """,
                    (str(error)[:4000], generation_id),
                )
                self._enqueue_cleanup_events(
                    conn,
                    source_id=source_id,
                    generation_id=generation_id,
                    point_ids=cleanup,
                )
                conn.execute("DELETE FROM staged_documents WHERE generation_id = %s", (generation_id,))

    def _enqueue_cleanup_events(
        self,
        conn: Any,
        *,
        source_id: str,
        generation_id: Optional[str],
        point_ids: Iterable[str],
    ) -> None:
        for batch in _batches(point_ids):
            conn.execute(
                """
                INSERT INTO publication_outbox(event_type, source_id, generation_id, payload)
                VALUES ('delete_qdrant_points', %s, %s, %s)
                """,
                (source_id, generation_id, self._jsonb({"point_ids": batch})),
            )

    def get_active_generation_id(self, source_id: str) -> Optional[str]:
        with self._pool.connection() as conn:
            row = conn.execute(
                "SELECT generation_id FROM source_active_generations WHERE source_id = %s",
                (source_id,),
            ).fetchone()
        if not row:
            return None
        return str(row["generation_id"] if isinstance(row, dict) else row[0])

    def active_vector_points(self, chunk_ids: Iterable[str]) -> Dict[str, str]:
        ids = list(dict.fromkeys(str(item) for item in chunk_ids if item))
        if not ids:
            return {}
        with self._pool.connection() as conn:
            rows = conn.execute(
                """
                SELECT chunk_id, qdrant_point_id
                FROM chunks
                WHERE chunk_id = ANY(%s) AND qdrant_point_id IS NOT NULL
                """,
                (ids,),
            ).fetchall()
        result: Dict[str, str] = {}
        for row in rows:
            if isinstance(row, dict):
                result[str(row["chunk_id"])] = str(row["qdrant_point_id"])
            else:
                result[str(row[0])] = str(row[1])
        return result

    def claim_publication_outbox(
        self,
        worker_id: str,
        lease_seconds: int = 120,
        limit: int = 10,
    ) -> List[PublicationOutboxEvent]:
        self.reconcile_publication_outbox()
        with self._pool.connection() as conn:
            rows = conn.execute(
                """
                WITH ready AS (
                    SELECT outbox_id
                    FROM publication_outbox
                    WHERE status = 'pending' AND available_at <= NOW()
                    ORDER BY outbox_id
                    FOR UPDATE SKIP LOCKED
                    LIMIT %s
                )
                UPDATE publication_outbox AS o
                SET status = 'running',
                    attempts = o.attempts + 1,
                    lease_owner = %s,
                    lease_expires_at = NOW() + (%s * INTERVAL '1 second')
                FROM ready
                WHERE o.outbox_id = ready.outbox_id
                RETURNING o.*
                """,
                (max(1, int(limit)), worker_id, max(1, int(lease_seconds))),
            ).fetchall()
        return [self._row_to_publication_event(row) for row in rows]

    @staticmethod
    def _row_to_publication_event(row: Any) -> PublicationOutboxEvent:
        data = dict(row) if hasattr(row, "keys") else {
            "outbox_id": row[0],
            "event_type": row[1],
            "source_id": row[2],
            "generation_id": row[3],
            "payload": row[4],
            "status": row[5],
            "attempts": row[6],
            "available_at": row[7],
            "lease_owner": row[8],
            "lease_expires_at": row[9],
            "last_error": row[10],
            "created_at": row[11],
            "completed_at": row[12],
        }
        payload = data.get("payload") or {}
        return PublicationOutboxEvent(
            outbox_id=int(data["outbox_id"]),
            event_type=str(data["event_type"]),
            source_id=str(data["source_id"]),
            generation_id=data.get("generation_id"),
            payload=dict(payload),
            status=str(data.get("status") or "pending"),
            attempts=int(data.get("attempts") or 0),
            available_at=_iso(data.get("available_at")),
            lease_owner=data.get("lease_owner"),
            lease_expires_at=_iso(data.get("lease_expires_at")),
            last_error=data.get("last_error"),
            created_at=_iso(data.get("created_at")),
            completed_at=_iso(data.get("completed_at")),
        )

    def complete_publication_outbox(self, outbox_id: int, worker_id: str) -> bool:
        with self._pool.connection() as conn:
            result = conn.execute(
                """
                UPDATE publication_outbox
                SET status = 'completed', completed_at = NOW(),
                    lease_owner = NULL, lease_expires_at = NULL, last_error = NULL
                WHERE outbox_id = %s AND status = 'running' AND lease_owner = %s
                """,
                (int(outbox_id), worker_id),
            )
            return (result.rowcount or 0) > 0

    def retry_publication_outbox(
        self,
        outbox_id: int,
        worker_id: str,
        error: str,
        delay_seconds: float,
        max_attempts: int = 10,
    ) -> bool:
        with self._pool.connection() as conn:
            row = conn.execute(
                "SELECT attempts FROM publication_outbox WHERE outbox_id = %s AND lease_owner = %s FOR UPDATE",
                (int(outbox_id), worker_id),
            ).fetchone()
            if not row:
                return False
            attempts = int(row["attempts"] if isinstance(row, dict) else row[0])
            if attempts >= max_attempts:
                result = conn.execute(
                    """
                    UPDATE publication_outbox
                    SET status = 'failed', last_error = %s,
                        lease_owner = NULL, lease_expires_at = NULL
                    WHERE outbox_id = %s AND lease_owner = %s
                    """,
                    (str(error)[:4000], int(outbox_id), worker_id),
                )
            else:
                result = conn.execute(
                    """
                    UPDATE publication_outbox
                    SET status = 'pending', last_error = %s,
                        available_at = NOW() + (%s * INTERVAL '1 second'),
                        lease_owner = NULL, lease_expires_at = NULL
                    WHERE outbox_id = %s AND lease_owner = %s
                    """,
                    (max(0.0, float(delay_seconds)), str(error)[:4000], int(outbox_id), worker_id),
                )
            return (result.rowcount or 0) > 0

    def reconcile_publication_outbox(self, max_attempts: int = 10) -> Dict[str, int]:
        with self._pool.connection() as conn:
            with conn.transaction():
                failed = conn.execute(
                    """
                    UPDATE publication_outbox
                    SET status = 'failed', lease_owner = NULL, lease_expires_at = NULL,
                        last_error = COALESCE(last_error, 'publication outbox lease exhausted')
                    WHERE status = 'running'
                      AND lease_expires_at < NOW()
                      AND attempts >= %s
                    """,
                    (max_attempts,),
                ).rowcount or 0
                recovered = conn.execute(
                    """
                    UPDATE publication_outbox
                    SET status = 'pending', available_at = NOW(),
                        lease_owner = NULL, lease_expires_at = NULL
                    WHERE status = 'running'
                      AND lease_expires_at < NOW()
                      AND attempts < %s
                    """,
                    (max_attempts,),
                ).rowcount or 0
        return {"recovered": int(recovered), "failed": int(failed)}


class InMemoryGenerationMixin:
    """Generation publication semantics for the development/test repository."""

    def _ensure_generation_state(self) -> None:
        if not hasattr(self, "_knowledge_generations"):
            self._knowledge_generations: Dict[str, KnowledgeGeneration] = {}
            self._active_generations: Dict[str, str] = {}
            self._staged_documents: Dict[str, Dict[str, Document]] = {}
            self._staged_chunks: Dict[str, Dict[str, Chunk]] = {}
            self._publication_outbox: Dict[int, PublicationOutboxEvent] = {}
            self._publication_outbox_next_id = 1

    def begin_knowledge_generation(self, generation: KnowledgeGeneration) -> None:
        with self._lock:
            self._ensure_generation_state()
            self._knowledge_generations.setdefault(generation.generation_id, generation)

    def stage_knowledge_generation(
        self,
        generation_id: str,
        documents: Iterable[Document],
        chunks: Iterable[Chunk],
    ) -> Dict[str, int]:
        docs = list(documents)
        items = list(chunks)
        with self._lock:
            self._ensure_generation_state()
            generation = self._knowledge_generations.get(generation_id)
            if generation is None or generation.status not in {"staging", "prepared"}:
                raise ValueError(f"Generation is not stageable: {generation_id}")
            self._staged_documents[generation_id] = {doc.doc_id: doc for doc in docs}
            self._staged_chunks[generation_id] = {chunk.chunk_id: chunk for chunk in items}
        return {"documents": len(docs), "chunks": len(items)}

    def mark_knowledge_generation_prepared(
        self,
        generation_id: str,
        stats: Optional[Dict[str, Any]] = None,
    ) -> None:
        with self._lock:
            self._ensure_generation_state()
            generation = self._knowledge_generations.get(generation_id)
            if generation is None or generation.status not in {"staging", "prepared"}:
                raise ValueError(f"Generation is not preparable: {generation_id}")
            generation.status = "prepared"
            generation.prepared_at = _now().isoformat()
            generation.stats = dict(stats or {})

    def activate_knowledge_generation(
        self,
        source_id: str,
        generation_id: str,
        cleanup_point_ids: Iterable[str] = (),
        previous_doc_ids: Iterable[str] = (),
    ) -> Optional[str]:
        with self._lock:
            self._ensure_generation_state()
            generation = self._knowledge_generations.get(generation_id)
            if generation is None or generation.status != "prepared" or generation.source_id != source_id:
                raise ValueError(f"Generation is not activatable: {generation_id}")
            old_generation_id = self._active_generations.get(source_id)
            previous = set(str(item) for item in previous_doc_ids if item)
            previous.update(
                doc_id
                for doc_id, doc in self._documents.items()
                if getattr(doc, "source_id", None) == source_id
            )
            for doc_id in previous:
                self._documents.pop(doc_id, None)
            for chunk_id in [cid for cid, chunk in self._chunks.items() if chunk.doc_id in previous]:
                self._chunks.pop(chunk_id, None)
            for doc in self._staged_documents.get(generation_id, {}).values():
                self._documents[doc.doc_id] = doc
            for chunk in self._staged_chunks.get(generation_id, {}).values():
                self._chunks[chunk.chunk_id] = chunk
            self._active_generations[source_id] = generation_id
            generation.status = "active"
            generation.activated_at = _now().isoformat()
            if old_generation_id and old_generation_id != generation_id:
                old = self._knowledge_generations.get(old_generation_id)
                if old is not None:
                    old.status = "retired"
                    old.retired_at = _now().isoformat()
            self._enqueue_memory_cleanup(source_id, old_generation_id, cleanup_point_ids)
            self._staged_documents.pop(generation_id, None)
            self._staged_chunks.pop(generation_id, None)
            return old_generation_id

    def fail_knowledge_generation(
        self,
        generation_id: str,
        error: str,
        cleanup_point_ids: Iterable[str] = (),
    ) -> None:
        with self._lock:
            self._ensure_generation_state()
            generation = self._knowledge_generations.get(generation_id)
            if generation is None or generation.status in {"active", "retired"}:
                return
            generation.status = "failed"
            generation.failed_at = _now().isoformat()
            generation.error = str(error)
            self._enqueue_memory_cleanup(generation.source_id, generation_id, cleanup_point_ids)
            self._staged_documents.pop(generation_id, None)
            self._staged_chunks.pop(generation_id, None)

    def _enqueue_memory_cleanup(
        self,
        source_id: str,
        generation_id: Optional[str],
        point_ids: Iterable[str],
    ) -> None:
        self._ensure_generation_state()
        for batch in _batches(point_ids):
            event_id = self._publication_outbox_next_id
            self._publication_outbox_next_id += 1
            self._publication_outbox[event_id] = PublicationOutboxEvent(
                outbox_id=event_id,
                event_type="delete_qdrant_points",
                source_id=source_id,
                generation_id=generation_id,
                payload={"point_ids": batch},
                created_at=_now().isoformat(),
                available_at=_now().isoformat(),
            )

    def get_active_generation_id(self, source_id: str) -> Optional[str]:
        with self._lock:
            self._ensure_generation_state()
            return self._active_generations.get(source_id)

    def active_vector_points(self, chunk_ids: Iterable[str]) -> Dict[str, str]:
        ids = set(str(item) for item in chunk_ids if item)
        with self._lock:
            return {
                chunk_id: str(chunk.qdrant_point_id)
                for chunk_id, chunk in self._chunks.items()
                if chunk_id in ids and chunk.qdrant_point_id
            }

    def claim_publication_outbox(
        self,
        worker_id: str,
        lease_seconds: int = 120,
        limit: int = 10,
    ) -> List[PublicationOutboxEvent]:
        now = _now()
        self.reconcile_publication_outbox()
        claimed: List[PublicationOutboxEvent] = []
        with self._lock:
            self._ensure_generation_state()
            for event in sorted(self._publication_outbox.values(), key=lambda item: item.outbox_id):
                if len(claimed) >= max(1, int(limit)) or event.status != "pending":
                    continue
                available = datetime.fromisoformat(event.available_at) if event.available_at else now
                if available > now:
                    continue
                event.status = "running"
                event.attempts += 1
                event.lease_owner = worker_id
                event.lease_expires_at = (now + timedelta(seconds=max(1, lease_seconds))).isoformat()
                claimed.append(event)
        return claimed

    def complete_publication_outbox(self, outbox_id: int, worker_id: str) -> bool:
        with self._lock:
            self._ensure_generation_state()
            event = self._publication_outbox.get(int(outbox_id))
            if event is None or event.status != "running" or event.lease_owner != worker_id:
                return False
            event.status = "completed"
            event.completed_at = _now().isoformat()
            event.lease_owner = None
            event.lease_expires_at = None
            event.last_error = None
            return True

    def retry_publication_outbox(
        self,
        outbox_id: int,
        worker_id: str,
        error: str,
        delay_seconds: float,
        max_attempts: int = 10,
    ) -> bool:
        with self._lock:
            self._ensure_generation_state()
            event = self._publication_outbox.get(int(outbox_id))
            if event is None or event.lease_owner != worker_id:
                return False
            event.last_error = str(error)
            event.lease_owner = None
            event.lease_expires_at = None
            if event.attempts >= max_attempts:
                event.status = "failed"
            else:
                event.status = "pending"
                event.available_at = (_now() + timedelta(seconds=max(0.0, delay_seconds))).isoformat()
            return True

    def reconcile_publication_outbox(self, max_attempts: int = 10) -> Dict[str, int]:
        now = _now()
        recovered = 0
        failed = 0
        with self._lock:
            self._ensure_generation_state()
            for event in self._publication_outbox.values():
                if event.status != "running" or not event.lease_expires_at:
                    continue
                if datetime.fromisoformat(event.lease_expires_at) >= now:
                    continue
                event.lease_owner = None
                event.lease_expires_at = None
                if event.attempts >= max_attempts:
                    event.status = "failed"
                    failed += 1
                else:
                    event.status = "pending"
                    event.available_at = now.isoformat()
                    recovered += 1
        return {"recovered": recovered, "failed": failed}
