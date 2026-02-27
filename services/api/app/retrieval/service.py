from __future__ import annotations

from typing import Any, Dict, List

from .pg_fts import fts_search
from .qdrant import embed_text
from .rerank import rrf_fuse
from ..storage.models import Chunk
from ..storage.repo import InMemoryRepo
from contracts.types import RetrievalChunk


def build_citation(chunk: Chunk) -> str:
    if chunk.path:
        return f"{chunk.doc_id}:{chunk.path}:{chunk.chunk_index}"
    if chunk.url:
        return f"{chunk.doc_id}:{chunk.url}:{chunk.chunk_index}"
    return f"{chunk.doc_id}:{chunk.chunk_index}"


class Retriever:
    def __init__(self, repo: InMemoryRepo, qdrant: Any) -> None:
        self._repo = repo
        self._qdrant = qdrant

    def retrieve(self, query: str, filters: Dict[str, Any], top_k: int = 20) -> List[RetrievalChunk]:
        query_vector = embed_text(query, self._qdrant.dim)
        qdrant_hits = self._qdrant.search(query_vector, filters, top_k * 2)
        fts_hits = fts_search(self._repo, query, filters, top_k * 2)

        qdrant_ranked = [(point_id, score) for point_id, score, _payload in qdrant_hits]
        fts_ranked = [(chunk.chunk_id, score) for chunk, score in fts_hits]
        fused = rrf_fuse(qdrant_ranked, fts_ranked)

        payload_map: Dict[str, Dict[str, Any]] = {
            point_id: payload for point_id, _score, payload in qdrant_hits
        }

        results: List[RetrievalChunk] = []
        for chunk_id, fused_score in fused[:top_k]:
            chunk = self._repo.get_chunk(chunk_id)
            if not chunk:
                payload = payload_map.get(chunk_id, {})
                if not payload:
                    continue
                text = payload.get("text", "")
                doc_id = payload.get("doc_id", "unknown")
                citations = [payload.get("citation", chunk_id)]
                results.append(
                    RetrievalChunk(
                        chunk_id=chunk_id,
                        doc_id=doc_id,
                        text=text,
                        score=fused_score,
                        citations=citations,
                        metadata=payload,
                    )
                )
                continue
            citations = [build_citation(chunk)]
            metadata = dict(chunk.metadata)
            metadata.update(
                {
                    "doc_id": chunk.doc_id,
                    "chunk_index": chunk.chunk_index,
                    "path": chunk.path,
                    "url": chunk.url,
                    "page": chunk.page,
                    "section": chunk.section,
                }
            )
            results.append(
                RetrievalChunk(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    text=chunk.text,
                    score=fused_score,
                    citations=citations,
                    metadata=metadata,
                )
            )
        return results

