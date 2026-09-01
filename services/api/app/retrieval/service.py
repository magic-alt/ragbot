from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .embedder import Embedder
from .pg_fts import fts_search
from .rerank import rrf_fuse
from ..storage.models import Chunk
from ..storage.protocol import Repo
from contracts.types import RetrievalChunk

logger = logging.getLogger(__name__)


def build_citation(chunk: Chunk) -> str:
    if chunk.path:
        return f"{chunk.doc_id}:{chunk.path}:{chunk.chunk_index}"
    if chunk.url:
        return f"{chunk.doc_id}:{chunk.url}:{chunk.chunk_index}"
    return f"{chunk.doc_id}:{chunk.chunk_index}"


class Retriever:
    def __init__(self, repo: Repo, qdrant: Any, embedder: Optional[Embedder] = None, reranker: Any = None) -> None:
        self._repo = repo
        self._qdrant = qdrant
        self._embedder = embedder
        self._reranker = reranker

    def retrieve(self, query: str, filters: Dict[str, Any], top_k: int = 20) -> List[RetrievalChunk]:
        if self._embedder:
            query_vector = self._embedder.embed(query)
        else:
            from .embedder import HashEmbedder
            query_vector = HashEmbedder(dim=self._qdrant.dim).embed(query)
        qdrant_hits = self._qdrant.search(query_vector, filters, top_k * 2)
        fts_hits = fts_search(self._repo, query, filters, top_k * 2)

        qdrant_ranked = [(point_id, score) for point_id, score, _payload in qdrant_hits]
        fts_ranked = [(chunk.chunk_id, score) for chunk, score in fts_hits]
        fused = rrf_fuse(qdrant_ranked, fts_ranked)

        payload_map: Dict[str, Dict[str, Any]] = {
            point_id: payload for point_id, _score, payload in qdrant_hits
        }

        # Reranking is an optional quality layer. A Cohere/local endpoint outage
        # must not take down otherwise healthy vector + lexical retrieval.
        if self._reranker and hasattr(self._reranker, "enabled") and self._reranker.enabled:
            candidates = fused[:top_k * 2]
            candidate_texts = []
            for cid, _ in candidates:
                chunk = self._repo.get_chunk(cid)
                if chunk:
                    candidate_texts.append(chunk.text)
                else:
                    candidate_texts.append(payload_map.get(cid, {}).get("text", ""))
            try:
                reranked = self._reranker.rerank(query, candidate_texts, top_k=top_k)
                valid = [
                    (candidates[idx][0], score)
                    for idx, score in reranked
                    if isinstance(idx, int) and 0 <= idx < len(candidates)
                ]
                if valid:
                    fused = valid
            except Exception:
                logger.exception("Optional reranker failed; falling back to RRF ordering")

        results: List[RetrievalChunk] = []
        for chunk_id, fused_score in fused[:top_k]:
            chunk = self._repo.get_chunk(chunk_id)
            if not chunk:
                payload = payload_map.get(chunk_id, {})
                if not payload:
                    continue
                results.append(
                    RetrievalChunk(
                        chunk_id=chunk_id,
                        doc_id=payload.get("doc_id", "unknown"),
                        text=payload.get("text", ""),
                        score=fused_score,
                        citations=[payload.get("citation", chunk_id)],
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
