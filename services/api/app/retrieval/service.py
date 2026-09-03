from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .embedder import Embedder, HashEmbedder
from .lexical import contains_cjk
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

    def diagnostics(self, query: Optional[str] = None) -> Dict[str, Any]:
        """Return non-secret runtime metadata useful for retrieval debugging."""
        embedder = self._embedder or HashEmbedder(dim=self._qdrant.dim)
        semantic = not isinstance(embedder, HashEmbedder)
        reranker_enabled = bool(
            self._reranker
            and hasattr(self._reranker, "enabled")
            and self._reranker.enabled
        )
        warnings: List[str] = []
        if not semantic:
            warnings.append(
                "HashEmbedder is a development fallback, not a semantic embedding model. "
                "Configure EMBEDDING_MODEL + EMBEDDING_API_KEY and re-ingest before judging semantic quality."
            )
            if query and contains_cjk(query):
                warnings.append(
                    "The current CJK query is not meaningfully represented by HashEmbedder; "
                    "cross-lingual Chinese-to-English retrieval requires a multilingual semantic embedding model."
                )
        return {
            "embedding_backend": type(embedder).__name__,
            "embedding_model": embedder.model_name,
            "embedding_dimension": embedder.dimension,
            "semantic_embedding": semantic,
            "vector_store": type(self._qdrant).__name__,
            "repository": type(self._repo).__name__,
            "reranker": type(self._reranker).__name__ if self._reranker is not None else None,
            "reranker_enabled": reranker_enabled,
            "warnings": warnings,
        }

    def retrieve(self, query: str, filters: Dict[str, Any], top_k: int = 20) -> List[RetrievalChunk]:
        embedder = self._embedder or HashEmbedder(dim=self._qdrant.dim)
        query_vector = embedder.embed(query)
        qdrant_hits = self._qdrant.search(query_vector, filters, top_k * 2)
        fts_hits = fts_search(self._repo, query, filters, top_k * 2)

        # Qdrant point IDs are storage UUIDs; ranking/fusion operates on Ragbot's
        # logical chunk IDs so the same hit from vector and lexical retrieval is
        # fused instead of appearing as two unrelated candidates.
        qdrant_ranked = []
        payload_map: Dict[str, Dict[str, Any]] = {}
        vector_trace: Dict[str, Dict[str, Any]] = {}
        for rank, (point_id, score, payload) in enumerate(qdrant_hits, 1):
            logical_id = str(payload.get("chunk_id") or point_id)
            qdrant_ranked.append((logical_id, score))
            payload_map[logical_id] = payload
            vector_trace[logical_id] = {"rank": rank, "score": float(score)}

        fts_ranked = []
        lexical_trace: Dict[str, Dict[str, Any]] = {}
        for rank, (chunk, score) in enumerate(fts_hits, 1):
            fts_ranked.append((chunk.chunk_id, score))
            lexical_trace[chunk.chunk_id] = {"rank": rank, "score": float(score)}

        fused = rrf_fuse(qdrant_ranked, fts_ranked)
        rrf_scores = {chunk_id: float(score) for chunk_id, score in fused}
        rerank_scores: Dict[str, float] = {}

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
                valid = []
                for idx, score in reranked:
                    if isinstance(idx, int) and 0 <= idx < len(candidates):
                        chunk_id = candidates[idx][0]
                        rerank_scores[chunk_id] = float(score)
                        valid.append((chunk_id, score))
                if valid:
                    fused = valid
            except Exception:
                logger.exception("Optional reranker failed; falling back to RRF ordering")

        results: List[RetrievalChunk] = []
        for final_rank, (chunk_id, fused_score) in enumerate(fused[:top_k], 1):
            trace = {
                "final_rank": final_rank,
                "final_score": float(fused_score),
                "rrf_score": rrf_scores.get(chunk_id),
                "vector": vector_trace.get(chunk_id),
                "lexical": lexical_trace.get(chunk_id),
                "rerank_score": rerank_scores.get(chunk_id),
                "embedding_model": embedder.model_name,
            }
            chunk = self._repo.get_chunk(chunk_id)
            if not chunk:
                payload = payload_map.get(chunk_id, {})
                if not payload:
                    continue
                logical_id = str(payload.get("chunk_id") or chunk_id)
                metadata = dict(payload)
                metadata["_retrieval"] = trace
                results.append(
                    RetrievalChunk(
                        chunk_id=logical_id,
                        doc_id=payload.get("doc_id", "unknown"),
                        text=payload.get("text", ""),
                        score=fused_score,
                        citations=[payload.get("citation", logical_id)],
                        metadata=metadata,
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
                    "qdrant_point_id": chunk.qdrant_point_id,
                    "embedding_model": embedder.model_name,
                    "_retrieval": trace,
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
