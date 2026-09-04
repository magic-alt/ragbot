from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

from .embedder import Embedder, HashEmbedder
from .lexical import contains_cjk
from .pg_fts import fts_search
from .policy import adaptive_fusion_policy, resolve_candidate_pool, validate_retrieval_mode
from .rerank import rrf_fuse
from ..storage.models import Chunk
from ..storage.protocol import Repo
from contracts.types import RetrievalChunk

logger = logging.getLogger(__name__)


def build_citation(chunk: Chunk) -> str:
    location = chunk.path or chunk.url
    if location and chunk.page is not None:
        return f"{chunk.doc_id}:{location}:page={chunk.page}:chunk={chunk.chunk_index}"
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

    def diagnostics(
        self,
        query: Optional[str] = None,
        retrieval_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        embedder = self._embedder or HashEmbedder(dim=self._qdrant.dim)
        semantic = not isinstance(embedder, HashEmbedder)
        reranker_configured = bool(
            self._reranker
            and hasattr(self._reranker, "enabled")
            and self._reranker.enabled
        )
        warnings: List[str] = []
        fusion_mode = "adaptive-hybrid"
        if not semantic:
            fusion_mode = "lexical-first-development"
            warnings.append(
                "HashEmbedder is a development fallback, not a semantic embedding model. "
                "Configure EMBEDDING_MODEL and a real embedding endpoint, then re-ingest before "
                "judging semantic retrieval quality."
            )
            if query and contains_cjk(query):
                warnings.append(
                    "The current CJK query is not meaningfully represented by HashEmbedder; "
                    "the invalid hash-vector branch is disabled for this query. Cross-lingual "
                    "Chinese-to-English retrieval requires a multilingual semantic embedding model."
                )
        result = {
            "embedding_backend": type(embedder).__name__,
            "embedding_model": embedder.model_name,
            "embedding_dimension": embedder.dimension,
            "semantic_embedding": semantic,
            "vector_store": type(self._qdrant).__name__,
            "repository": type(self._repo).__name__,
            "reranker": type(self._reranker).__name__ if self._reranker is not None else None,
            "reranker_configured": reranker_configured,
            "reranker_enabled": reranker_configured,
            "fusion_mode": fusion_mode,
            "generation_visibility": callable(getattr(self._repo, "active_vector_points", None)),
            "warnings": warnings,
        }
        if retrieval_context:
            result.update(retrieval_context)
        return result

    def retrieve(
        self,
        query: str,
        filters: Dict[str, Any],
        top_k: int = 20,
        *,
        mode: str = "hybrid",
        candidate_pool: Optional[int] = None,
        rerank: bool = True,
    ) -> List[RetrievalChunk]:
        """Retrieve evidence using vector, lexical, or adaptive-hybrid ranking."""
        retrieval_mode = validate_retrieval_mode(mode)
        pool_size = resolve_candidate_pool(top_k, candidate_pool)
        embedder = self._embedder or HashEmbedder(dim=self._qdrant.dim)
        is_hash_fallback = isinstance(embedder, HashEmbedder)
        cjk_query = contains_cjk(query)
        active_points = getattr(self._repo, "active_vector_points", None)
        generation_visibility = callable(active_points)
        vector_search_limit = min(1000, max(pool_size, pool_size * 4)) if generation_visibility else pool_size

        def vector_search():
            if is_hash_fallback and cjk_query:
                return []
            embed_query = getattr(embedder, "embed_query", None)
            query_vector = embed_query(query) if callable(embed_query) else embedder.embed(query)
            hits = self._qdrant.search(query_vector, filters, vector_search_limit)
            if not generation_visibility or not hits:
                return hits[:pool_size]

            logical_ids = [
                str(payload.get("chunk_id") or point_id)
                for point_id, _score, payload in hits
            ]
            visible_points = active_points(logical_ids)
            visible = []
            for point_id, score, payload in hits:
                logical_id = str(payload.get("chunk_id") or point_id)
                expected_point_id = visible_points.get(logical_id)
                if expected_point_id is None:
                    continue
                if str(point_id) != str(expected_point_id):
                    continue
                visible.append((point_id, score, payload))
                if len(visible) >= pool_size:
                    break
            return visible

        def lexical_search():
            return fts_search(self._repo, query, filters, pool_size)

        if retrieval_mode == "hybrid":
            # Embedding+Qdrant and PostgreSQL lexical retrieval are independent
            # I/O branches. Fan them out concurrently, then fuse after both join.
            with ThreadPoolExecutor(max_workers=2, thread_name_prefix="ragbot-retrieval") as executor:
                vector_future = executor.submit(vector_search)
                lexical_future = executor.submit(lexical_search)
                qdrant_hits = vector_future.result()
                fts_hits = lexical_future.result()
        elif retrieval_mode == "vector":
            qdrant_hits = vector_search()
            fts_hits = []
        else:
            qdrant_hits = []
            fts_hits = lexical_search()

        qdrant_ranked: List[tuple[str, float]] = []
        payload_map: Dict[str, Dict[str, Any]] = {}
        vector_trace: Dict[str, Dict[str, Any]] = {}
        for rank, (point_id, score, payload) in enumerate(qdrant_hits, 1):
            logical_id = str(payload.get("chunk_id") or point_id)
            raw_score = float(score)
            qdrant_ranked.append((logical_id, raw_score))
            payload_map[logical_id] = payload
            vector_trace[logical_id] = {
                "rank": rank,
                "score": raw_score,
                "raw_score": raw_score,
                "physical_point_id": str(point_id),
                "generation_id": payload.get("generation_id"),
            }

        fts_ranked: List[tuple[str, float]] = []
        lexical_trace: Dict[str, Dict[str, Any]] = {}
        lexical_chunk_map: Dict[str, Chunk] = {}
        for rank, (chunk, score) in enumerate(fts_hits, 1):
            raw_score = float(score)
            fts_ranked.append((chunk.chunk_id, raw_score))
            lexical_chunk_map[chunk.chunk_id] = chunk
            lexical_trace[chunk.chunk_id] = {
                "rank": rank,
                "score": raw_score,
                "raw_score": raw_score,
            }

        fusion_policy: Dict[str, Any]
        fusion_method: str
        if retrieval_mode == "vector":
            ranked = list(qdrant_ranked)
            fusion_method = "vector-only"
            fusion_policy = {
                "vector_weight": 1.0,
                "lexical_weight": 0.0,
                "reason": "ablation-vector-only",
            }
        elif retrieval_mode == "lexical":
            ranked = list(fts_ranked)
            fusion_method = "lexical-only"
            fusion_policy = {
                "vector_weight": 0.0,
                "lexical_weight": 1.0,
                "reason": "ablation-lexical-only",
            }
        else:
            policy = adaptive_fusion_policy(
                query,
                qdrant_hits,
                fts_hits,
                hash_fallback=is_hash_fallback,
            )
            fusion_policy = policy.as_dict()
            fusion_method = "adaptive-rrf"
            ranked = rrf_fuse(
                qdrant_ranked,
                fts_ranked,
                weight_primary=policy.vector_weight,
                weight_secondary=policy.lexical_weight,
            )

        pre_rerank_scores = {chunk_id: float(score) for chunk_id, score in ranked}
        rerank_scores: Dict[str, float] = {}
        reranker_configured = bool(
            self._reranker
            and hasattr(self._reranker, "enabled")
            and self._reranker.enabled
        )
        reranker_enabled = bool(rerank and reranker_configured)

        if reranker_enabled and ranked:
            candidates = ranked[:pool_size]
            # Candidate text already arrived from PostgreSQL FTS or Qdrant
            # payloads. Reuse it instead of issuing one get_chunk query per hit.
            candidate_texts = [
                lexical_chunk_map[cid].text
                if cid in lexical_chunk_map
                else str(payload_map.get(cid, {}).get("text", ""))
                for cid, _score in candidates
            ]
            try:
                reranked = self._reranker.rerank(query, candidate_texts, top_k=top_k)
                valid: List[tuple[str, float]] = []
                for idx, score in reranked:
                    if isinstance(idx, int) and 0 <= idx < len(candidates):
                        chunk_id = candidates[idx][0]
                        rerank_scores[chunk_id] = float(score)
                        valid.append((chunk_id, float(score)))
                if valid:
                    ranked = valid
            except Exception:
                logger.exception("Optional reranker failed; falling back to pre-rerank ordering")

        retrieval_context = {
            "retrieval_mode": retrieval_mode,
            "candidate_pool": pool_size,
            "parallel_fanout": retrieval_mode == "hybrid",
            "generation_visibility_filter": generation_visibility,
            "vector_search_limit": vector_search_limit,
            "vector_candidates": len(qdrant_ranked),
            "lexical_candidates": len(fts_ranked),
            "fusion_method": fusion_method,
            "fusion_policy": fusion_policy,
            "reranker_configured": reranker_configured,
            "reranker_requested": bool(rerank),
            "reranker_enabled": reranker_enabled,
            "reranker_candidate_count": min(len(pre_rerank_scores), pool_size) if reranker_enabled else 0,
        }

        results: List[RetrievalChunk] = []
        for final_rank, (chunk_id, final_score) in enumerate(ranked[:top_k], 1):
            pre_score = pre_rerank_scores.get(chunk_id)
            trace = {
                "final_rank": final_rank,
                "final_score": float(final_score),
                "pre_rerank_score": pre_score,
                "rrf_score": pre_score if retrieval_mode == "hybrid" else None,
                "vector": vector_trace.get(chunk_id),
                "lexical": lexical_trace.get(chunk_id),
                "rerank_score": rerank_scores.get(chunk_id),
                "embedding_model": embedder.model_name,
                "fusion_mode": fusion_method,
                "context": retrieval_context,
            }
            chunk = lexical_chunk_map.get(chunk_id)
            if chunk is None:
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
                        score=float(final_score),
                        citations=[_payload_citation(payload, logical_id)],
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
                    "source_id": chunk.source_id,
                    "generation_id": chunk.generation_id,
                    "embedding_model": embedder.model_name,
                    "_retrieval": trace,
                }
            )
            results.append(
                RetrievalChunk(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    text=chunk.text,
                    score=float(final_score),
                    citations=citations,
                    metadata=metadata,
                )
            )
        return results


def _payload_citation(payload: Dict[str, Any], logical_id: str) -> str:
    doc_id = str(payload.get("doc_id") or "unknown")
    location = payload.get("path") or payload.get("url")
    page = payload.get("page")
    chunk_index = payload.get("chunk_index")
    if location and page is not None:
        return f"{doc_id}:{location}:page={page}:chunk={chunk_index}"
    if location:
        return f"{doc_id}:{location}:{chunk_index}"
    return logical_id
