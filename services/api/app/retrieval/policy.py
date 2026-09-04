from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from .lexical import contains_cjk, lexicalize

VALID_RETRIEVAL_MODES = frozenset({"vector", "lexical", "hybrid"})
DEFAULT_CANDIDATE_POOL = 40
MAX_CANDIDATE_POOL = 200

_ENGLISH_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "between",
    "by", "can", "could", "did", "do", "does", "for", "from", "had", "has",
    "have", "how", "i", "if", "in", "into", "is", "it", "its", "of", "on",
    "or", "our", "should", "that", "the", "their", "them", "there", "these",
    "this", "those", "to", "was", "were", "what", "when", "where", "which",
    "who", "why", "will", "with", "would", "you", "your",
}


@dataclass(frozen=True)
class FusionPolicy:
    vector_weight: float
    lexical_weight: float
    lexical_confidence: float
    cross_language_lexical: bool
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "vector_weight": round(self.vector_weight, 4),
            "lexical_weight": round(self.lexical_weight, 4),
            "lexical_confidence": round(self.lexical_confidence, 4),
            "cross_language_lexical": self.cross_language_lexical,
            "reason": self.reason,
        }


def validate_retrieval_mode(mode: str) -> str:
    normalized = str(mode or "hybrid").strip().lower()
    if normalized not in VALID_RETRIEVAL_MODES:
        raise ValueError(
            f"Unsupported retrieval mode {mode!r}; expected one of {sorted(VALID_RETRIEVAL_MODES)}"
        )
    return normalized


def resolve_candidate_pool(top_k: int, requested: int | None = None) -> int:
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    raw = requested
    if raw is None:
        configured = os.getenv("RAGBOT_RETRIEVAL_CANDIDATE_POOL", "").strip()
        raw = int(configured) if configured else max(DEFAULT_CANDIDATE_POOL, top_k * 4)
    return max(top_k, min(MAX_CANDIDATE_POOL, int(raw)))


def meaningful_terms(text: str) -> list[str]:
    terms = lexicalize(text).split()
    return [term for term in terms if term not in _ENGLISH_STOPWORDS]


def lexical_confidence(query: str, lexical_hits: Sequence[Any]) -> tuple[float, bool]:
    """Estimate whether lexical ranking expresses the query rather than one token.

    The score is query-term coverage over the first three lexical candidates.
    For a CJK query against an English corpus, ASCII terms such as ``GPU`` can
    otherwise make FTS rank a document first despite almost no query-language
    evidence. That situation is explicitly marked cross-language and capped by
    the adaptive fusion policy.
    """
    if not lexical_hits:
        return 0.0, False

    query_terms = list(dict.fromkeys(meaningful_terms(query)))
    if not query_terms:
        return 0.0, False

    texts: list[str] = []
    for item in lexical_hits[:3]:
        chunk = item[0] if isinstance(item, tuple) else item
        texts.append(str(getattr(chunk, "text", "") or ""))
    joined = " ".join(texts)
    doc_terms = set(meaningful_terms(joined))
    matched = sum(1 for term in query_terms if term in doc_terms)
    coverage = matched / len(query_terms)

    cross_language = contains_cjk(query) and bool(joined) and not contains_cjk(joined)
    return coverage, cross_language


def adaptive_fusion_policy(
    query: str,
    vector_hits: Sequence[Any],
    lexical_hits: Sequence[Any],
    *,
    hash_fallback: bool,
) -> FusionPolicy:
    if not vector_hits and lexical_hits:
        return FusionPolicy(0.0, 1.0, 1.0, False, "vector-unavailable")
    if vector_hits and not lexical_hits:
        return FusionPolicy(1.0, 0.0, 0.0, False, "lexical-unavailable")
    if not vector_hits and not lexical_hits:
        return FusionPolicy(0.5, 0.5, 0.0, False, "no-candidates")

    confidence, cross_language = lexical_confidence(query, lexical_hits)
    if hash_fallback:
        return FusionPolicy(0.2, 0.8, confidence, cross_language, "hash-development-fallback")
    if cross_language:
        return FusionPolicy(0.9, 0.1, confidence, True, "cross-language-semantic-first")
    if confidence >= 0.65:
        return FusionPolicy(0.5, 0.5, confidence, False, "strong-lexical-evidence")
    if confidence >= 0.35:
        return FusionPolicy(0.65, 0.35, confidence, False, "moderate-lexical-evidence")
    return FusionPolicy(0.8, 0.2, confidence, False, "weak-lexical-evidence")
