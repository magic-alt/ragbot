"""Native Ragbot/LangChain/LlamaIndex retrieval comparison on one Golden Dataset.

This is intentionally separate from ``benchmarks.rag_framework_compare``:
that benchmark holds cosine search constant and isolates the splitter. This
module exercises each framework's native retrieval abstraction while keeping
the corpus, queries, relevance labels, embedding model and top-k fixed.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
import time
import tracemalloc
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Protocol, Sequence

from services.api.app.retrieval.embedder import HashEmbedder, build_embedder

SUPPORTED_SUFFIXES = {".txt", ".md", ".rst", ".pdf"}
DEFAULT_SERVER = "http://127.0.0.1:8000"


@dataclass(frozen=True)
class CorpusUnit:
    doc_id: str
    path: str
    text: str
    page: Optional[int] = None


@dataclass
class RetrievedHit:
    chunk_id: str
    doc_id: str
    path: str
    text: str
    score: Optional[float] = None
    page: Optional[int] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CaseScore:
    case_id: str
    category: str
    query: str
    first_relevant_rank: Optional[int]
    reciprocal_rank_at_10: float
    precision_at_5: float
    precision_at_10: float
    recall_at_10: Optional[float]
    ndcg_at_10: float
    retrieval_pass: bool
    top_hits: list[RetrievedHit]


class NativeBackend(Protocol):
    name: str

    def build(self, units: Sequence[CorpusUnit]) -> dict[str, Any]: ...

    def search(
        self,
        query: str,
        *,
        top_k: int,
        filters: Optional[Mapping[str, Any]] = None,
    ) -> tuple[list[RetrievedHit], dict[str, Any]]: ...


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _norm(value: Any) -> str:
    return " ".join(str(value or "").casefold().split())


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    low = math.floor(pos)
    high = math.ceil(pos)
    if low == high:
        return ordered[low]
    weight = pos - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def load_golden_dataset(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Golden Dataset must be a JSON object")
    cases = raw.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Golden Dataset requires a non-empty 'cases' array")
    seen: set[str] = set()
    for index, case in enumerate(cases, 1):
        if not isinstance(case, dict):
            raise ValueError(f"cases[{index}] must be an object")
        case_id = str(case.get("id") or "").strip()
        query = str(case.get("query") or "").strip()
        if not case_id or not query:
            raise ValueError(f"cases[{index}] requires id and query")
        if case_id in seen:
            raise ValueError(f"Duplicate case id: {case_id}")
        seen.add(case_id)
    return raw


def _is_labeled(case: Mapping[str, Any]) -> bool:
    rel = case.get("relevance") or {}
    if not isinstance(rel, Mapping):
        return False
    return any(
        bool(rel.get(key))
        for key in (
            "expected_chunk_ids",
            "doc_ids",
            "pages",
            "path_contains",
            "all_terms",
            "any_terms",
        )
    )


def _stable_label(case: Mapping[str, Any]) -> bool:
    rel = case.get("relevance") or {}
    if not isinstance(rel, Mapping):
        return False
    return any(bool(rel.get(key)) for key in ("expected_chunk_ids", "doc_ids", "pages", "path_contains"))


def audit_golden_dataset(dataset: Mapping[str, Any], profile: str = "development") -> dict[str, Any]:
    cases = list(dataset.get("cases") or [])
    labeled = [case for case in cases if _is_labeled(case)]
    stable = [case for case in cases if _stable_label(case)]
    categories = sorted({str(case.get("category") or "default") for case in cases})
    with_answers = [case for case in cases if bool(case.get("answer"))]
    stats = {
        "cases": len(cases),
        "labeled_cases": len(labeled),
        "stable_label_cases": len(stable),
        "stable_label_rate": round(len(stable) / len(cases), 4) if cases else 0.0,
        "categories": categories,
        "category_count": len(categories),
        "answer_labeled_cases": len(with_answers),
    }
    checks: list[dict[str, Any]] = []

    def add(name: str, actual: Any, expected: str, passed: bool) -> None:
        checks.append({"name": name, "actual": actual, "expected": expected, "passed": bool(passed)})

    if profile == "off":
        return {"profile": profile, "passed": True, "stats": stats, "checks": []}
    if profile not in {"development", "production"}:
        raise ValueError(f"Unsupported Golden Dataset profile: {profile}")

    min_cases = 10 if profile == "development" else 50
    add("case_count", len(cases), f">={min_cases}", len(cases) >= min_cases)
    add("all_cases_labeled", len(labeled), f"={len(cases)}", len(labeled) == len(cases))
    add("category_count", len(categories), ">=2" if profile == "development" else ">=3", len(categories) >= (2 if profile == "development" else 3))
    if profile == "production":
        add("stable_label_rate", stats["stable_label_rate"], ">=0.80", stats["stable_label_rate"] >= 0.80)

    return {
        "profile": profile,
        "passed": all(item["passed"] for item in checks),
        "stats": stats,
        "checks": checks,
    }


def _load_pdf_units(path: Path, relative: str) -> list[CorpusUnit]:
    try:
        from PyPDF2 import PdfReader
    except ImportError as exc:  # pragma: no cover - optional worker dependency
        raise RuntimeError("PDF corpus loading requires the worker extra (PyPDF2)") from exc
    units: list[CorpusUnit] = []
    for page_index, page in enumerate(PdfReader(str(path)).pages, 1):
        text = (page.extract_text() or "").strip()
        if text:
            units.append(CorpusUnit(doc_id=relative, path=relative, text=text, page=page_index))
    return units


def load_corpus_units(directory: Path) -> list[CorpusUnit]:
    units: list[CorpusUnit] = []
    for path in sorted(item for item in directory.rglob("*") if item.is_file()):
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        relative = path.relative_to(directory).as_posix()
        if path.suffix.lower() == ".pdf":
            units.extend(_load_pdf_units(path, relative))
        else:
            text = path.read_text(encoding="utf-8", errors="replace").strip()
            if text:
                units.append(CorpusUnit(doc_id=relative, path=relative, text=text))
    if not units:
        raise ValueError(f"No supported documents found under {directory}")
    return units


def corpus_manifest(units: Sequence[CorpusUnit]) -> dict[str, Any]:
    digest = hashlib.sha256()
    doc_ids = sorted({unit.doc_id for unit in units})
    for unit in sorted(units, key=lambda item: (item.path, item.page or 0)):
        digest.update(unit.path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(unit.page or 0).encode("ascii"))
        digest.update(b"\0")
        digest.update(unit.text.encode("utf-8"))
        digest.update(b"\0")
    return {
        "documents": len(doc_ids),
        "units": len(units),
        "characters": sum(len(unit.text) for unit in units),
        "sha256": digest.hexdigest(),
    }


def _case_filters(dataset: Mapping[str, Any], case: Mapping[str, Any]) -> Optional[dict[str, Any]]:
    defaults = dataset.get("defaults") or {}
    merged = dict(defaults.get("filters") or {}) if isinstance(defaults, Mapping) else {}
    merged.update(case.get("filters") or {})
    return merged or None


def _match_relevance(hit: RetrievedHit, case: Mapping[str, Any]) -> bool:
    rel = case.get("relevance") or {}
    if not isinstance(rel, Mapping):
        return False

    expected_chunk_ids = {str(v) for v in _as_list(rel.get("expected_chunk_ids")) if v}
    if expected_chunk_ids:
        return hit.chunk_id in expected_chunk_ids

    doc_ids = {str(v) for v in _as_list(rel.get("doc_ids")) if v}
    if doc_ids and hit.doc_id not in doc_ids:
        return False

    pages = {str(v) for v in _as_list(rel.get("pages")) if v is not None}
    if pages and str(hit.page) not in pages:
        return False

    path_needles = [_norm(v) for v in _as_list(rel.get("path_contains")) if v]
    if path_needles and not any(needle in _norm(hit.path) for needle in path_needles):
        return False

    all_terms = [_norm(v) for v in _as_list(rel.get("all_terms")) if v]
    if all_terms and not all(term in _norm(hit.text) for term in all_terms):
        return False

    any_terms = [_norm(v) for v in _as_list(rel.get("any_terms")) if v]
    if any_terms and not any(term in _norm(hit.text) for term in any_terms):
        return False

    return bool(expected_chunk_ids or doc_ids or pages or path_needles or all_terms or any_terms)


def _relevance_entity(hit: RetrievedHit, case: Mapping[str, Any]) -> str:
    rel = case.get("relevance") or {}
    if rel.get("expected_chunk_ids"):
        return f"chunk:{hit.chunk_id}"
    if rel.get("pages"):
        return f"page:{hit.doc_id}:{hit.page}"
    return f"doc:{hit.doc_id or hit.path}"


def _known_relevant_total(case: Mapping[str, Any]) -> Optional[int]:
    rel = case.get("relevance") or {}
    if not isinstance(rel, Mapping):
        return None
    for key in ("expected_chunk_ids", "pages", "doc_ids", "path_contains"):
        values = [str(v) for v in _as_list(rel.get(key)) if v is not None and str(v)]
        if values:
            return len(set(values))
    if rel.get("all_terms") or rel.get("any_terms"):
        return 1
    return None


def score_case(case: Mapping[str, Any], hits: Sequence[RetrievedHit]) -> CaseScore:
    relevant_entities: set[str] = set()
    relevance_flags: list[int] = []
    first_rank: Optional[int] = None
    for rank, hit in enumerate(hits, 1):
        relevant = _match_relevance(hit, case)
        if relevant and first_rank is None:
            first_rank = rank
        if relevant:
            entity = _relevance_entity(hit, case)
            if entity in relevant_entities:
                relevant = False
            else:
                relevant_entities.add(entity)
        relevance_flags.append(1 if relevant else 0)

    def precision(k: int) -> float:
        return sum(relevance_flags[:k]) / float(k)

    dcg = sum(flag / math.log2(rank + 1) for rank, flag in enumerate(relevance_flags[:10], 1))
    total = _known_relevant_total(case)
    ideal_count = min(total or 1, 10)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
    recall = min(1.0, len(relevant_entities) / total) if total else None
    max_rank = int((case.get("relevance") or {}).get("max_rank") or 5)
    return CaseScore(
        case_id=str(case["id"]),
        category=str(case.get("category") or "default"),
        query=str(case["query"]),
        first_relevant_rank=first_rank,
        reciprocal_rank_at_10=(1.0 / first_rank if first_rank and first_rank <= 10 else 0.0),
        precision_at_5=precision(5),
        precision_at_10=precision(10),
        recall_at_10=recall,
        ndcg_at_10=(dcg / idcg if idcg else 0.0),
        retrieval_pass=first_rank is not None and first_rank <= max_rank,
        top_hits=list(hits),
    )


def summarize_scores(scores: Sequence[CaseScore], latencies_ms: Sequence[float]) -> dict[str, Any]:
    if not scores:
        raise ValueError("Cannot summarize zero cases")

    def hit_at(k: int) -> float:
        return sum(1 for score in scores if score.first_relevant_rank and score.first_relevant_rank <= k) / len(scores)

    recalls = [score.recall_at_10 for score in scores if score.recall_at_10 is not None]
    categories: dict[str, dict[str, Any]] = {}
    for category in sorted({score.category for score in scores}):
        items = [score for score in scores if score.category == category]
        categories[category] = {
            "cases": len(items),
            "hit_at_5": round(sum(1 for item in items if item.first_relevant_rank and item.first_relevant_rank <= 5) / len(items), 4),
            "mrr_at_10": round(statistics.fmean(item.reciprocal_rank_at_10 for item in items), 4),
            "ndcg_at_10": round(statistics.fmean(item.ndcg_at_10 for item in items), 4),
        }
    return {
        "cases": len(scores),
        "pass_rate": round(sum(1 for score in scores if score.retrieval_pass) / len(scores), 4),
        "hit_at_1": round(hit_at(1), 4),
        "hit_at_3": round(hit_at(3), 4),
        "hit_at_5": round(hit_at(5), 4),
        "hit_at_10": round(hit_at(10), 4),
        "mrr_at_10": round(statistics.fmean(score.reciprocal_rank_at_10 for score in scores), 4),
        "precision_at_5": round(statistics.fmean(score.precision_at_5 for score in scores), 4),
        "precision_at_10": round(statistics.fmean(score.precision_at_10 for score in scores), 4),
        "recall_at_10": round(statistics.fmean(recalls), 4) if recalls else None,
        "ndcg_at_10": round(statistics.fmean(score.ndcg_at_10 for score in scores), 4),
        "query_latency_ms_p50": round(_percentile(latencies_ms, 0.50), 3),
        "query_latency_ms_p95": round(_percentile(latencies_ms, 0.95), 3),
        "query_latency_ms_mean": round(statistics.fmean(latencies_ms), 3) if latencies_ms else 0.0,
        "queries_per_second": round(1000.0 / statistics.fmean(latencies_ms), 3) if latencies_ms and statistics.fmean(latencies_ms) > 0 else 0.0,
        "categories": categories,
    }


class RagbotHTTPBackend:
    name = "ragbot"

    def __init__(
        self,
        *,
        server: str,
        tenant: str,
        user: str,
        api_key: Optional[str],
        timeout: float,
        mode: str,
        rerank: bool,
    ) -> None:
        self.server = server.rstrip("/")
        self.tenant = tenant
        self.user = user
        self.api_key = api_key
        self.timeout = timeout
        self.mode = mode
        self.rerank = rerank
        self.runtime: dict[str, Any] = {}

    def build(self, units: Sequence[CorpusUnit]) -> dict[str, Any]:
        del units
        ready = self._request("/admin/ready", None)
        if ready.get("status") != "ready":
            raise RuntimeError(f"Ragbot is not ready: {ready}")
        return {"kind": "live-index", "index_seconds": 0.0, "chunks": None}

    def _request(self, path: str, payload: Optional[dict[str, Any]]) -> dict[str, Any]:
        headers = {"Accept": "application/json"}
        body = None
        method = "GET"
        if payload is not None:
            headers["Content-Type"] = "application/json"
            body = json.dumps(payload).encode("utf-8")
            method = "POST"
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        request = urllib.request.Request(f"{self.server}{path}", data=body, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code} {path}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Could not reach Ragbot at {self.server}: {exc}") from exc

    def search(
        self,
        query: str,
        *,
        top_k: int,
        filters: Optional[Mapping[str, Any]] = None,
    ) -> tuple[list[RetrievedHit], dict[str, Any]]:
        payload = {
            "query": query,
            "tenant_id": self.tenant,
            "user_id": self.user,
            "top_k": top_k,
            "mode": self.mode,
            "rerank": self.rerank,
            "filters": dict(filters) if filters else None,
        }
        result = self._request("/search", payload)
        diagnostics = dict(result.get("diagnostics") or {})
        if diagnostics:
            self.runtime = diagnostics
        hits = []
        for index, chunk in enumerate(result.get("chunks") or []):
            metadata = dict(chunk.get("metadata") or {})
            hits.append(
                RetrievedHit(
                    chunk_id=str(chunk.get("chunk_id") or f"ragbot-{index}"),
                    doc_id=str(chunk.get("doc_id") or ""),
                    path=str(metadata.get("path") or metadata.get("url") or ""),
                    page=metadata.get("page"),
                    text=str(chunk.get("text") or ""),
                    score=float(chunk["score"]) if chunk.get("score") is not None else None,
                    metadata=metadata,
                )
            )
        return hits, diagnostics


class LangChainNativeBackend:
    name = "langchain"

    def __init__(self, *, embedder: Any, chunk_size: int, chunk_overlap: int) -> None:
        self.embedder = embedder
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._store: Any = None

    def build(self, units: Sequence[CorpusUnit]) -> dict[str, Any]:
        try:
            from langchain_core.documents import Document
            from langchain_core.embeddings import Embeddings
            from langchain_core.vectorstores import InMemoryVectorStore
            from langchain_text_splitters import RecursiveCharacterTextSplitter
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("LangChain native benchmark requires .[benchmark-frameworks]") from exc

        delegate = self.embedder

        class RagbotEmbeddings(Embeddings):
            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                return delegate.embed_batch(texts)

            def embed_query(self, text: str) -> list[float]:
                method = getattr(delegate, "embed_query", None)
                return method(text) if callable(method) else delegate.embed(text)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
        )
        documents = []
        ids = []
        split_started = time.perf_counter()
        for unit in units:
            for index, text in enumerate(splitter.split_text(unit.text)):
                chunk_id = f"{unit.doc_id}::p{unit.page or 0}::{index}"
                metadata = {"doc_id": unit.doc_id, "path": unit.path, "chunk_id": chunk_id}
                if unit.page is not None:
                    metadata["page"] = unit.page
                documents.append(Document(id=chunk_id, page_content=text, metadata=metadata))
                ids.append(chunk_id)
        split_seconds = time.perf_counter() - split_started
        started = time.perf_counter()
        self._store = InMemoryVectorStore(embedding=RagbotEmbeddings())
        self._store.add_documents(documents=documents, ids=ids)
        index_seconds = time.perf_counter() - started
        return {
            "kind": "native-vector-store",
            "splitter": "RecursiveCharacterTextSplitter",
            "chunks": len(documents),
            "split_seconds": round(split_seconds, 6),
            "index_seconds": round(index_seconds, 6),
        }

    def search(
        self,
        query: str,
        *,
        top_k: int,
        filters: Optional[Mapping[str, Any]] = None,
    ) -> tuple[list[RetrievedHit], dict[str, Any]]:
        del filters
        if self._store is None:
            raise RuntimeError("LangChain backend has not been built")
        results = self._store.similarity_search_with_score(query, k=top_k)
        hits = []
        for index, (document, score) in enumerate(results):
            metadata = dict(document.metadata or {})
            hits.append(
                RetrievedHit(
                    chunk_id=str(metadata.get("chunk_id") or document.id or f"langchain-{index}"),
                    doc_id=str(metadata.get("doc_id") or ""),
                    path=str(metadata.get("path") or ""),
                    page=metadata.get("page"),
                    text=str(document.page_content or ""),
                    score=float(score) if score is not None else None,
                    metadata=metadata,
                )
            )
        return hits, {}


class LlamaIndexNativeBackend:
    name = "llamaindex"

    def __init__(self, *, embedder: Any, chunk_size: int, chunk_overlap: int) -> None:
        self.embedder = embedder
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._index: Any = None

    def build(self, units: Sequence[CorpusUnit]) -> dict[str, Any]:
        try:
            from llama_index.core import Document, VectorStoreIndex
            from llama_index.core.bridge.pydantic import PrivateAttr
            from llama_index.core.embeddings import BaseEmbedding
            from llama_index.core.node_parser import SentenceSplitter
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("LlamaIndex native benchmark requires .[benchmark-frameworks]") from exc

        delegate = self.embedder

        class RagbotEmbedding(BaseEmbedding):
            _delegate: Any = PrivateAttr()

            def __init__(self) -> None:
                super().__init__(model_name=str(delegate.model_name))
                self._delegate = delegate

            def _get_query_embedding(self, query: str) -> list[float]:
                method = getattr(self._delegate, "embed_query", None)
                return method(query) if callable(method) else self._delegate.embed(query)

            async def _aget_query_embedding(self, query: str) -> list[float]:
                return self._get_query_embedding(query)

            def _get_text_embedding(self, text: str) -> list[float]:
                return self._delegate.embed(text)

            def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
                return self._delegate.embed_batch(texts)

            async def _aget_text_embedding(self, text: str) -> list[float]:
                return self._get_text_embedding(text)

        documents = []
        for index, unit in enumerate(units):
            metadata = {"doc_id": unit.doc_id, "path": unit.path}
            if unit.page is not None:
                metadata["page"] = unit.page
            documents.append(Document(text=unit.text, metadata=metadata, id_=f"{unit.doc_id}::unit::{unit.page or index}"))

        splitter = SentenceSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            tokenizer=list,
        )
        split_started = time.perf_counter()
        nodes = splitter.get_nodes_from_documents(documents)
        split_seconds = time.perf_counter() - split_started
        started = time.perf_counter()
        self._index = VectorStoreIndex(nodes, embed_model=RagbotEmbedding(), show_progress=False)
        index_seconds = time.perf_counter() - started
        return {
            "kind": "native-vector-store",
            "splitter": "SentenceSplitter(character-equivalent budget)",
            "chunks": len(nodes),
            "split_seconds": round(split_seconds, 6),
            "index_seconds": round(index_seconds, 6),
        }

    def search(
        self,
        query: str,
        *,
        top_k: int,
        filters: Optional[Mapping[str, Any]] = None,
    ) -> tuple[list[RetrievedHit], dict[str, Any]]:
        del filters
        if self._index is None:
            raise RuntimeError("LlamaIndex backend has not been built")
        retriever = self._index.as_retriever(similarity_top_k=top_k)
        results = retriever.retrieve(query)
        hits = []
        for index, item in enumerate(results):
            node = item.node
            metadata = dict(node.metadata or {})
            content = node.get_content() if hasattr(node, "get_content") else str(getattr(node, "text", ""))
            hits.append(
                RetrievedHit(
                    chunk_id=str(getattr(node, "node_id", None) or getattr(node, "id_", None) or f"llamaindex-{index}"),
                    doc_id=str(metadata.get("doc_id") or ""),
                    path=str(metadata.get("path") or ""),
                    page=metadata.get("page"),
                    text=str(content or ""),
                    score=float(item.score) if item.score is not None else None,
                    metadata=metadata,
                )
            )
        return hits, {}


def _backend_names(raw: str | Sequence[str]) -> list[str]:
    values = [value.strip().lower() for value in (raw.split(",") if isinstance(raw, str) else raw) if value.strip()]
    supported = {"ragbot", "langchain", "llamaindex"}
    unknown = [value for value in values if value not in supported]
    if unknown:
        raise ValueError(f"Unsupported native benchmark backends: {unknown}")
    if not values:
        raise ValueError("At least one native benchmark backend is required")
    return values


def _backend_versions(backends: Sequence[str]) -> dict[str, Optional[str]]:
    from importlib.metadata import PackageNotFoundError, version

    packages = {"langchain": "langchain-core", "llamaindex": "llama-index-core", "ragbot": "ragbot"}
    result: dict[str, Optional[str]] = {}
    for backend in backends:
        try:
            result[backend] = version(packages[backend])
        except PackageNotFoundError:
            result[backend] = None
    return result


def _make_backend(
    name: str,
    *,
    embedder: Any,
    chunk_size: int,
    chunk_overlap: int,
    server: str,
    tenant: str,
    user: str,
    api_key: Optional[str],
    timeout: float,
    ragbot_mode: str,
    rerank: bool,
) -> NativeBackend:
    if name == "ragbot":
        return RagbotHTTPBackend(server=server, tenant=tenant, user=user, api_key=api_key, timeout=timeout, mode=ragbot_mode, rerank=rerank)
    if name == "langchain":
        return LangChainNativeBackend(embedder=embedder, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if name == "llamaindex":
        return LlamaIndexNativeBackend(embedder=embedder, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    raise ValueError(name)


def _score_dict(score: CaseScore) -> dict[str, Any]:
    return {
        "case_id": score.case_id,
        "category": score.category,
        "query": score.query,
        "first_relevant_rank": score.first_relevant_rank,
        "reciprocal_rank_at_10": round(score.reciprocal_rank_at_10, 6),
        "precision_at_5": round(score.precision_at_5, 6),
        "precision_at_10": round(score.precision_at_10, 6),
        "recall_at_10": round(score.recall_at_10, 6) if score.recall_at_10 is not None else None,
        "ndcg_at_10": round(score.ndcg_at_10, 6),
        "retrieval_pass": score.retrieval_pass,
        "top_hits": [
            {
                "rank": rank,
                "chunk_id": hit.chunk_id,
                "doc_id": hit.doc_id,
                "path": hit.path,
                "page": hit.page,
                "score": hit.score,
                "text": " ".join(hit.text.split())[:500],
            }
            for rank, hit in enumerate(score.top_hits, 1)
        ],
    }


def _embedding_match(runtime: Mapping[str, Any], embedder: Any) -> dict[str, Any]:
    model = runtime.get("embedding_model")
    dimension = runtime.get("embedding_dimension")
    model_match = not model or str(model) == str(embedder.model_name)
    dimension_match = dimension in (None, "") or int(dimension) == int(embedder.dimension)
    return {
        "passed": bool(model_match and dimension_match),
        "ragbot_model": model,
        "ragbot_dimension": dimension,
        "local_model": embedder.model_name,
        "local_dimension": embedder.dimension,
    }


def run_comparison(
    *,
    dataset: Mapping[str, Any],
    units: Sequence[CorpusUnit],
    backends: str | Sequence[str] = "ragbot,langchain,llamaindex",
    embedding: str = "env",
    hash_dimension: int = 256,
    chunk_size: int = 800,
    chunk_overlap: int = 100,
    top_k: int = 10,
    repetitions: int = 1,
    server: str = DEFAULT_SERVER,
    tenant: str = "default",
    user: str = "rag-benchmark",
    api_key: Optional[str] = None,
    timeout: float = 60.0,
    ragbot_mode: str = "vector",
    rerank: bool = False,
    enforce_embedding_match: bool = True,
) -> dict[str, Any]:
    if top_k < 10:
        raise ValueError("top_k must be >= 10 because the benchmark reports @10 metrics")
    if repetitions < 1:
        raise ValueError("repetitions must be >= 1")
    if chunk_size < 2 or chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk budget must satisfy chunk_size>=2 and 0<=overlap<chunk_size")
    if embedding not in {"env", "hash"}:
        raise ValueError("embedding must be 'env' or 'hash'")
    if ragbot_mode not in {"vector", "lexical", "hybrid"}:
        raise ValueError("ragbot_mode must be vector, lexical or hybrid")

    names = _backend_names(backends)
    if "ragbot" in names and embedding == "hash" and len(names) > 1 and enforce_embedding_match:
        raise ValueError("Do not compare a live Ragbot semantic index with local HashEmbedder; use --embedding env")

    embedder = build_embedder() if embedding == "env" else HashEmbedder(hash_dimension)
    cases = [case for case in dataset.get("cases") or [] if _is_labeled(case)]
    if not cases:
        raise ValueError("Native comparison requires labeled Golden Dataset cases")

    backend_results: list[dict[str, Any]] = []
    for backend_name in names:
        backend = _make_backend(
            backend_name,
            embedder=embedder,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            server=server,
            tenant=tenant,
            user=user,
            api_key=api_key,
            timeout=timeout,
            ragbot_mode=ragbot_mode,
            rerank=rerank,
        )
        tracemalloc.start()
        build_stats = backend.build(units)
        scores: list[CaseScore] = []
        latencies_ms: list[float] = []
        runtime: dict[str, Any] = {}
        for case in cases:
            first_hits: Optional[list[RetrievedHit]] = None
            for _ in range(repetitions):
                started = time.perf_counter()
                hits, diagnostics = backend.search(str(case["query"]), top_k=top_k, filters=_case_filters(dataset, case))
                latencies_ms.append((time.perf_counter() - started) * 1000.0)
                if diagnostics:
                    runtime = diagnostics
                if first_hits is None:
                    first_hits = hits
            scores.append(score_case(case, first_hits or []))
        _current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        summary = summarize_scores(scores, latencies_ms)
        result = {
            "backend": backend_name,
            "build": build_stats,
            "summary": summary,
            "runtime": runtime,
            "peak_python_bytes": peak,
            "cases": [_score_dict(score) for score in scores],
        }
        if backend_name == "ragbot" and runtime:
            result["embedding_match"] = _embedding_match(runtime, embedder)
            if enforce_embedding_match and len(names) > 1 and not result["embedding_match"]["passed"]:
                raise RuntimeError(f"Embedding mismatch makes framework comparison invalid: {result['embedding_match']}")
        backend_results.append(result)

    baseline = next((item for item in backend_results if item["backend"] == "ragbot"), backend_results[0])
    baseline_summary = baseline["summary"]
    for result in backend_results:
        summary = result["summary"]
        result["delta_vs_baseline"] = {
            "hit_at_5": round(summary["hit_at_5"] - baseline_summary["hit_at_5"], 4),
            "mrr_at_10": round(summary["mrr_at_10"] - baseline_summary["mrr_at_10"], 4),
            "ndcg_at_10": round(summary["ndcg_at_10"] - baseline_summary["ndcg_at_10"], 4),
            "p95_ms": round(summary["query_latency_ms_p95"] - baseline_summary["query_latency_ms_p95"], 3),
        }

    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "methodology": {
            "kind": "native-retrieval-comparison",
            "controlled_variables": [
                "same local corpus for LangChain and LlamaIndex",
                "same Golden Dataset queries and relevance labels",
                "same embedding model for local framework adapters",
                "same top_k",
                "character-equivalent chunk budget",
            ],
            "ragbot_scope_note": (
                "Ragbot queries the live deployment. For an apples-to-apples run, the live tenant/filter scope "
                "must contain the same corpus represented by corpus_manifest."
            ),
            "attribution_note": (
                "This native benchmark changes splitter + vector-store/retriever implementation together. "
                "Use benchmarks.rag_framework_compare when you need splitter-only attribution."
            ),
        },
        "configuration": {
            "backends": names,
            "embedding": embedding,
            "embedding_model": embedder.model_name,
            "embedding_dimension": embedder.dimension,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "top_k": top_k,
            "repetitions": repetitions,
            "ragbot_mode": ragbot_mode,
            "rerank": rerank,
            "server": server if "ragbot" in names else None,
            "tenant": tenant if "ragbot" in names else None,
            "backend_versions": _backend_versions(names),
        },
        "corpus_manifest": corpus_manifest(units),
        "dataset": {"name": dataset.get("name"), "cases": len(cases)},
        "baseline": baseline["backend"],
        "results": backend_results,
    }


def _fmt_percent(value: Any) -> str:
    return "-" if value is None else f"{float(value) * 100:.1f}%"


def markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        f"# Native RAG framework comparison — {report.get('dataset', {}).get('name') or 'Golden Dataset'}",
        "",
        f"Generated: `{report['generated_at']}`  ",
        f"Baseline: `{report['baseline']}`  ",
        f"Corpus SHA256: `{report['corpus_manifest']['sha256']}`",
        "",
        "## Comparison",
        "",
        "| Backend | Hit@1 | Hit@5 | MRR@10 | P@5 | Recall@10 | nDCG@10 | p50 | p95 | Index | Chunks |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in report["results"]:
        summary = result["summary"]
        build = result["build"]
        lines.append(
            f"| `{result['backend']}` | {_fmt_percent(summary['hit_at_1'])} | {_fmt_percent(summary['hit_at_5'])} | "
            f"{summary['mrr_at_10']:.3f} | {_fmt_percent(summary['precision_at_5'])} | {_fmt_percent(summary['recall_at_10'])} | "
            f"{summary['ndcg_at_10']:.3f} | {summary['query_latency_ms_p50']:.2f} ms | {summary['query_latency_ms_p95']:.2f} ms | "
            f"{build.get('index_seconds', 0):.3f} s | {build.get('chunks') if build.get('chunks') is not None else '-'} |"
        )

    lines.extend(["", "## Delta vs baseline", "", "| Backend | Δ Hit@5 | Δ MRR@10 | Δ nDCG@10 | Δ p95 |", "| --- | ---: | ---: | ---: | ---: |"])
    for result in report["results"]:
        delta = result["delta_vs_baseline"]
        lines.append(
            f"| `{result['backend']}` | {delta['hit_at_5']:+.4f} | {delta['mrr_at_10']:+.4f} | "
            f"{delta['ndcg_at_10']:+.4f} | {delta['p95_ms']:+.2f} ms |"
        )

    lines.extend([
        "",
        "## Methodology guardrails",
        "",
        f"- {report['methodology']['ragbot_scope_note']}",
        f"- {report['methodology']['attribution_note']}",
        "- `HashEmbedder` is only for deterministic smoke tests. Use `--embedding env` for semantic conclusions.",
        "- Compare multiple repetitions and the same hardware/runtime before drawing latency conclusions.",
    ])
    return "\n".join(lines) + "\n"


def write_reports(report: Mapping[str, Any], output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    json_path = output_dir / f"native-framework-{stamp}.json"
    md_path = output_dir / f"native-framework-{stamp}.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(markdown_report(report), encoding="utf-8")
    (output_dir / "latest.json").write_text(json_path.read_text(encoding="utf-8"), encoding="utf-8")
    (output_dir / "latest.md").write_text(md_path.read_text(encoding="utf-8"), encoding="utf-8")
    return {"json": json_path, "markdown": md_path}


def synthetic_native_dataset(documents: int = 24, queries: int = 12) -> tuple[list[CorpusUnit], dict[str, Any]]:
    units = []
    cases = []
    selected = list(range(min(documents, queries)))
    if queries < documents and queries > 1:
        selected = sorted({round(i * (documents - 1) / (queries - 1)) for i in range(queries)})
    for index in range(documents):
        marker = f"RBNATIVE{index:05d}"
        text = (
            f"{marker}. Native retrieval note {index}. The unique retry budget is {3 + index % 11} and "
            f"the timeout is {20 + index * 7} milliseconds. This paragraph exists to test semantic retrieval.\n"
        ) * 6
        units.append(CorpusUnit(doc_id=f"synthetic-{index:05d}.txt", path=f"synthetic-{index:05d}.txt", text=text))
    for index in selected:
        cases.append(
            {
                "id": f"native-{index:05d}",
                "category": "synthetic",
                "query": f"RBNATIVE{index:05d} retry budget timeout",
                "relevance": {"doc_ids": [f"synthetic-{index:05d}.txt"], "max_rank": 5},
            }
        )
    return units, {"schema_version": 1, "name": "native synthetic smoke", "cases": cases}


def parse_args(argv: Optional[Iterable[str]] = None):
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backends", default="langchain,llamaindex")
    parser.add_argument("--corpus-dir")
    parser.add_argument("--golden")
    parser.add_argument("--embedding", choices=["hash", "env"], default="hash")
    parser.add_argument("--hash-dimension", type=int, default=256)
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--chunk-overlap", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--server", default=DEFAULT_SERVER)
    parser.add_argument("--tenant", default="default")
    parser.add_argument("--user", default="rag-benchmark")
    parser.add_argument("--api-key", default=os.getenv("RAGBOT_API_KEY"))
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--ragbot-mode", choices=["vector", "lexical", "hybrid"], default="vector")
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument("--no-enforce-embedding-match", action="store_true")
    parser.add_argument("--synthetic-documents", type=int, default=24)
    parser.add_argument("--synthetic-queries", type=int, default=12)
    parser.add_argument("--output-dir", default="reports/rag-benchmark/native")
    args = parser.parse_args(argv)
    if bool(args.corpus_dir) != bool(args.golden):
        parser.error("--corpus-dir and --golden must be supplied together")
    return args


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    if args.corpus_dir:
        units = load_corpus_units(Path(args.corpus_dir).resolve())
        dataset = load_golden_dataset(Path(args.golden).resolve())
    else:
        units, dataset = synthetic_native_dataset(args.synthetic_documents, args.synthetic_queries)
        if "ragbot" in _backend_names(args.backends):
            raise SystemExit("Synthetic mode cannot include live Ragbot; use --corpus-dir/--golden for three-way comparison")
    report = run_comparison(
        dataset=dataset,
        units=units,
        backends=args.backends,
        embedding=args.embedding,
        hash_dimension=args.hash_dimension,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        top_k=args.top_k,
        repetitions=args.repetitions,
        server=args.server,
        tenant=args.tenant,
        user=args.user,
        api_key=args.api_key,
        timeout=args.timeout,
        ragbot_mode=args.ragbot_mode,
        rerank=args.rerank,
        enforce_embedding_match=not args.no_enforce_embedding_match,
    )
    paths = write_reports(report, Path(args.output_dir).resolve())
    print(markdown_report(report))
    print(f"JSON: {paths['json']}")
    print(f"Markdown: {paths['markdown']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
