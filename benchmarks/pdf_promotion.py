"""Control/candidate promotion gate for Ragbot PDF ingestion.

The gate validates two things independently:

1. the Golden Dataset / corpus is production-grade enough to justify a default change;
2. the vNext candidate does not regress retrieval, fidelity or resource budgets.

The current default control is PyPDF2 + raw page blocks + Ragbot fixed-window
chunking. The current vNext candidate is PyMuPDF + block coalescing + LlamaIndex
sentence splitting. Both pipelines use the production parser bridge and the same
embedding/cosine scorer as the component and factorial benchmarks.
"""
from __future__ import annotations

import hashlib
import statistics
import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from benchmarks.component_compare import (
    RawPdf,
    _boundary_ending,
    _cosine_search,
    _label_text_coverage,
    _normalize,
    _percentile,
)
from benchmarks.factorial_compare import PreparedParser, prepare_parser_backend, splitter_config
from benchmarks.rag_native_compare import RetrievedHit, audit_golden_dataset, score_case, summarize_scores
from services.worker.parsing import coalesce_document_blocks, iter_document_segments


RECOMMENDED_DOCUMENT_TYPES = (
    "technical_book",
    "multi_column_paper",
    "datasheet",
    "table_report",
    "figure_report",
    "cjk_document",
    "mixed_language_document",
    "scanned_document",
)
RECOMMENDED_CASE_TAGS = (
    "numeric",
    "page-specific",
    "paraphrase",
    "table",
    "figure-caption",
    "cross-lingual",
)


@dataclass(frozen=True)
class PipelineSpec:
    name: str
    parser: str
    splitter: str
    coalescing: str
    coalesce_target_multiplier: float = 4.0


CONTROL_PIPELINE = PipelineSpec(
    name="control",
    parser="pypdf2",
    splitter="ragbot",
    coalescing="raw",
)
CANDIDATE_PIPELINE = PipelineSpec(
    name="candidate",
    parser="pymupdf",
    splitter="llamaindex",
    coalescing="coalesced",
    coalesce_target_multiplier=4.0,
)


@dataclass(frozen=True)
class PromotionThresholds:
    hit_at_5_delta_min: float = 0.0
    mrr_delta_min: float = -0.02
    ndcg_delta_min: float = -0.02
    category_mrr_delta_min: float = -0.05
    label_text_coverage_min: float = 0.99
    page_metadata_delta_min: float = -0.005
    chunk_ratio_max: float = 1.15
    embedding_time_ratio_max: float = 1.15
    memory_ratio_max: float = 1.25
    non_boundary_ratio_max: float = 0.50
    max_new_case_failures: int = 0


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _ratio(candidate: float, control: float) -> float:
    if control <= 0:
        return 1.0 if candidate <= 0 else float("inf")
    return candidate / control


def _case_tags(case: Mapping[str, Any]) -> list[str]:
    raw = case.get("tags") or []
    if isinstance(raw, str):
        raw = [raw]
    return sorted({str(item).strip().lower() for item in raw if str(item).strip()})


def _review_approved(case: Mapping[str, Any]) -> bool:
    review = case.get("review") or {}
    return isinstance(review, Mapping) and str(review.get("status") or "").strip().lower() == "approved"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def audit_pdf_structure_suite(
    dataset: Mapping[str, Any],
    documents: Sequence[RawPdf],
    *,
    profile: str = "production",
) -> dict[str, Any]:
    """Audit dataset quality plus PDF-structure coverage and corpus identity."""
    base = audit_golden_dataset(dataset, profile=profile)
    suite = dataset.get("pdf_structure_suite") or {}
    cases = list(dataset.get("cases") or [])
    checks: list[dict[str, Any]] = []
    warnings: list[str] = []

    def add(name: str, actual: Any, expected: str, passed: bool) -> None:
        checks.append({"name": name, "actual": actual, "expected": expected, "passed": bool(passed)})

    if not isinstance(suite, Mapping) or not suite:
        if profile == "production":
            add("suite_manifest", False, "present", False)
        else:
            warnings.append("pdf_structure_suite manifest is absent; development smoke only")
        return {
            "profile": profile,
            "passed": bool(base.get("passed")) and profile != "production",
            "golden_dataset": base,
            "checks": checks,
            "warnings": warnings,
            "stats": {"documents": len(documents), "document_types": [], "case_tags": []},
        }

    manifest_docs = suite.get("documents") or []
    requirements = suite.get("requirements") or {}
    if not isinstance(manifest_docs, list):
        manifest_docs = []
    if not isinstance(requirements, Mapping):
        requirements = {}

    actual_by_path = {document.path: document for document in documents}
    declared_paths: set[str] = set()
    document_types: set[str] = set()
    file_identity_ok = True
    for item in manifest_docs:
        if not isinstance(item, Mapping):
            file_identity_ok = False
            continue
        path = str(item.get("path") or "").strip()
        document_type = str(item.get("document_type") or "").strip().lower()
        if path:
            declared_paths.add(path)
        if document_type:
            document_types.add(document_type)
        actual = actual_by_path.get(path)
        if actual is None:
            file_identity_ok = False
            continue
        expected_sha = str(item.get("sha256") or "").strip().lower()
        if expected_sha and expected_sha != _sha256(actual.data):
            file_identity_ok = False

    case_tags = sorted({tag for case in cases for tag in _case_tags(case)})
    approved = sum(1 for case in cases if _review_approved(case))
    reviewed_rate = approved / len(cases) if cases else 0.0

    min_documents = int(requirements.get("min_documents") or (8 if profile == "production" else 1))
    min_reviewed_rate = float(requirements.get("min_reviewed_rate") or (0.80 if profile == "production" else 0.0))
    required_document_types = {
        str(item).strip().lower()
        for item in requirements.get("required_document_types") or []
        if str(item).strip()
    }
    required_case_tags = {
        str(item).strip().lower()
        for item in requirements.get("required_case_tags") or []
        if str(item).strip()
    }

    add("suite_manifest", True, "present", True)
    add("document_count", len(manifest_docs), f">={min_documents}", len(manifest_docs) >= min_documents)
    add("corpus_paths_match_manifest", sorted(actual_by_path), "exact manifest coverage", declared_paths == set(actual_by_path))
    add("corpus_file_identity", file_identity_ok, "all declared files present and sha256 matches when pinned", file_identity_ok)
    add(
        "required_document_types",
        sorted(document_types),
        f"contains {sorted(required_document_types)}",
        required_document_types.issubset(document_types),
    )
    add(
        "required_case_tags",
        case_tags,
        f"contains {sorted(required_case_tags)}",
        required_case_tags.issubset(set(case_tags)),
    )
    add("approved_review_rate", round(reviewed_rate, 4), f">={min_reviewed_rate:.2f}", reviewed_rate >= min_reviewed_rate)

    return {
        "profile": profile,
        "passed": bool(base.get("passed")) and all(item["passed"] for item in checks),
        "golden_dataset": base,
        "checks": checks,
        "warnings": warnings,
        "stats": {
            "documents": len(manifest_docs),
            "document_types": sorted(document_types),
            "case_tags": case_tags,
            "approved_review_rate": round(reviewed_rate, 4),
        },
    }


def build_suite_skeleton(documents: Sequence[RawPdf], *, name: str = "PDF Structure Golden Suite") -> dict[str, Any]:
    """Create an authoring skeleton without inventing questions or relevance labels."""
    return {
        "schema_version": 1,
        "name": name,
        "pdf_structure_suite": {
            "version": 1,
            "requirements": {
                "min_documents": 8,
                "min_reviewed_rate": 0.80,
                "required_document_types": list(RECOMMENDED_DOCUMENT_TYPES),
                "required_case_tags": list(RECOMMENDED_CASE_TAGS),
            },
            "documents": [
                {
                    "path": document.path,
                    "sha256": _sha256(document.data),
                    "page_count": document.page_count,
                    "document_type": "TODO",
                    "languages": [],
                }
                for document in documents
            ],
        },
        "cases": [],
    }


def _parser_fidelity(prepared: PreparedParser, cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    blocks = [block for document in prepared.documents for block in document.normalized.blocks]
    expected_pages = sum(document.page_count for document in prepared.documents)
    extracted_pages = {
        (document.doc_id, int(block.page))
        for document in prepared.documents
        for block in document.normalized.blocks
        if block.page is not None
    }
    text = "\n".join(document.normalized.text for document in prepared.documents)
    characters = sum(len(document.normalized.text) for document in prepared.documents)
    count = len(blocks)
    return {
        "characters_extracted": characters,
        "page_metadata_coverage": round(len(extracted_pages) / max(expected_pages, 1), 6),
        "label_text_coverage": _label_text_coverage(cases, text),
        "bbox_block_rate": round(sum(block.bbox is not None for block in blocks) / max(count, 1), 6),
        "table_block_rate": round(sum("table" in block.kind.casefold() for block in blocks) / max(count, 1), 6),
        "section_block_rate": round(sum(bool(block.section) for block in blocks) / max(count, 1), 6),
    }


def _evaluate_with_details(
    cases: Sequence[Mapping[str, Any]],
    *,
    hits: Sequence[RetrievedHit],
    vectors: Sequence[Sequence[float]],
    embedder: Any,
    top_k: int,
    repetitions: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    latencies_ms: list[float] = []
    scores = []
    details: list[dict[str, Any]] = []
    for case in cases:
        first_hits: list[RetrievedHit] | None = None
        case_latencies: list[float] = []
        for _ in range(max(1, repetitions)):
            started = time.perf_counter()
            current = _cosine_search(
                str(case["query"]),
                embedder=embedder,
                vectors=vectors,
                hits=hits,
                top_k=top_k,
            )
            elapsed = (time.perf_counter() - started) * 1000.0
            latencies_ms.append(elapsed)
            case_latencies.append(elapsed)
            if first_hits is None:
                first_hits = current
        score = score_case(case, first_hits or [])
        scores.append(score)
        details.append(
            {
                "id": score.case_id,
                "category": score.category,
                "tags": _case_tags(case),
                "query": score.query,
                "retrieval_pass": score.retrieval_pass,
                "first_relevant_rank": score.first_relevant_rank,
                "mrr_at_10": round(score.reciprocal_rank_at_10, 4),
                "ndcg_at_10": round(score.ndcg_at_10, 4),
                "latency_ms_p50": round(_percentile(case_latencies, 0.50), 3),
                "top_hits": [
                    {
                        "doc_id": hit.doc_id,
                        "page": hit.page,
                        "score": round(float(hit.score), 6) if hit.score is not None else None,
                        "text": hit.text[:240],
                    }
                    for hit in (first_hits or [])[:3]
                ],
            }
        )
    return summarize_scores(scores, latencies_ms), details


def run_pipeline(
    prepared: PreparedParser,
    spec: PipelineSpec,
    cases: Sequence[Mapping[str, Any]],
    *,
    embedder: Any,
    chunk_size: int,
    chunk_overlap: int,
    top_k: int,
    repetitions: int,
) -> dict[str, Any]:
    """Run one promotion pipeline through the production parser bridge."""
    coalesced = spec.coalescing == "coalesced"
    if spec.coalescing not in {"raw", "coalesced"}:
        raise ValueError("pipeline coalescing must be raw or coalesced")
    config = splitter_config(
        spec.splitter,
        coalesced=coalesced,
        chunk_size=chunk_size,
        coalesce_target_multiplier=spec.coalesce_target_multiplier,
    )

    tracemalloc.start()
    raw_blocks = sum(len(document.normalized.blocks) for document in prepared.documents)
    bridge_blocks = 0
    segment_started = time.perf_counter()
    hits: list[RetrievedHit] = []
    for document in prepared.documents:
        if coalesced:
            target = int(config["block_coalescing"]["target_chars"])
            bridge_blocks += len(
                coalesce_document_blocks(
                    document.normalized,
                    target_chars=target,
                    respect_page=True,
                    respect_section=True,
                )
            )
        else:
            bridge_blocks += len(document.normalized.blocks)
        for index, segment in enumerate(
            iter_document_segments(
                document.normalized,
                config,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        ):
            block_metadata = dict(segment.metadata.get("block_metadata") or {})
            hits.append(
                RetrievedHit(
                    chunk_id=f"{document.doc_id}::{spec.name}::{index}",
                    doc_id=document.doc_id,
                    path=document.path,
                    text=segment.text,
                    page=segment.page,
                    metadata={
                        "pipeline": spec.name,
                        "parser": spec.parser,
                        "splitter": spec.splitter,
                        "coalescing": spec.coalescing,
                        "section": segment.section,
                        "source_block_count": int(block_metadata.get("source_block_count") or 1),
                    },
                )
            )
    segment_seconds = time.perf_counter() - segment_started
    if not hits:
        raise RuntimeError(f"Pipeline {spec.name} produced zero chunks")

    embed_started = time.perf_counter()
    vectors = [_normalize(vector) for vector in embedder.embed_batch([hit.text for hit in hits])]
    embedding_seconds = time.perf_counter() - embed_started
    quality, case_details = _evaluate_with_details(
        cases,
        hits=hits,
        vectors=vectors,
        embedder=embedder,
        top_k=top_k,
        repetitions=repetitions,
    )
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    lengths = [len(hit.text) for hit in hits]
    source_chars = sum(len(document.normalized.text) for document in prepared.documents)
    chunk_chars = sum(lengths)
    return {
        "name": spec.name,
        "spec": {
            "parser": spec.parser,
            "splitter": spec.splitter,
            "coalescing": spec.coalescing,
            "coalesce_target_multiplier": spec.coalesce_target_multiplier if coalesced else None,
        },
        "documents": len(prepared.documents),
        "raw_blocks": raw_blocks,
        "bridge_blocks": bridge_blocks,
        "chunks": len(hits),
        "quality": quality,
        "cases": case_details,
        "timing": {
            "parser_seconds": round(prepared.parser_seconds, 6),
            "segment_seconds": round(segment_seconds, 6),
            "embedding_seconds": round(embedding_seconds, 6),
        },
        "chunk_shape": {
            "chars_mean": round(statistics.fmean(lengths), 3),
            "chars_p50": round(_percentile(lengths, 0.50), 3),
            "chars_p95": round(_percentile(lengths, 0.95), 3),
            "non_boundary_end_rate": round(sum(1 for hit in hits if not _boundary_ending(hit.text)) / len(hits), 6),
            "character_inflation_ratio": round(chunk_chars / max(source_chars, 1), 6),
        },
        "fidelity": _parser_fidelity(prepared, cases),
        "memory": {"tracemalloc_peak_bytes": peak},
    }


def prepare_pipelines(documents: Sequence[RawPdf]) -> tuple[PreparedParser, PreparedParser]:
    return prepare_parser_backend(CONTROL_PIPELINE.parser, documents), prepare_parser_backend(CANDIDATE_PIPELINE.parser, documents)


def evaluate_promotion_gate(
    control: Mapping[str, Any],
    candidate: Mapping[str, Any],
    suite_audit: Mapping[str, Any],
    *,
    thresholds: PromotionThresholds | None = None,
) -> dict[str, Any]:
    """Compare candidate against control and return machine-readable promotion checks."""
    limits = thresholds or PromotionThresholds()
    cq = control.get("quality") or {}
    nq = candidate.get("quality") or {}
    cf = control.get("fidelity") or {}
    nf = candidate.get("fidelity") or {}
    cm = control.get("memory") or {}
    nm = candidate.get("memory") or {}
    ct = control.get("timing") or {}
    nt = candidate.get("timing") or {}
    cs = control.get("chunk_shape") or {}
    ns = candidate.get("chunk_shape") or {}
    checks: list[dict[str, Any]] = []

    def add(name: str, actual: Any, expected: str, passed: bool) -> None:
        checks.append({"name": name, "actual": actual, "expected": expected, "passed": bool(passed)})

    add("suite_audit", bool(suite_audit.get("passed")), "pass", bool(suite_audit.get("passed")))

    hit_delta = _as_float(nq.get("hit_at_5")) - _as_float(cq.get("hit_at_5"))
    mrr_delta = _as_float(nq.get("mrr_at_10")) - _as_float(cq.get("mrr_at_10"))
    ndcg_delta = _as_float(nq.get("ndcg_at_10")) - _as_float(cq.get("ndcg_at_10"))
    add("hit_at_5_delta", round(hit_delta, 4), f">={limits.hit_at_5_delta_min:.3f}", hit_delta >= limits.hit_at_5_delta_min)
    add("mrr_at_10_delta", round(mrr_delta, 4), f">={limits.mrr_delta_min:.3f}", mrr_delta >= limits.mrr_delta_min)
    add("ndcg_at_10_delta", round(ndcg_delta, 4), f">={limits.ndcg_delta_min:.3f}", ndcg_delta >= limits.ndcg_delta_min)

    label_coverage = _as_float(nf.get("label_text_coverage"))
    add("candidate_label_text_coverage", label_coverage, f">={limits.label_text_coverage_min:.3f}", label_coverage >= limits.label_text_coverage_min)
    page_delta = _as_float(nf.get("page_metadata_coverage")) - _as_float(cf.get("page_metadata_coverage"))
    add("page_metadata_delta", round(page_delta, 4), f">={limits.page_metadata_delta_min:.3f}", page_delta >= limits.page_metadata_delta_min)

    chunk_ratio = _ratio(float(candidate.get("chunks") or 0), float(control.get("chunks") or 0))
    embed_ratio = _ratio(_as_float(nt.get("embedding_seconds")), _as_float(ct.get("embedding_seconds")))
    memory_ratio = _ratio(float(nm.get("tracemalloc_peak_bytes") or 0), float(cm.get("tracemalloc_peak_bytes") or 0))
    boundary_ratio = _ratio(_as_float(ns.get("non_boundary_end_rate")), _as_float(cs.get("non_boundary_end_rate")))
    add("chunk_ratio", round(chunk_ratio, 4), f"<={limits.chunk_ratio_max:.2f}", chunk_ratio <= limits.chunk_ratio_max)
    add("embedding_time_ratio", round(embed_ratio, 4), f"<={limits.embedding_time_ratio_max:.2f}", embed_ratio <= limits.embedding_time_ratio_max)
    add("memory_ratio", round(memory_ratio, 4), f"<={limits.memory_ratio_max:.2f}", memory_ratio <= limits.memory_ratio_max)
    add("non_boundary_ratio", round(boundary_ratio, 4), f"<={limits.non_boundary_ratio_max:.2f}", boundary_ratio <= limits.non_boundary_ratio_max)

    control_cases = {str(item.get("id")): item for item in control.get("cases") or []}
    candidate_cases = {str(item.get("id")): item for item in candidate.get("cases") or []}
    new_failures = []
    rank_regressions = []
    for case_id, base in control_cases.items():
        current = candidate_cases.get(case_id)
        if current is None:
            new_failures.append(case_id)
            continue
        if bool(base.get("retrieval_pass")) and not bool(current.get("retrieval_pass")):
            new_failures.append(case_id)
        base_rank = base.get("first_relevant_rank")
        current_rank = current.get("first_relevant_rank")
        if base_rank is not None and (current_rank is None or int(current_rank) > int(base_rank)):
            rank_regressions.append(
                {
                    "id": case_id,
                    "category": current.get("category") or base.get("category"),
                    "tags": current.get("tags") or base.get("tags") or [],
                    "control_rank": base_rank,
                    "candidate_rank": current_rank,
                }
            )
    add("new_case_failures", len(new_failures), f"<={limits.max_new_case_failures}", len(new_failures) <= limits.max_new_case_failures)

    category_regressions: list[dict[str, Any]] = []
    control_categories = cq.get("categories") or {}
    candidate_categories = nq.get("categories") or {}
    for category, base in control_categories.items():
        current = candidate_categories.get(category) or {}
        delta = _as_float(current.get("mrr_at_10")) - _as_float((base or {}).get("mrr_at_10"))
        if delta < limits.category_mrr_delta_min:
            category_regressions.append({"category": category, "mrr_delta": round(delta, 4)})
    add("category_mrr_regressions", len(category_regressions), "=0", not category_regressions)

    return {
        "passed": all(item["passed"] for item in checks),
        "checks": checks,
        "diagnostics": {
            "new_case_failures": new_failures,
            "rank_regressions": rank_regressions,
            "category_regressions": category_regressions,
        },
        "thresholds": limits.__dict__,
    }
