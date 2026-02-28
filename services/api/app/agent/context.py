"""Context strategy: IDE context injection, evidence dedup, and compression.

Processes client_context from ChatRequest to enrich constraints and
provide initial evidence. Also handles evidence dedup and compression
to control token costs.
"""
from __future__ import annotations

import hashlib
import logging
from typing import Any, Dict, List, Optional

from contracts.types import Citation, Constraints, EvidenceItem

logger = logging.getLogger(__name__)

# Maximum total evidence text length (characters) before compression kicks in
MAX_EVIDENCE_CHARS = 12000
# Maximum length for a single evidence item
MAX_SINGLE_EVIDENCE_CHARS = 3000


def process_client_context(
    client_context: Optional[Dict[str, Any]],
    constraints: Optional[Constraints] = None,
) -> tuple[Optional[Constraints], List[EvidenceItem]]:
    """Convert IDE client_context into enriched constraints and initial evidence.

    Supports fields:
    - open_files: List[{path, content, language}] — currently open files
    - selected_text: {path, content, start_line, end_line} — selected code
    - git_diff: str — current git diff
    - recent_errors: List[str] — recent error logs
    - workspace_root: str — workspace root path
    - repo: str — repo name
    - ref: str — git ref
    """
    if not client_context:
        return constraints, []

    c = constraints or Constraints()
    evidence: List[EvidenceItem] = []

    # Enrich constraints from context
    if client_context.get("repo") and not c.repo:
        c.repo = client_context["repo"]
    if client_context.get("ref") and not c.ref:
        c.ref = client_context["ref"]
    if client_context.get("path_prefix") and not c.path_prefix:
        c.path_prefix = client_context["path_prefix"]

    # Process selected text as initial evidence
    selected = client_context.get("selected_text")
    if selected and isinstance(selected, dict) and selected.get("content"):
        path = selected.get("path", "unknown")
        start_line = selected.get("start_line")
        end_line = selected.get("end_line")
        evidence.append(EvidenceItem(
            kind="file_content",
            score=1.0,
            text=selected["content"][:MAX_SINGLE_EVIDENCE_CHARS],
            citations=[Citation(
                kind="code", path=path,
                line_start=start_line, line_end=end_line,
            )],
            metadata={"source": "client_context", "path": path},
        ))

    # Process open files as context (lower priority)
    open_files = client_context.get("open_files")
    if open_files and isinstance(open_files, list):
        for f in open_files[:5]:  # Limit to 5 files
            if not isinstance(f, dict) or not f.get("content"):
                continue
            content = f["content"][:MAX_SINGLE_EVIDENCE_CHARS]
            path = f.get("path", "unknown")
            evidence.append(EvidenceItem(
                kind="file_content",
                score=0.5,  # Lower priority than agent-retrieved evidence
                text=content,
                citations=[Citation(kind="code", path=path)],
                metadata={"source": "client_context", "path": path,
                           "language": f.get("language", "unknown")},
            ))

    # Process git diff
    git_diff = client_context.get("git_diff")
    if git_diff and isinstance(git_diff, str):
        evidence.append(EvidenceItem(
            kind="patch",
            score=0.7,
            text=git_diff[:MAX_SINGLE_EVIDENCE_CHARS],
            citations=[],
            metadata={"source": "client_context", "type": "git_diff"},
        ))

    # Process recent errors
    recent_errors = client_context.get("recent_errors")
    if recent_errors and isinstance(recent_errors, list):
        error_text = "\n---\n".join(str(e) for e in recent_errors[:3])
        evidence.append(EvidenceItem(
            kind="error_analysis",
            score=0.8,
            text=error_text[:MAX_SINGLE_EVIDENCE_CHARS],
            citations=[],
            metadata={"source": "client_context", "error_count": len(recent_errors)},
        ))

    return c, evidence


def dedup_evidence(evidence: List[EvidenceItem]) -> List[EvidenceItem]:
    """Remove duplicate evidence items based on text content hash."""
    seen_hashes: set[str] = set()
    deduped: List[EvidenceItem] = []

    for item in evidence:
        text_hash = hashlib.md5(item.text.encode("utf-8")).hexdigest()
        if text_hash in seen_hashes:
            logger.debug("Dedup: skipping duplicate evidence (kind=%s)", item.kind)
            continue
        seen_hashes.add(text_hash)
        deduped.append(item)

    if len(deduped) < len(evidence):
        logger.info("Evidence dedup: %d -> %d items", len(evidence), len(deduped))
    return deduped


def compress_evidence(evidence: List[EvidenceItem], max_total: int = MAX_EVIDENCE_CHARS) -> List[EvidenceItem]:
    """Compress evidence to fit within token budget.

    Strategy:
    1. Sort by score (highest first)
    2. Truncate individual items if too long
    3. Drop lowest-score items if total exceeds budget
    """
    if not evidence:
        return evidence

    # Sort by score descending, preserving order for equal scores
    sorted_ev = sorted(evidence, key=lambda e: -e.score)

    total_chars = 0
    result: List[EvidenceItem] = []

    for item in sorted_ev:
        text = item.text
        # Truncate individual item if too long
        if len(text) > MAX_SINGLE_EVIDENCE_CHARS:
            text = text[:MAX_SINGLE_EVIDENCE_CHARS] + "\n... [truncated]"
            item = EvidenceItem(
                kind=item.kind, score=item.score, text=text,
                citations=item.citations, metadata=item.metadata,
            )

        if total_chars + len(text) > max_total and result:
            logger.info("Evidence compression: dropping item (kind=%s, score=%.2f) to fit budget",
                        item.kind, item.score)
            continue

        total_chars += len(text)
        result.append(item)

    if len(result) < len(evidence):
        logger.info("Evidence compressed: %d -> %d items, %d chars",
                     len(evidence), len(result), total_chars)
    return result
