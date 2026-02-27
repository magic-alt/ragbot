from __future__ import annotations

import logging
import re
from typing import Any, Iterable, List, Sequence, Tuple

from ..state import AgentState, Draft, EvidenceItem, Citation

logger = logging.getLogger(__name__)


def synthesize_node(state: AgentState, services: Any) -> AgentState:
    if not state.evidence:
        state.draft = Draft(answer_text="未找到可用证据。")
        return state

    llm = getattr(services, "llm", None)
    if llm and getattr(llm, "enabled", False):
        try:
            claims, used_citations, insufficient, missing = _llm_build_claims(state, llm)
            claim_lines = _render_claim_lines(claims)
            answer_text = " ".join(claim_lines).strip()
            if insufficient and not claim_lines:
                answer_text = "证据不足，无法给出可靠结论。"
            state.draft = Draft(
                answer_outline=claim_lines,
                answer_text=answer_text,
                used_citations=used_citations,
            )
            return state
        except Exception as exc:
            logger.warning("LLM synthesis failed: %s: %s", type(exc).__name__, exc)

    doc_evidence = [ev for ev in state.evidence if ev.kind == "doc_chunk"]
    if doc_evidence:
        summary = _summarize_chunks(state.query, doc_evidence)
        citations = _merge_citations(doc_evidence)
        state.draft = Draft(answer_text=summary, used_citations=citations)
        return state

    lines: List[str] = []
    citations: List[Citation] = []
    for evidence in state.evidence:
        lines.append(_summarize_evidence(evidence))
        citations.extend(evidence.citations)
    state.draft = Draft(answer_text=" ".join(lines).strip(), used_citations=_dedupe_citations(citations))
    return state


def _llm_build_claims(
    state: AgentState,
    llm: Any,
) -> Tuple[List[dict], List[Citation], bool, List[str]]:
    pack, citation_map = _build_evidence_pack(state.evidence)
    system = (
        "You are a RAG assistant. Use only the provided evidence. "
        "Every claim must cite at least one citation id. "
        "Return JSON only."
    )
    user = (
        "Question:\n"
        f"{state.query}\n\n"
        "Evidence (use these citation ids only):\n"
        f"{pack}\n\n"
        "Return JSON with fields: claims (array of {text, citation_ids}), "
        "insufficient (boolean), missing (array of strings)."
    )
    schema = {
        "type": "object",
        "properties": {
            "claims": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "citation_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 1,
                        },
                    },
                    "required": ["text", "citation_ids"],
                    "additionalProperties": False,
                },
            },
            "insufficient": {"type": "boolean"},
            "missing": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["claims", "insufficient", "missing"],
        "additionalProperties": False,
    }
    result = llm.chat_json(system=system, user=user, schema=schema)
    claims = result.get("claims", []) or []
    insufficient = bool(result.get("insufficient"))
    missing = result.get("missing", []) or []
    used_citations = _collect_citations(claims, citation_map)
    return claims, used_citations, insufficient, missing


def _render_claim_lines(claims: Sequence[dict]) -> List[str]:
    lines: List[str] = []
    for claim in claims:
        text = claim.get("text", "").strip()
        citation_ids = claim.get("citation_ids", [])
        if not text:
            continue
        if citation_ids:
            cite = ", ".join(citation_ids)
            lines.append(f"{text} (cite: {cite})")
        else:
            lines.append(text)
    return lines


def _collect_citations(claims: Sequence[dict], citation_map: dict) -> List[Citation]:
    seen: set[Citation] = set()
    used: List[Citation] = []
    for claim in claims:
        for cid in claim.get("citation_ids", []) or []:
            cite = citation_map.get(cid)
            if not cite or cite in seen:
                continue
            seen.add(cite)
            used.append(cite)
    return used


def _build_evidence_pack(evidence: Sequence[EvidenceItem]) -> Tuple[str, dict]:
    lines: List[str] = []
    citation_map: dict = {}
    cite_counter = 1
    for idx, ev in enumerate(evidence, start=1):
        lines.append(f"Evidence {idx} ({ev.kind}): {ev.text}")
        if ev.citations:
            cite_ids: List[str] = []
            for cite in ev.citations:
                cid = f"c{cite_counter}"
                cite_counter += 1
                citation_map[cid] = cite
                cite_ids.append(cid)
            lines.append("Citations: " + ", ".join(cite_ids))
    return "\n".join(lines), citation_map


def _summarize_evidence(evidence: EvidenceItem) -> str:
    if evidence.kind == "sql_rows":
        row_count = evidence.metadata.get("row_count") if evidence.metadata else None
        if row_count is not None:
            return f"SQL 返回 {row_count} 行。"
        return evidence.text or "SQL 返回结果。"
    if evidence.kind == "code_snippets":
        return "已检索到相关代码片段。"
    if evidence.kind == "web_snippets":
        return "已检索到相关网页片段。"
    return evidence.text or "证据片段。"


def _summarize_chunks(query: str, evidence: Sequence[EvidenceItem]) -> str:
    texts = [ev.text for ev in evidence if ev.text]
    sentences = _extract_sentences(" ".join(texts))
    keywords = _keywords(query)
    ranked = sorted(sentences, key=lambda s: _sentence_score(s, keywords), reverse=True)
    picked = []
    for sent in ranked:
        clean = sent.strip()
        if not clean or clean in picked:
            continue
        picked.append(clean)
        if len(picked) >= 3:
            break
    if not picked:
        fallback = " ".join(texts).strip()
        if fallback:
            picked = [fallback[:200]]
        else:
            picked = ["未能从文档中提取到有效摘要。"]
    return "根据检索内容，总结如下：" + " ".join(picked)


def _extract_sentences(text: str) -> List[str]:
    parts = re.split(r"[。！？!?\.]+", text)
    return [part.strip() for part in parts if part.strip()]


def _keywords(text: str) -> List[str]:
    tokens = re.findall(r"[A-Za-z0-9_\-]+|[\u4e00-\u9fff]+", text.lower())
    return [tok for tok in tokens if len(tok) > 1]


def _sentence_score(sentence: str, keywords: Iterable[str]) -> int:
    lower = sentence.lower()
    score = 0
    for kw in keywords:
        if kw in lower:
            score += 2
    score += min(len(sentence) // 80, 2)
    return score


def _merge_citations(evidence: Sequence[EvidenceItem]) -> List[Citation]:
    seen: set[Citation] = set()
    ordered: List[Citation] = []
    for ev in evidence:
        for cite in ev.citations:
            if cite in seen:
                continue
            seen.add(cite)
            ordered.append(cite)
    return ordered


def _dedupe_citations(citations: List[Citation]) -> List[Citation]:
    seen: set[Citation] = set()
    result: List[Citation] = []
    for cite in citations:
        if cite in seen:
            continue
        seen.add(cite)
        result.append(cite)
    return result

