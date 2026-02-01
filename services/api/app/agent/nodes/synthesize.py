from __future__ import annotations

import re
from typing import Any, Iterable, List, Sequence

from ..state import AgentState, Draft, EvidenceItem, Citation


def synthesize_node(state: AgentState, services: Any) -> AgentState:
    if not state.evidence:
        state.draft = Draft(answer_text="未找到可用证据。")
        return state

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
    seen = set()
    ordered: List[Citation] = []
    for ev in evidence:
        for cite in ev.citations:
            key = (
                cite.kind,
                cite.chunk_id,
                cite.doc_id,
                cite.page,
                cite.section,
                cite.url,
                cite.path,
                cite.ref,
                cite.line_start,
                cite.line_end,
                cite.row_ref,
                cite.title,
            )
            if key in seen:
                continue
            seen.add(key)
            ordered.append(cite)
    return ordered


def _dedupe_citations(citations: List[Citation]) -> List[Citation]:
    return _merge_citations([EvidenceItem(kind="doc_chunk", citations=citations)])

