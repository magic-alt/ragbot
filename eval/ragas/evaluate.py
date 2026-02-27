"""RAGAS-style evaluation for ragbot retrieval quality.

This module provides a lightweight evaluation framework that measures:
- **Faithfulness**: Does the answer only use facts from the retrieved context?
- **Answer Relevancy**: Does the answer address the question?
- **Context Precision**: Are the retrieved chunks relevant to the question?

Usage::

    from eval.ragas.evaluate import evaluate_dataset
    results = evaluate_dataset("eval/datasets/sample.jsonl")
    print(results)
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class EvalSample:
    question: str
    ground_truth: str
    contexts: List[str] = field(default_factory=list)
    answer: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalResult:
    sample: EvalSample
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    context_precision: float = 0.0

    @property
    def overall(self) -> float:
        return (self.faithfulness + self.answer_relevancy + self.context_precision) / 3.0


def load_dataset(path: str) -> List[EvalSample]:
    """Load evaluation samples from a JSONL file.

    Each line should be a JSON object with keys:
    ``question``, ``ground_truth``, and optionally ``contexts``.
    """
    samples: List[EvalSample] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            samples.append(
                EvalSample(
                    question=data["question"],
                    ground_truth=data["ground_truth"],
                    contexts=data.get("contexts", []),
                    metadata=data.get("metadata", {}),
                )
            )
    return samples


def evaluate_sample(sample: EvalSample) -> EvalResult:
    """Evaluate a single sample using keyword-overlap heuristics.

    For production use, replace these heuristics with LLM-based scoring
    (e.g., using OpenAI or a dedicated evaluation model).
    """
    faithfulness = _keyword_overlap(sample.answer, " ".join(sample.contexts))
    answer_relevancy = _keyword_overlap(sample.answer, sample.question)
    context_precision = _keyword_overlap(" ".join(sample.contexts), sample.question)
    return EvalResult(
        sample=sample,
        faithfulness=faithfulness,
        answer_relevancy=answer_relevancy,
        context_precision=context_precision,
    )


def evaluate_dataset(path: str, run_fn: Optional[Any] = None) -> Dict[str, float]:
    """Run evaluation on a dataset file.

    Args:
        path: Path to JSONL evaluation dataset.
        run_fn: Optional callable ``(question: str) -> (answer: str, contexts: list[str])``
                 that executes the RAG pipeline.  If not provided, uses
                 pre-existing ``answer`` and ``contexts`` fields from the
                 dataset.

    Returns:
        Aggregated metric averages.
    """
    samples = load_dataset(path)
    results: List[EvalResult] = []

    for sample in samples:
        if run_fn:
            answer, contexts = run_fn(sample.question)
            sample.answer = answer
            sample.contexts = contexts
        result = evaluate_sample(sample)
        results.append(result)

    if not results:
        return {"faithfulness": 0.0, "answer_relevancy": 0.0, "context_precision": 0.0, "overall": 0.0}

    n = len(results)
    return {
        "faithfulness": sum(r.faithfulness for r in results) / n,
        "answer_relevancy": sum(r.answer_relevancy for r in results) / n,
        "context_precision": sum(r.context_precision for r in results) / n,
        "overall": sum(r.overall for r in results) / n,
        "num_samples": n,
    }


def _keyword_overlap(text_a: str, text_b: str) -> float:
    """Simple keyword overlap score between two texts."""
    if not text_a or not text_b:
        return 0.0
    tokens_a = set(text_a.lower().split())
    tokens_b = set(text_b.lower().split())
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    return len(intersection) / max(len(tokens_a), len(tokens_b))
