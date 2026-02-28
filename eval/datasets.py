"""Evaluation dataset management for ragbot.

Supports loading, saving, and managing evaluation datasets for
doc QA, DB QA, and code tasks. Each dataset entry includes:
- query, expected answer (or pattern), expected citations, category, tags.

File format: JSON Lines (.jsonl) or JSON array.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

logger = logging.getLogger(__name__)

EvalCategory = Literal["doc_qa", "db_qa", "code_task", "mixed"]


@dataclass
class EvalCase:
    """A single evaluation case."""

    case_id: str
    query: str
    category: EvalCategory
    tenant_id: str = "eval"
    user_id: str = "eval-user"

    # Expected outputs (at least one should be set)
    expected_answer_contains: Optional[List[str]] = None
    expected_answer_not_contains: Optional[List[str]] = None
    expected_route: Optional[str] = None
    expected_confidence: Optional[str] = None
    expected_citation_kinds: Optional[List[str]] = None
    expected_min_citations: int = 0
    expected_min_evidence: int = 0

    # Constraints
    constraints: Optional[Dict[str, Any]] = None

    # Setup data (for DB/code tests)
    setup_tables: Optional[List[Dict[str, Any]]] = None
    setup_files: Optional[Dict[str, str]] = None

    tags: List[str] = field(default_factory=list)


@dataclass
class EvalResult:
    """Result of running one evaluation case."""

    case_id: str
    category: str
    passed: bool
    checks: Dict[str, bool] = field(default_factory=dict)
    actual_answer: str = ""
    actual_route: str = ""
    actual_confidence: str = ""
    actual_citation_count: int = 0
    actual_evidence_count: int = 0
    duration_ms: int = 0
    error: Optional[str] = None
    failure_category: Optional[str] = None  # "bad_retrieval" | "bad_synthesis" | "bad_tool" | "error"


def load_dataset(path: str) -> List[EvalCase]:
    """Load an evaluation dataset from a JSON or JSONL file."""
    filepath = Path(path)
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    content = filepath.read_text(encoding="utf-8")

    if filepath.suffix == ".jsonl":
        entries = [json.loads(line) for line in content.strip().splitlines() if line.strip()]
    else:
        data = json.loads(content)
        entries = data if isinstance(data, list) else data.get("cases", [])

    cases = []
    for entry in entries:
        cases.append(EvalCase(**{
            k: v for k, v in entry.items()
            if k in EvalCase.__dataclass_fields__
        }))

    logger.info("Loaded %d eval cases from %s", len(cases), path)
    return cases


def save_dataset(cases: List[EvalCase], path: str) -> None:
    """Save an evaluation dataset to a JSON file."""
    data = [asdict(c) for c in cases]
    Path(path).write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Saved %d eval cases to %s", len(cases), path)


def save_results(results: List[EvalResult], path: str) -> None:
    """Save evaluation results to a JSON file."""
    data = [asdict(r) for r in results]
    Path(path).write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def build_sample_dataset() -> List[EvalCase]:
    """Build a small sample evaluation dataset for testing."""
    return [
        EvalCase(
            case_id="doc-001",
            query="What is Postgres used for?",
            category="doc_qa",
            expected_answer_contains=["Postgres"],
            expected_route="doc_rag",
            expected_min_evidence=1,
        ),
        EvalCase(
            case_id="db-001",
            query="SELECT region FROM sales WHERE amount > 10",
            category="db_qa",
            expected_route="sql",
            expected_answer_contains=["SQL"],
            setup_tables=[{
                "name": "sales",
                "columns": [
                    {"name": "region", "type": "text"},
                    {"name": "amount", "type": "int"},
                ],
                "rows": [
                    {"region": "cn", "amount": 15},
                    {"region": "us", "amount": 5},
                ],
            }],
        ),
        EvalCase(
            case_id="code-001",
            query="Find the hello function",
            category="code_task",
            expected_route="code",
            expected_min_evidence=1,
            setup_files={"default": {"main.py": "def hello():\n    print('world')\n"}},
        ),
    ]
