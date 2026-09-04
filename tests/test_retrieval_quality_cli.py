from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from cli.rag import build_parser
from services.api.app.routes.search import SearchRequest


ROOT = Path(__file__).resolve().parents[1]


def test_search_cli_exposes_ablation_explain_and_reranker_controls():
    parser = build_parser()
    args = parser.parse_args(
        [
            "--tenant",
            "engineering",
            "search",
            "lower VRAM usage",
            "--mode",
            "vector",
            "--candidate-pool",
            "50",
            "--no-rerank",
            "--explain",
        ]
    )

    assert args.mode == "vector"
    assert args.candidate_pool == 50
    assert args.no_rerank is True
    assert args.explain is True


def test_search_api_defaults_preserve_normal_hybrid_reranking():
    request = SearchRequest(
        query="GPU memory",
        tenant_id="engineering",
        user_id="tester",
    )

    assert request.mode == "hybrid"
    assert request.candidate_pool is None
    assert request.rerank is True
    assert request.explain is False


def test_retrieval_ablation_help_is_importable_without_live_server():
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "retrieval_ablation.py"), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--with-reranker" in result.stdout
    assert "--candidate-pool" in result.stdout
