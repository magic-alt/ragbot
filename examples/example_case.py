"""Ragbot executable example: PDF ingestion + multi-route Agent queries.

The example intentionally uses in-memory storage and deterministic hash
embeddings so it can run in CI without external credentials. It exercises the
same Source -> ingestion pipeline used by the service and then validates doc,
SQL, code and ACL behavior through the async Agent API.

Run from the project root:
    python examples/example_case.py
"""
from __future__ import annotations

import asyncio
import os
import sys
import textwrap

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

if sys.platform == "win32" and hasattr(sys.stdout, "buffer"):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from services.api.app.agent.graph import build_default_services
from services.api.app.agent.nodes.code import CodeSearch
from services.api.app.auth.acl import build_policy
from services.api.app.main import chat
from services.api.app.storage.models import Source, TableData
from services.worker.pipeline import run_ingest_pipeline

TENANT_ID = "demo-tenant"
USER_ID = "demo-user"
PDF_FILENAME = "OpenVLA_AnOpen-Source Vision-Language-Action_Model.pdf"


def _separator(title: str) -> None:
    print(f"\n{'=' * 64}\n  {title}\n{'=' * 64}")


def _print_result(result: dict) -> None:
    print(f"  Route      : {result['debug']['route']}")
    print(f"  Confidence : {result['confidence']}")
    answer = str(result.get("answer", ""))
    print("  Answer     :")
    for line in textwrap.wrap(answer, width=76) or [""]:
        print(f"    {line}")
    calls = result["debug"].get("tool_calls", [])
    if calls:
        print(f"  Tool calls : {len(calls)}")
        for call in calls:
            status = "OK" if call.get("ok") else f"FAIL({call.get('error', '?')})"
            print(f"    - {call['name']} [{status}]")
    citations = result.get("citations", [])
    if citations:
        print(f"  Citations  : {len(citations)}")
        for index, cite in enumerate(citations[:3], 1):
            identifier = cite.get("chunk_id") or cite.get("path") or cite.get("url") or ""
            print(f"    [{index}] {cite.get('kind', '?')}: {identifier}")


def _build_services():
    services = build_default_services()
    services.code_search = CodeSearch(
        repo_roots={},
        in_memory_files={
            "default": {
                "robot_controller.py": textwrap.dedent("""\
                    class OpenVLAController:
                        \"\"\"Controller using OpenVLA for robot manipulation.\"\"\"

                        def predict_action(self, image, instruction: str):
                            \"\"\"Predict a 7-DoF action from image and language.\"\"\"
                            pass
                """),
            }
        },
    )
    return services


async def _run_queries(services):
    _separator("Query 1: Document RAG")
    doc_result = await chat(
        "What is OpenVLA and what does it do?", TENANT_ID, USER_ID, services
    )
    _print_result(doc_result)

    _separator("Query 2: SQL")
    sql_result = await chat(
        "SELECT model, success_rate FROM model_benchmarks WHERE task = 'pick-and-place'",
        TENANT_ID,
        USER_ID,
        services,
    )
    _print_result(sql_result)

    _separator("Query 3: Code search")
    code_result = await chat("class OpenVLAController", TENANT_ID, USER_ID, services)
    _print_result(code_result)

    _separator("Query 4: ACL negative test")
    denied_result = await chat(
        "What is OpenVLA?", TENANT_ID, "unauthorized-user", services
    )
    _print_result(denied_result)

    # Keep assertions semantic rather than pinning exact citation counts or
    # heuristic wording, which can legitimately evolve with the Agent graph.
    assert doc_result["debug"]["route"] in {"doc_rag", "mixed"}
    assert doc_result["citations"], "Document RAG returned no citations"
    assert sql_result["debug"]["route"] == "sql"
    assert any(call["name"] == "sql_query" and call["ok"] for call in sql_result["debug"]["tool_calls"])
    assert code_result["debug"]["route"] == "code"
    assert any(call["name"] == "code_search" and call["ok"] for call in code_result["debug"]["tool_calls"])
    assert denied_result["confidence"] == "low"

    return doc_result, sql_result, code_result, denied_result


def main() -> int:
    _separator("Step 1: Initialize services")
    services = _build_services()
    print("  InMemoryRepo + InMemoryQdrant + deterministic HashEmbedder ready")

    _separator("Step 2: Configure ACL")
    policy = build_policy("policy-openvla", TENANT_ID, {"allow_users": [USER_ID]})
    services.repo.add_policy(policy)
    print(f"  ACL policy: {policy.acl_policy_id}")

    _separator("Step 3: Ingest bundled OpenVLA PDF through Source pipeline")
    pdf_path = os.path.abspath(os.path.join(os.path.dirname(__file__), PDF_FILENAME))
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"Bundled example PDF not found: {pdf_path}")

    source = Source(
        source_id="source-openvla-example",
        tenant_id=TENANT_ID,
        source_type="pdf",
        name="OpenVLA paper",
        config={"path": pdf_path, "doc_id": "doc-openvla", "version": "v1"},
        acl_policy_id=policy.acl_policy_id,
        tags=["paper", "robotics", "vla"],
    )
    services.repo.add_source(source)
    job = run_ingest_pipeline(
        source,
        services.repo,
        services.qdrant,
        job_id="example-openvla-ingest",
        embedder=services.embedder,
    )
    assert job.status == "completed", job.error
    assert job.stats.get("chunks_total", 0) > 0, "PDF produced no chunks"
    print(
        f"  Indexed {job.stats['chunks_total']} chunks; "
        f"vector points={services.qdrant.count()}"
    )

    _separator("Step 4: Register sample SQL table")
    services.repo.register_table(
        TableData(
            name="model_benchmarks",
            columns=[
                {"name": "model", "type": "text"},
                {"name": "task", "type": "text"},
                {"name": "success_rate", "type": "float"},
            ],
            rows=[
                {"model": "OpenVLA", "task": "pick-and-place", "success_rate": 73.2},
                {"model": "OpenVLA", "task": "wipe-table", "success_rate": 68.5},
                {"model": "RT-2-X", "task": "pick-and-place", "success_rate": 52.8},
                {"model": "Octo-Base", "task": "pick-and-place", "success_rate": 34.6},
                {"model": "Diffusion Policy", "task": "pick-and-place", "success_rate": 52.8},
            ],
        )
    )

    results = asyncio.run(_run_queries(services))
    _separator("Validated summary")
    labels = ("Doc RAG", "SQL", "Code", "ACL denied")
    for label, result in zip(labels, results):
        print(f"  {label:12s} route={result['debug']['route']} confidence={result['confidence']}")
    print("\n  Example validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
