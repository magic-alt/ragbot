"""
Ragbot Example: PDF Ingestion + Multi-Route Query Demo
=======================================================

This example demonstrates the full ragbot pipeline using the bundled
OpenVLA paper (PDF) as a knowledge source:

  1. Initialize in-memory services (no external DB required)
  2. Ingest a local PDF into chunks and embed them
  3. Register a sample SQL table
  4. Run three different query types through the agent:
     - Doc RAG  : semantic retrieval over PDF chunks
     - SQL      : structured data query
     - Code     : code search in project source

Run from the project root:
    python examples/example_case.py

Or from anywhere (the script auto-fixes sys.path):
    python <absolute-path>/examples/example_case.py
"""
from __future__ import annotations

import json
import os
import sys
import textwrap

# ---------------------------------------------------------------------------
# Fix sys.path so that `services.*` and `contracts.*` are importable
# regardless of the current working directory.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Ensure stdout uses UTF-8 on Windows (avoids GBK encoding errors for Unicode text)
if sys.platform == "win32" and hasattr(sys.stdout, "buffer"):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from services.api.app.agent.graph import build_default_services
from services.api.app.agent.nodes.code import CodeSearch
from services.api.app.auth.acl import build_policy
from services.api.app.main import chat
from services.api.app.storage.models import Chunk, Document, TableData
from services.worker.connectors.pdf import fetch_pdf
from services.worker.jobs.embed_and_upsert import embed_and_upsert
from services.worker.jobs.ingest_pdf import ingest_pdf

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TENANT_ID = "demo-tenant"
USER_ID = "demo-user"
PDF_FILENAME = "OpenVLA_AnOpen-Source Vision-Language-Action_Model.pdf"


def _separator(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def _print_result(result: dict) -> None:
    """Pretty-print a chat() response."""
    print(f"  Route      : {result['debug']['route']}")
    print(f"  Confidence : {result['confidence']}")
    print(f"  Answer     :")
    for line in textwrap.wrap(result["answer"], width=72):
        print(f"    {line}")
    tool_calls = result["debug"].get("tool_calls", [])
    if tool_calls:
        print(f"  Tool calls : {len(tool_calls)}")
        for tc in tool_calls:
            status = "OK" if tc.get("ok") else f"FAIL({tc.get('error', '?')})"
            print(f"    - {tc['name']} [{status}]")
    citations = result.get("citations", [])
    if citations:
        print(f"  Citations  : {len(citations)}")
        for i, cite in enumerate(citations[:3], 1):
            kind = cite.get("kind", "?")
            cid = cite.get("chunk_id") or cite.get("path") or cite.get("url") or ""
            print(f"    [{i}] {kind}: {cid}")
        if len(citations) > 3:
            print(f"    ... and {len(citations) - 3} more")


def main() -> None:
    # ------------------------------------------------------------------
    # Step 1: Build services (all in-memory, no external deps needed)
    # ------------------------------------------------------------------
    _separator("Step 1: Initialize Services")
    services = build_default_services()

    # Override code_search so it does NOT scan the whole filesystem,
    # instead provide a small in-memory sample for the code-search demo.
    services.code_search = CodeSearch(
        repo_roots={},
        in_memory_files={
            "default": {
                "robot_controller.py": textwrap.dedent("""\
                    class OpenVLAController:
                        \"\"\"A controller that uses OpenVLA for robot manipulation.\"\"\"

                        def __init__(self, model_path: str):
                            self.model_path = model_path

                        def predict_action(self, image, instruction: str):
                            \"\"\"Predict a 7-DoF robot action from image + language instruction.\"\"\"
                            pass

                        def execute(self, env, instruction: str, max_steps: int = 100):
                            for step in range(max_steps):
                                obs = env.get_observation()
                                action = self.predict_action(obs, instruction)
                                env.step(action)
                """),
            }
        },
    )
    print("  Services initialized (InMemory mode, no external DB required)")

    # ------------------------------------------------------------------
    # Step 2: Setup ACL policy
    # ------------------------------------------------------------------
    _separator("Step 2: Setup ACL Policy")
    policy = build_policy("policy-1", TENANT_ID, {"allow_users": [USER_ID]})
    services.repo.add_policy(policy)
    print(f"  Policy created: allow_users=[{USER_ID}]")
    print(f"  Policy hash  : {policy.policy_hash}")

    # ------------------------------------------------------------------
    # Step 3: Ingest the PDF file into chunks
    # ------------------------------------------------------------------
    _separator("Step 3: Ingest PDF")
    pdf_path = os.path.join(os.path.dirname(__file__), PDF_FILENAME)
    if not os.path.isfile(pdf_path):
        print(f"  ERROR: PDF not found at {pdf_path}")
        print(f"  Please place '{PDF_FILENAME}' in the examples/ folder.")
        sys.exit(1)

    # First, register a Document metadata record
    doc = Document(
        doc_id="doc-openvla",
        tenant_id=TENANT_ID,
        source_type="pdf",
        title="OpenVLA: An Open-Source Vision-Language-Action Model",
        uri=f"file://{pdf_path}",
        version="v1",
        doc_updated_at="2024-06-01",
        ingested_at="2025-01-01",
        tags=["paper", "robotics", "VLA"],
        acl_policy_id=policy.acl_policy_id,
    )
    services.repo.add_document(doc)

    # Ingest PDF: extract text -> split into chunks (800 chars, 100 overlap)
    chunks = list(ingest_pdf(
        path=pdf_path,
        doc_id=doc.doc_id,
        tenant_id=TENANT_ID,
        chunk_size=800,
        chunk_overlap=100,
        version=doc.version,
        tags=doc.tags,
        acl_hash=policy.policy_hash,
    ))
    print(f"  PDF extracted: {len(chunks)} chunks from {PDF_FILENAME}")
    if chunks:
        print(f"  First chunk preview ({len(chunks[0].text)} chars):")
        preview = chunks[0].text[:120].replace("\n", " ")
        print(f"    \"{preview}...\"")

    # Embed each chunk and upsert into the in-memory vector store
    embed_and_upsert(services.repo, services.qdrant, chunks, batch_size=50)
    print(f"  Embedded and indexed: {len(chunks)} chunks")

    # ------------------------------------------------------------------
    # Step 4: Register a sample SQL table
    # ------------------------------------------------------------------
    _separator("Step 4: Register SQL Table")
    table = TableData(
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
            {"model": "RT-2-X", "task": "wipe-table", "success_rate": 48.0},
            {"model": "Octo-Base", "task": "pick-and-place", "success_rate": 34.6},
            {"model": "Diffusion Policy", "task": "pick-and-place", "success_rate": 52.8},
        ],
    )
    services.repo.register_table(table)
    print(f"  Table '{table.name}' registered with {len(table.rows)} rows")

    # ------------------------------------------------------------------
    # Step 5: Query 1 - Doc RAG (Retrieve from PDF chunks)
    # ------------------------------------------------------------------
    _separator("Query 1: Doc RAG - 'What is OpenVLA?'")
    result1 = chat("What is OpenVLA and what does it do?", TENANT_ID, USER_ID, services)
    _print_result(result1)

    # ------------------------------------------------------------------
    # Step 6: Query 2 - SQL Query
    # ------------------------------------------------------------------
    _separator("Query 2: SQL - 'Pick-and-place success rates'")
    result2 = chat(
        "SELECT model, success_rate FROM model_benchmarks WHERE task = 'pick-and-place'",
        TENANT_ID, USER_ID, services,
    )
    _print_result(result2)

    # ------------------------------------------------------------------
    # Step 7: Query 3 - Code Search
    # ------------------------------------------------------------------
    _separator("Query 3: Code Search - 'class OpenVLAController'")
    result3 = chat("class OpenVLAController", TENANT_ID, USER_ID, services)
    _print_result(result3)

    # ------------------------------------------------------------------
    # Step 8: Demo - ACL Blocking (unauthorized user)
    # ------------------------------------------------------------------
    _separator("Query 4: ACL Block - unauthorized user")
    result4 = chat("What is OpenVLA?", TENANT_ID, "unauthorized-user", services)
    _print_result(result4)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    _separator("Summary")
    print("  4 queries executed across 3 routes + 1 ACL test:")
    print(f"    1. Doc RAG   -> confidence={result1['confidence']}")
    print(f"    2. SQL       -> confidence={result2['confidence']}")
    print(f"    3. Code      -> confidence={result3['confidence']}")
    print(f"    4. ACL Block -> confidence={result4['confidence']}")
    print("\n  All done!")


if __name__ == "__main__":
    main()
