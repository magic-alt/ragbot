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
    expected_chunk_ids: Optional[List[str]] = None  # For MRR/Recall computation

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
    retrieved_chunk_ids: List[str] = field(default_factory=list)
    mrr_at_10: float = 0.0
    recall_at_10: float = 0.0
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


def build_full_dataset() -> List[EvalCase]:
    """Build a comprehensive evaluation dataset with 200+ cases.

    Distribution:
        doc_qa:    100  (single_chunk=40, multi_chunk=20, acl_filter=15, conditional=15, negative=10)
        db_qa:      50  (direct_sql=20, nl2sql=15, boundary=10, schema=5)
        code_task:  50  (function_search=15, open_file=10, explain_error=10, apply_patch=10, negative=5)
    """
    cases: List[EvalCase] = []

    # ── doc_qa: single chunk (40) ──
    _doc_topics = [
        ("Postgres", "database", "What is PostgreSQL used for in the system?", ["PostgreSQL", "database"]),
        ("Qdrant", "vector store", "How does Qdrant provide vector search?", ["Qdrant", "vector"]),
        ("FastAPI", "API framework", "What framework powers the API gateway?", ["FastAPI"]),
        ("Docker", "containerization", "How is the application containerized?", ["Docker"]),
        ("Helm", "Kubernetes", "What Helm charts are used for deployment?", ["Helm"]),
        ("Redis", "caching", "How does the system handle caching?", ["cache"]),
        ("OAuth", "authentication", "Describe the authentication mechanism.", ["auth"]),
        ("OpenTelemetry", "observability", "How is distributed tracing implemented?", ["tracing"]),
        ("PDF", "document parsing", "How are PDF documents parsed?", ["PDF"]),
        ("Git", "version control", "How does the system integrate with Git repos?", ["Git"]),
        ("SSE", "streaming", "How does SSE streaming work?", ["SSE", "stream"]),
        ("RRF", "fusion", "What is Reciprocal Rank Fusion?", ["RRF", "fusion"]),
        ("ACL", "access control", "How does ACL filtering work?", ["ACL"]),
        ("Embedding", "vector", "What embedding models are supported?", ["embedding"]),
        ("Reranking", "retrieval", "How does cross-encoder reranking improve results?", ["rerank"]),
        ("Chunking", "text split", "How are documents chunked for indexing?", ["chunk"]),
        ("Dedup", "deduplication", "How are duplicate chunks detected?", ["dedup"]),
        ("Ingestion", "pipeline", "Describe the document ingestion pipeline.", ["ingest"]),
        ("Webhook", "notification", "How are webhook notifications sent?", ["webhook"]),
        ("Rate limit", "throttle", "How is API rate limiting implemented?", ["rate"]),
        ("Pagination", "API", "How does the API handle pagination?", ["page"]),
        ("Error handling", "resilience", "How are errors handled in the API layer?", ["error"]),
        ("Logging", "observability", "What logging framework is used?", ["log"]),
        ("Config", "environment", "How is the application configured?", ["config"]),
        ("Migration", "database schema", "How are database migrations managed?", ["migration"]),
        ("FTS", "full-text search", "How does full-text search work in Postgres?", ["FTS"]),
        ("Worker", "background jobs", "How do background workers process jobs?", ["worker"]),
        ("Connector", "data source", "What data source connectors are available?", ["connector"]),
        ("Evaluation", "quality", "How is retrieval quality evaluated?", ["eval"]),
        ("Cost tracking", "billing", "How are LLM API costs tracked?", ["cost"]),
        ("Circuit breaker", "reliability", "How does the circuit breaker pattern work?", ["circuit"]),
        ("Timeout", "reliability", "What timeout mechanisms are in place?", ["timeout"]),
        ("Retry", "reliability", "How are retries handled for failed requests?", ["retry"]),
        ("Queue", "callback", "How does the callback queue work?", ["queue"]),
        ("Tenant", "multi-tenancy", "How does multi-tenancy work?", ["tenant"]),
        ("Session", "conversation", "How are chat sessions managed?", ["session"]),
        ("Citation", "reference", "How are citations generated in answers?", ["citation"]),
        ("Confidence", "scoring", "How is answer confidence scored?", ["confidence"]),
        ("Routing", "agent", "How does the agent route queries?", ["route"]),
        ("Synthesis", "answer", "How are answers synthesized from evidence?", ["synthesis"]),
    ]
    for i, (topic, area, query, keywords) in enumerate(_doc_topics):
        cases.append(EvalCase(
            case_id=f"doc-single-{i+1:03d}",
            query=query,
            category="doc_qa",
            expected_route="doc_rag",
            expected_answer_contains=keywords,
            expected_min_evidence=1,
            tags=["single_chunk", area],
        ))

    # ── doc_qa: multi chunk (20) ──
    _multi_chunk_topics = [
        ("Compare Postgres FTS with Qdrant vector search for retrieval.", ["Postgres", "Qdrant"]),
        ("Explain the differences between SSE streaming and WebSocket.", ["SSE"]),
        ("How do ACL policies interact with tenant isolation?", ["ACL", "tenant"]),
        ("Describe the full ingestion pipeline from PDF upload to indexed chunks.", ["ingest", "chunk"]),
        ("What are the trade-offs between Cohere and local cross-encoder reranking?", ["rerank"]),
        ("How do embedding models affect retrieval quality?", ["embedding"]),
        ("Compare Docker Compose and Helm deployment approaches.", ["Docker", "Helm"]),
        ("How does the evaluation framework measure retrieval and synthesis quality?", ["eval"]),
        ("Explain the interaction between circuit breaker and timeout in tool calls.", ["circuit", "timeout"]),
        ("How do cost tracking and rate limiting work together?", ["cost", "rate"]),
        ("Describe the complete flow from user query to final answer.", ["query", "answer"]),
        ("What monitoring and observability tools are integrated?", ["tracing"]),
        ("How do documents flow from upload through processing to vector storage?", ["ingest"]),
        ("Compare direct SQL queries with NL2SQL for database questions.", ["SQL"]),
        ("How does the code search system index and query code files?", ["code"]),
        ("Explain cache invalidation across LRU, retrieval, and embedding caches.", ["cache"]),
        ("How are document sources managed across different connector types?", ["connector", "source"]),
        ("Describe the relationship between evaluation cases and regression testing.", ["eval", "test"]),
        ("How do ACL, API key auth, and audit logging provide defense in depth?", ["ACL", "auth"]),
        ("What data model changes require database migrations?", ["migration"]),
    ]
    for i, (query, keywords) in enumerate(_multi_chunk_topics):
        cases.append(EvalCase(
            case_id=f"doc-multi-{i+1:03d}",
            query=query,
            category="doc_qa",
            expected_route="doc_rag",
            expected_answer_contains=keywords,
            expected_min_evidence=2,
            tags=["multi_chunk"],
        ))

    # ── doc_qa: ACL filter (15) ──
    for i in range(15):
        tenant = f"tenant-{i % 3}"
        cases.append(EvalCase(
            case_id=f"doc-acl-{i+1:03d}",
            query=f"Retrieve tenant-specific document about topic-{i+1}",
            category="doc_qa",
            tenant_id=tenant,
            expected_route="doc_rag",
            expected_min_evidence=0,
            tags=["acl_filter"],
            constraints={"source_types": ["pdf"]},
        ))

    # ── doc_qa: conditional filter (15) ──
    _cond_filters = [
        ({"source_types": ["pdf"]}, "PDF"),
        ({"source_types": ["web"]}, "web page"),
        ({"source_types": ["git"]}, "Git repository"),
        ({"path_prefix": "/docs/"}, "/docs/ path"),
        ({"url_prefix": "https://example.com"}, "example.com"),
    ]
    for i in range(15):
        constraints, desc = _cond_filters[i % len(_cond_filters)]
        cases.append(EvalCase(
            case_id=f"doc-cond-{i+1:03d}",
            query=f"Find information about system architecture from {desc} sources",
            category="doc_qa",
            expected_route="doc_rag",
            constraints=constraints,
            tags=["conditional_filter"],
        ))

    # ── doc_qa: negative (10) ──
    _negative_queries = [
        "What is the recipe for chocolate cake?",
        "Who won the 2024 Olympics 100m?",
        "Explain quantum entanglement in detail.",
        "What is the weather today in Tokyo?",
        "How to file taxes in Germany?",
        "Describe the plot of Hamlet.",
        "What programming language was used to build Mars rover?",
        "How many calories are in a banana?",
        "Explain the rules of cricket.",
        "What is the capital of Burkina Faso?",
    ]
    for i, query in enumerate(_negative_queries):
        cases.append(EvalCase(
            case_id=f"doc-neg-{i+1:03d}",
            query=query,
            category="doc_qa",
            expected_route="doc_rag",
            expected_answer_not_contains=["error"],
            tags=["negative"],
        ))

    # ── db_qa: direct SQL (20) ──
    _sales_table = [{
        "name": "sales",
        "columns": [
            {"name": "region", "type": "text"},
            {"name": "product", "type": "text"},
            {"name": "amount", "type": "int"},
            {"name": "quarter", "type": "text"},
        ],
        "rows": [
            {"region": "cn", "product": "widget", "amount": 100, "quarter": "Q1"},
            {"region": "us", "product": "widget", "amount": 200, "quarter": "Q1"},
            {"region": "eu", "product": "gadget", "amount": 150, "quarter": "Q2"},
            {"region": "cn", "product": "gadget", "amount": 80, "quarter": "Q2"},
            {"region": "us", "product": "widget", "amount": 300, "quarter": "Q3"},
        ],
    }]
    _sql_queries = [
        ("SELECT * FROM sales WHERE region='cn'", ["cn"]),
        ("SELECT SUM(amount) FROM sales", ["SQL"]),
        ("SELECT region, SUM(amount) FROM sales GROUP BY region", ["SQL"]),
        ("SELECT * FROM sales WHERE amount > 100", ["SQL"]),
        ("SELECT DISTINCT product FROM sales", ["SQL"]),
        ("SELECT * FROM sales ORDER BY amount DESC LIMIT 3", ["SQL"]),
        ("SELECT COUNT(*) FROM sales WHERE quarter='Q1'", ["SQL"]),
        ("SELECT region FROM sales WHERE product='widget'", ["SQL"]),
        ("SELECT AVG(amount) FROM sales GROUP BY region", ["SQL"]),
        ("SELECT * FROM sales WHERE quarter IN ('Q1','Q2')", ["SQL"]),
        ("SELECT product, COUNT(*) FROM sales GROUP BY product", ["SQL"]),
        ("SELECT MAX(amount) FROM sales", ["SQL"]),
        ("SELECT MIN(amount) FROM sales", ["SQL"]),
        ("SELECT * FROM sales WHERE region='us' AND product='widget'", ["SQL"]),
        ("SELECT quarter, SUM(amount) FROM sales GROUP BY quarter ORDER BY quarter", ["SQL"]),
        ("SELECT region FROM sales WHERE amount = (SELECT MAX(amount) FROM sales)", ["SQL"]),
        ("SELECT * FROM sales WHERE amount BETWEEN 100 AND 200", ["SQL"]),
        ("SELECT DISTINCT quarter FROM sales ORDER BY quarter", ["SQL"]),
        ("SELECT region, product FROM sales WHERE amount > 150", ["SQL"]),
        ("SELECT COUNT(DISTINCT region) FROM sales", ["SQL"]),
    ]
    for i, (query, keywords) in enumerate(_sql_queries):
        cases.append(EvalCase(
            case_id=f"db-direct-{i+1:03d}",
            query=query,
            category="db_qa",
            expected_route="sql",
            expected_answer_contains=keywords,
            setup_tables=_sales_table,
            tags=["direct_sql"],
        ))

    # ── db_qa: NL2SQL (15) ──
    _nl2sql = [
        ("What are the total sales by region?", ["SQL"]),
        ("Which product has the highest sales?", ["SQL"]),
        ("Show me Q1 sales figures", ["SQL"]),
        ("How many sales records are from China?", ["SQL"]),
        ("What is the average sale amount?", ["SQL"]),
        ("List all sales over 100 units", ["SQL"]),
        ("Which quarter had the most sales?", ["SQL"]),
        ("Compare widget and gadget total sales", ["SQL"]),
        ("Show the top 3 sales by amount", ["SQL"]),
        ("What is the total revenue from US market?", ["SQL"]),
        ("Count distinct products in the sales data", ["SQL"]),
        ("What is the minimum sale amount recorded?", ["SQL"]),
        ("Show all sales from European market", ["SQL"]),
        ("Which region-product pair has highest sales?", ["SQL"]),
        ("Summarize sales data by quarter", ["SQL"]),
    ]
    for i, (query, keywords) in enumerate(_nl2sql):
        cases.append(EvalCase(
            case_id=f"db-nl2sql-{i+1:03d}",
            query=query,
            category="db_qa",
            expected_route="sql",
            expected_answer_contains=keywords,
            setup_tables=_sales_table,
            tags=["nl2sql"],
        ))

    # ── db_qa: boundary (10) ──
    _boundary = [
        "SELECT * FROM nonexistent_table",
        "SELECT 1/0 FROM sales",
        "SELECT * FROM sales WHERE 1=0",
        "SELECT '' FROM sales LIMIT 0",
        "SELECT NULL FROM sales",
        "What is in the empty_table?",
        "DROP TABLE sales",
        "UPDATE sales SET amount=0",
        "DELETE FROM sales WHERE region='cn'",
        "INSERT INTO sales VALUES ('jp','thing',999,'Q4')",
    ]
    for i, query in enumerate(_boundary):
        cases.append(EvalCase(
            case_id=f"db-boundary-{i+1:03d}",
            query=query,
            category="db_qa",
            expected_route="sql",
            setup_tables=_sales_table,
            tags=["boundary"],
        ))

    # ── db_qa: schema (5) ──
    _schema_queries = [
        "What columns does the sales table have?",
        "Describe the schema of the sales table",
        "What data types are used in the sales table?",
        "List all tables in the database",
        "What is the primary key of the sales table?",
    ]
    for i, query in enumerate(_schema_queries):
        cases.append(EvalCase(
            case_id=f"db-schema-{i+1:03d}",
            query=query,
            category="db_qa",
            expected_route="sql",
            setup_tables=_sales_table,
            tags=["schema"],
        ))

    # ── code_task: function search (15) ──
    _code_files = {"default": {
        "main.py": "def hello():\n    print('world')\n\ndef goodbye():\n    print('farewell')\n",
        "utils.py": "def parse_config(path):\n    with open(path) as f:\n        return json.load(f)\n\ndef validate_input(data):\n    if not data:\n        raise ValueError('empty')\n",
        "api.py": "def create_app():\n    app = FastAPI()\n    return app\n\ndef health_check():\n    return {'status': 'ok'}\n",
        "models.py": "class User:\n    def __init__(self, name):\n        self.name = name\n\nclass Document:\n    def __init__(self, title):\n        self.title = title\n",
    }}
    _func_searches = [
        ("Find the hello function", "hello"),
        ("Where is the goodbye function defined?", "goodbye"),
        ("Find parse_config", "parse_config"),
        ("Show me the validate_input function", "validate_input"),
        ("Where is create_app defined?", "create_app"),
        ("Find health_check", "health_check"),
        ("Search for the User class", "User"),
        ("Find the Document class", "Document"),
        ("Where is print used?", "print"),
        ("Find functions in utils.py", "parse_config"),
        ("Search for error handling code", "ValueError"),
        ("Find JSON parsing code", "json"),
        ("Where is FastAPI initialized?", "FastAPI"),
        ("Find all function definitions in main.py", "def"),
        ("Search for class definitions", "class"),
    ]
    for i, (query, keyword) in enumerate(_func_searches):
        cases.append(EvalCase(
            case_id=f"code-search-{i+1:03d}",
            query=query,
            category="code_task",
            expected_route="code",
            expected_answer_contains=[keyword],
            expected_min_evidence=1,
            setup_files=_code_files,
            tags=["function_search"],
        ))

    # ── code_task: open_file (10) ──
    _open_file_queries = [
        ("Open main.py", "hello"),
        ("Show contents of utils.py", "parse_config"),
        ("Read api.py", "create_app"),
        ("Display models.py", "User"),
        ("Open the main module", "hello"),
        ("Show me utils.py source", "validate_input"),
        ("Read the API module", "health_check"),
        ("Open models.py and show classes", "Document"),
        ("Show main.py contents", "goodbye"),
        ("Display the api module", "FastAPI"),
    ]
    for i, (query, keyword) in enumerate(_open_file_queries):
        cases.append(EvalCase(
            case_id=f"code-open-{i+1:03d}",
            query=query,
            category="code_task",
            expected_route="code",
            expected_answer_contains=[keyword],
            setup_files=_code_files,
            tags=["open_file"],
        ))

    # ── code_task: explain_error (10) ──
    _error_queries = [
        ("Explain this error: ValueError('empty')", "ValueError"),
        ("What does 'NameError: name x is not defined' mean?", "NameError"),
        ("Explain TypeError: unsupported operand type", "TypeError"),
        ("What causes ImportError: No module named foo?", "ImportError"),
        ("Explain KeyError: 'missing_key'", "KeyError"),
        ("What does IndexError: list index out of range mean?", "IndexError"),
        ("Explain AttributeError: object has no attribute bar", "AttributeError"),
        ("What causes FileNotFoundError: No such file?", "FileNotFoundError"),
        ("Explain ZeroDivisionError: division by zero", "ZeroDivisionError"),
        ("What does ConnectionError mean?", "ConnectionError"),
    ]
    for i, (query, keyword) in enumerate(_error_queries):
        cases.append(EvalCase(
            case_id=f"code-error-{i+1:03d}",
            query=query,
            category="code_task",
            expected_route="code",
            expected_answer_contains=[keyword],
            setup_files=_code_files,
            tags=["explain_error"],
        ))

    # ── code_task: apply_patch (10) ──
    _patch_queries = [
        ("Add a docstring to the hello function", "hello"),
        ("Rename goodbye to farewell", "farewell"),
        ("Add type hints to parse_config", "parse_config"),
        ("Add error handling to validate_input", "validate_input"),
        ("Add a new endpoint to create_app", "app"),
        ("Add a __repr__ method to User class", "User"),
        ("Add a save method to Document class", "Document"),
        ("Fix the health_check to return version", "health_check"),
        ("Add logging to parse_config", "parse_config"),
        ("Add a new helper function to utils.py", "def"),
    ]
    for i, (query, keyword) in enumerate(_patch_queries):
        cases.append(EvalCase(
            case_id=f"code-patch-{i+1:03d}",
            query=query,
            category="code_task",
            expected_route="code",
            expected_answer_contains=[keyword],
            setup_files=_code_files,
            tags=["apply_patch"],
        ))

    # ── code_task: negative (5) ──
    _code_neg = [
        "Find the quantum_processor function",
        "Open the file blockchain_handler.py",
        "Explain the machine learning pipeline",
        "Search for the kubernetes_deploy function",
        "Find the neural_network class",
    ]
    for i, query in enumerate(_code_neg):
        cases.append(EvalCase(
            case_id=f"code-neg-{i+1:03d}",
            query=query,
            category="code_task",
            expected_route="code",
            expected_answer_not_contains=["error"],
            setup_files=_code_files,
            tags=["negative"],
        ))

    return cases
