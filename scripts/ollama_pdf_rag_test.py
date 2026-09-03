#!/usr/bin/env python3
"""One-command PDF -> Ollama embedding -> Qdrant -> Ollama RAG smoke test.

The script keeps Ollama on the host and runs Ragbot/PostgreSQL/Qdrant in Docker.
It intentionally avoids relying on tmp/ragbot-runtime.json so stale local/docker
state cannot leak host paths into container ingestion.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
TMP_DIR = ROOT / "tmp"
BASE_COMPOSE = ROOT / "docker-compose.yml"
OLLAMA_COMPOSE = ROOT / "docker-compose.ollama-host.yml"
ENV_FILE = ROOT / ".env"
DEFAULT_SERVER = "http://127.0.0.1:8000"
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_DOCKER_OLLAMA_URL = "http://host.docker.internal:11434"
DEFAULT_LLM_MODEL = "qwen3.8:27b"
DEFAULT_EMBEDDING_MODEL = "qwen3-embedding:0.6b"
DEFAULT_TENANT = "ollama-pdf-smoke"
MAX_BATCH_SIZE = 100


class UserError(RuntimeError):
    """Expected setup/runtime failure with an actionable message."""


@dataclass
class IngestSummary:
    pdf_count: int
    completed_jobs: int
    documents: int
    chunks: int
    job_ids: List[str]


def _parse_dotenv(path: Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    if not path.exists():
        return values
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        if key:
            values[key] = value
    return values


def _configured(name: str, fallback: str = "") -> str:
    if os.getenv(name):
        return str(os.environ[name])
    return _parse_dotenv(ENV_FILE).get(name, fallback)


def _run(
    command: Sequence[str],
    *,
    env: Optional[Dict[str, str]] = None,
    check: bool = True,
    capture: bool = False,
) -> subprocess.CompletedProcess[str]:
    cmd = [str(v) for v in command]
    print("+", " ".join(cmd))
    return subprocess.run(
        cmd,
        cwd=ROOT,
        env=env,
        check=check,
        text=True,
        capture_output=capture,
    )


def _request_json(
    method: str,
    url: str,
    *,
    payload: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: float = 60.0,
) -> Dict[str, Any]:
    body = None
    request_headers = {"Accept": "application/json"}
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        request_headers["Content-Type"] = "application/json"
    if headers:
        request_headers.update(headers)
    request = urllib.request.Request(url, data=body, headers=request_headers, method=method.upper())
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise UserError(f"HTTP {exc.code} from {url}: {detail[:800]}") from exc
    except urllib.error.URLError as exc:
        raise UserError(f"Could not reach {url}: {exc.reason}") from exc
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise UserError(f"Expected JSON from {url}, got: {raw[:400]}") from exc
    if not isinstance(data, dict):
        raise UserError(f"Expected JSON object from {url}")
    return data


def _auth_headers(api_key: Optional[str]) -> Dict[str, str]:
    return {"X-API-Key": api_key} if api_key else {}


def _docker_ready() -> bool:
    if shutil.which("docker") is None:
        return False
    for cmd in (["docker", "compose", "version"], ["docker", "info"]):
        result = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
        if result.returncode != 0:
            return False
    return True


def _compose_command() -> List[str]:
    return [
        "docker",
        "compose",
        "-f",
        str(BASE_COMPOSE),
        "-f",
        str(OLLAMA_COMPOSE),
    ]


def _model_ids(ollama_url: str, timeout: float) -> set[str]:
    response = _request_json("GET", f"{ollama_url.rstrip('/')}/v1/models", timeout=timeout)
    return {
        str(item["id"])
        for item in response.get("data") or []
        if isinstance(item, dict) and item.get("id")
    }


def _ensure_model(
    model: str,
    *,
    ollama_url: str,
    timeout: float,
    pull_missing: bool,
) -> None:
    if model in _model_ids(ollama_url, timeout):
        print(f"Ollama model ready: {model}")
        return
    if not pull_missing:
        raise UserError(
            f"Ollama model is not installed: {model}. Run `ollama pull {model}` "
            "or rerun with --pull-missing."
        )
    if shutil.which("ollama") is None:
        raise UserError("`ollama` CLI was not found")
    _run(["ollama", "pull", model])
    if model not in _model_ids(ollama_url, timeout):
        raise UserError(f"Ollama model still unavailable after pull: {model}")
    print(f"Ollama model ready: {model}")


def _probe_embedding(args: argparse.Namespace) -> int:
    response = _request_json(
        "POST",
        f"{args.ollama_url.rstrip('/')}/v1/embeddings",
        payload={"model": args.embedding_model, "input": "Ragbot embedding dimension probe"},
        timeout=args.ollama_timeout,
    )
    items = response.get("data") or []
    vector = items[0].get("embedding") if items and isinstance(items[0], dict) else None
    if not isinstance(vector, list) or not vector:
        raise UserError("Ollama embedding probe returned no valid vector")
    actual = len(vector)
    if args.embedding_dim is not None and args.embedding_dim != actual:
        raise UserError(
            "Embedding dimension mismatch before startup: "
            f"model={args.embedding_model}, actual={actual}, configured={args.embedding_dim}. "
            "Remove --embedding-dim to auto-detect or set it to the real dimension."
        )
    args.embedding_dim = actual
    if not args.collection:
        args.collection = _default_collection(args.embedding_model, actual)
    print(f"Embedding probe ready: {args.embedding_model} ({actual}D)")
    print(f"Qdrant collection: {args.collection}")
    return actual


def _default_collection(model: str, dimension: int) -> str:
    safe = re.sub(r"[^a-z0-9]+", "_", model.lower()).strip("_") or "embedding"
    return f"rag_chunks_smoke_{safe}_{dimension}"


def _probe_generation(args: argparse.Namespace) -> None:
    response = _request_json(
        "POST",
        f"{args.ollama_url.rstrip('/')}/v1/chat/completions",
        payload={
            "model": args.model,
            "messages": [{"role": "user", "content": "Reply with exactly RAGBOT_OLLAMA_OK"}],
            "max_tokens": 32,
            "reasoning_effort": args.reasoning_effort,
        },
        timeout=args.ollama_timeout,
    )
    choices = response.get("choices") or []
    text = ""
    if choices and isinstance(choices[0], dict):
        message = choices[0].get("message") or {}
        if isinstance(message, dict):
            text = str(message.get("content") or "")
    if not text.strip():
        raise UserError(f"Ollama generation probe returned empty output for {args.model}")
    print(f"Generation probe ready: {args.model}")


def _compose_env(args: argparse.Namespace) -> Dict[str, str]:
    if args.embedding_dim is None or not args.collection:
        raise ValueError("embedding dimension and collection must be resolved before Compose startup")
    env = os.environ.copy()
    env.update(_parse_dotenv(ENV_FILE))
    env.update(
        {
            "RAGBOT_ENV": "development",
            "RAGBOT_LLM_PROVIDER": "ollama",
            "OLLAMA_MODEL": args.model,
            "OLLAMA_TIMEOUT_SECONDS": str(args.ollama_timeout),
            "OLLAMA_REASONING_EFFORT": args.reasoning_effort,
            "RAGBOT_DOCKER_OLLAMA_BASE_URL": args.docker_ollama_url,
            "EMBEDDING_MODEL": args.embedding_model,
            "EMBEDDING_API_KEY": "ollama",
            "EMBEDDING_BASE_URL": args.docker_ollama_url,
            "QDRANT_DIM": str(args.embedding_dim),
            "QDRANT_COLLECTION": args.collection,
            "RAGBOT_DATA_DIR": str(args.data_dir.resolve()),
            "RAGBOT_API_PORT": str(args.port),
        }
    )
    return env


def _start_docker(args: argparse.Namespace, env: Dict[str, str]) -> None:
    command = _compose_command() + ["up", "-d"]
    if not args.no_build:
        command.append("--build")
    command.append("--force-recreate")
    _run(command, env=env)


def _wait_ready(server: str, timeout: float) -> Dict[str, Any]:
    deadline = time.monotonic() + timeout
    last: Optional[Exception] = None
    while time.monotonic() < deadline:
        try:
            result = _request_json("GET", f"{server.rstrip('/')}/admin/ready", timeout=5.0)
            if result.get("status") == "ready":
                print(f"Ragbot ready: {server}")
                return result
        except Exception as exc:
            last = exc
        time.sleep(1.0)
    raise UserError(f"Ragbot did not become ready within {timeout:.0f}s: {last}")


def _verify_container_contract(args: argparse.Namespace, env: Dict[str, str]) -> None:
    code = (
        "import os, urllib.request; "
        "assert os.environ.get('RAGBOT_ALLOWED_LOCAL_SOURCE_ROOTS') == '/data'; "
        f"assert os.environ.get('EMBEDDING_MODEL') == {args.embedding_model!r}; "
        f"assert os.environ.get('QDRANT_DIM') == {str(args.embedding_dim)!r}; "
        f"urllib.request.urlopen({(args.docker_ollama_url.rstrip('/') + '/v1/models')!r}, timeout=10).read(); "
        "print('container-contract-ok')"
    )
    _run(_compose_command() + ["exec", "-T", "worker", "python", "-c", code], env=env)


def _discover_pdfs(data_dir: Path) -> List[Path]:
    root = data_dir.resolve()
    if not root.is_dir():
        raise UserError(f"PDF data directory does not exist: {root}")
    pdfs = sorted(
        (p.resolve() for p in root.rglob("*") if p.is_file() and p.suffix.lower() == ".pdf"),
        key=lambda p: p.as_posix().lower(),
    )
    if not pdfs:
        raise UserError(f"No PDF files found below {root}")
    return pdfs


def _container_location(path: Path, data_dir: Path) -> str:
    relative = path.resolve().relative_to(data_dir.resolve())
    return f"/data/{relative.as_posix()}"


def _manifest_sources(
    pdfs: Iterable[Path],
    *,
    data_dir: Path,
    tag: str,
) -> List[Dict[str, Any]]:
    root = data_dir.resolve()
    return [
        {
            "location": _container_location(path, root),
            "source_type": "pdf",
            "name": path.resolve().relative_to(root).as_posix(),
            "tags": [tag],
            "reuse_source": False,
            "dedupe_active_job": False,
        }
        for path in pdfs
    ]


def _batches(items: Sequence[Dict[str, Any]]) -> Iterable[Sequence[Dict[str, Any]]]:
    for start in range(0, len(items), MAX_BATCH_SIZE):
        yield items[start : start + MAX_BATCH_SIZE]


def _write_manifest(
    sources: Sequence[Dict[str, Any]],
    *,
    tenant: str,
    batch_index: int,
) -> Path:
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    output = TMP_DIR / f"ollama-pdf-rag-{tenant}-batch-{batch_index:04d}.json"
    output.write_text(
        json.dumps({"tenant_id": tenant, "sources": list(sources)}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return output


def _wait_job(
    server: str,
    job_id: str,
    *,
    headers: Dict[str, str],
    timeout: float,
) -> Dict[str, Any]:
    deadline = time.monotonic() + timeout
    previous = None
    while True:
        job = _request_json(
            "GET",
            f"{server.rstrip('/')}/ingest/jobs/{job_id}",
            headers=headers,
            timeout=min(60.0, timeout),
        )
        status = str(job.get("status") or "unknown")
        if status != previous:
            print(
                f"Ingestion {job_id}: {status} "
                f"(docs={job.get('doc_count', 0)}, chunks={job.get('chunk_count', 0)})"
            )
            previous = status
        if status == "completed":
            return job
        if status == "failed":
            raise UserError(job.get("error") or f"Ingestion failed: {job_id}")
        if time.monotonic() >= deadline:
            raise UserError(f"Timed out waiting for ingestion job {job_id}")
        time.sleep(1.0)


def _ingest(args: argparse.Namespace, pdfs: Sequence[Path]) -> IngestSummary:
    headers = _auth_headers(args.api_key)
    sources = _manifest_sources(pdfs, data_dir=args.data_dir, tag=args.tag)
    jobs: List[str] = []
    docs = chunks = completed = 0
    for index, batch in enumerate(_batches(sources), 1):
        manifest = _write_manifest(batch, tenant=args.tenant, batch_index=index)
        print(f"PDF batch {index}: {len(batch)} source(s); manifest={manifest}")
        result = _request_json(
            "POST",
            f"{args.server.rstrip('/')}/ingest/batch",
            payload={"tenant_id": args.tenant, "sources": list(batch)},
            headers=headers,
            timeout=180.0,
        )
        if int(result.get("failed") or 0):
            raise UserError(f"Ragbot rejected PDF source(s): {result}")
        for item in result.get("items") or []:
            if not isinstance(item, dict) or not item.get("job_id"):
                continue
            job_id = str(item["job_id"])
            if job_id in jobs:
                continue
            jobs.append(job_id)
            job = _wait_job(args.server, job_id, headers=headers, timeout=args.ingest_timeout)
            completed += 1
            docs += int(job.get("doc_count") or 0)
            chunks += int(job.get("chunk_count") or 0)
    if chunks <= 0:
        raise UserError("Ingestion completed without fresh chunks")
    summary = IngestSummary(len(pdfs), completed, docs, chunks, jobs)
    print(
        f"Knowledge ready: pdfs={summary.pdf_count}, jobs={summary.completed_jobs}, "
        f"docs={summary.documents}, chunks={summary.chunks}"
    )
    return summary


def _search(args: argparse.Namespace) -> Dict[str, Any]:
    result = _request_json(
        "POST",
        f"{args.server.rstrip('/')}/search",
        payload={
            "query": args.query,
            "tenant_id": args.tenant,
            "user_id": args.user,
            "top_k": args.top_k,
        },
        headers=_auth_headers(args.api_key),
        timeout=120.0,
    )
    diagnostics = result.get("diagnostics") or {}
    chunks = result.get("chunks") or []
    if not chunks:
        raise UserError(
            "Ragbot /search returned 0 chunks after fresh vectorization. "
            f"Diagnostics: {json.dumps(diagnostics, ensure_ascii=False)}"
        )
    if diagnostics.get("semantic_embedding") is not True:
        raise UserError(
            "Search is not using semantic embeddings. "
            f"Diagnostics: {json.dumps(diagnostics, ensure_ascii=False)}"
        )
    actual_model = str(diagnostics.get("embedding_model") or "")
    if actual_model and actual_model != args.embedding_model:
        raise UserError(
            f"Search embedder mismatch: expected={args.embedding_model}, actual={actual_model}"
        )
    print(
        f"Retrieval ready: results={len(chunks)}, embedding={actual_model}, "
        f"vector_store={diagnostics.get('vector_store')}"
    )
    return result


def _chat(args: argparse.Namespace) -> Dict[str, Any]:
    result = _request_json(
        "POST",
        f"{args.server.rstrip('/')}/chat",
        payload={"query": args.query, "tenant_id": args.tenant, "user_id": args.user},
        headers=_auth_headers(args.api_key),
        timeout=args.ollama_timeout,
    )
    answer = str(result.get("answer") or "")
    if not answer.strip():
        raise UserError("Ragbot /chat returned an empty answer")
    print("\n=== Ollama RAG answer ===")
    print(answer)
    citations = result.get("citations") or []
    if citations:
        print("\nCitations:")
        for citation in citations:
            print(f"- {citation}")
    return result


def _report(
    args: argparse.Namespace,
    ready: Dict[str, Any],
    ingest: IngestSummary,
    search: Dict[str, Any],
    chat: Dict[str, Any],
) -> Dict[str, Any]:
    retrieved = search.get("chunks") or []
    return {
        "status": "pass",
        "tenant": args.tenant,
        "generation_model": args.model,
        "embedding_model": args.embedding_model,
        "embedding_dimension": args.embedding_dim,
        "qdrant_collection": args.collection,
        "readiness": ready,
        "ingestion": {
            "pdf_count": ingest.pdf_count,
            "documents": ingest.documents,
            "chunks": ingest.chunks,
            "job_ids": ingest.job_ids,
        },
        "retrieval": {
            "count": len(retrieved),
            "top_score": retrieved[0].get("score") if retrieved else None,
            "diagnostics": search.get("diagnostics") or {},
            "chunk_ids": [c.get("chunk_id") for c in retrieved],
        },
        "generation": {
            "answer": chat.get("answer"),
            "citations": chat.get("citations") or [],
            "confidence": chat.get("confidence"),
        },
    }


def _write_report(path: str, report: Dict[str, Any]) -> None:
    output = Path(path)
    if not output.is_absolute():
        output = (ROOT / output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Report: {output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Start Docker, vectorize ./data PDFs with Ollama, then verify semantic RAG"
    )
    parser.add_argument(
        "query",
        nargs="?",
        default="请根据已导入的PDF文档，总结最重要的技术要点，并给出引用依据。",
    )
    parser.add_argument("--data", default="data")
    parser.add_argument("--tenant", default=DEFAULT_TENANT)
    parser.add_argument("--user", default="ollama-pdf-smoke")
    parser.add_argument("--server", default=DEFAULT_SERVER)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--api-key")
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    parser.add_argument("--docker-ollama-url", default=DEFAULT_DOCKER_OLLAMA_URL)
    parser.add_argument("--model", default=_configured("OLLAMA_MODEL", DEFAULT_LLM_MODEL))
    parser.add_argument(
        "--embedding-model",
        default=_configured("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=None,
        help="Expected embedding dimension. Omit to auto-detect from Ollama.",
    )
    parser.add_argument(
        "--collection",
        default=None,
        help="Qdrant collection. Omit to derive an isolated collection from model + dimension.",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=["none", "low", "medium", "high", "max"],
        default=_configured("OLLAMA_REASONING_EFFORT", "none") or "none",
    )
    parser.add_argument("--ollama-timeout", type=float, default=300.0)
    parser.add_argument("--startup-timeout", type=float, default=180.0)
    parser.add_argument("--ingest-timeout", type=float, default=1800.0)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--tag", default="ollama-pdf-smoke")
    parser.add_argument("--pull-missing", action="store_true")
    parser.add_argument("--no-build", action="store_true")
    parser.add_argument("--skip-direct-generation", action="store_true")
    parser.add_argument("--down-after", action="store_true")
    parser.add_argument("--output")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    args.data_dir = Path(args.data)
    if not args.data_dir.is_absolute():
        args.data_dir = (ROOT / args.data_dir).resolve()
    if args.embedding_dim is not None and args.embedding_dim <= 0:
        print("ERROR: --embedding-dim must be positive", file=sys.stderr)
        return 2
    if args.top_k <= 0:
        print("ERROR: --top-k must be positive", file=sys.stderr)
        return 2

    env: Optional[Dict[str, str]] = None
    try:
        print("=== 1/7 Host prerequisites ===")
        if not _docker_ready():
            raise UserError("Docker daemon is not ready; verify `docker info`")
        _ensure_model(
            args.model,
            ollama_url=args.ollama_url,
            timeout=args.ollama_timeout,
            pull_missing=args.pull_missing,
        )
        _ensure_model(
            args.embedding_model,
            ollama_url=args.ollama_url,
            timeout=args.ollama_timeout,
            pull_missing=args.pull_missing,
        )

        print("=== 2/7 Ollama embedding/generation probes ===")
        _probe_embedding(args)
        if not args.skip_direct_generation:
            _probe_generation(args)

        print("=== 3/7 Discover PDF corpus ===")
        pdfs = _discover_pdfs(args.data_dir)
        print(f"PDF root: {args.data_dir}")
        print(f"Discovered PDFs: {len(pdfs)}")
        for pdf in pdfs:
            print(f"- {pdf.relative_to(args.data_dir)} -> {_container_location(pdf, args.data_dir)}")

        print("=== 4/7 Start Ragbot Docker stack ===")
        env = _compose_env(args)
        _start_docker(args, env)
        ready = _wait_ready(args.server, args.startup_timeout)
        _verify_container_contract(args, env)

        print("=== 5/7 Ingest and vectorize PDFs ===")
        ingest = _ingest(args, pdfs)

        print("=== 6/7 Verify semantic retrieval ===")
        search = _search(args)

        print("=== 7/7 Verify Ollama RAG answer ===")
        chat = _chat(args)
        report = _report(args, ready, ingest, search, chat)
        if args.output:
            _write_report(args.output, report)

        print("\nOllama PDF RAG smoke test: PASS")
        print(
            f"  PDFs={ingest.pdf_count}, chunks={ingest.chunks}, "
            f"retrieved={len(search.get('chunks') or [])}"
        )
        print(
            f"  embedding={args.embedding_model}/{args.embedding_dim}D, "
            f"collection={args.collection}, generation={args.model}"
        )
        return 0
    except (UserError, subprocess.CalledProcessError, OSError, ValueError) as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        print("\nInspect Docker logs with:", file=sys.stderr)
        print(
            "  docker compose "
            f"-f {BASE_COMPOSE} -f {OLLAMA_COMPOSE} logs --tail=200 api worker",
            file=sys.stderr,
        )
        return 1
    finally:
        if getattr(args, "down_after", False) and env is not None:
            try:
                _run(_compose_command() + ["down"], env=env)
            except Exception as exc:
                print(f"WARNING: failed to stop Compose stack: {exc}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
