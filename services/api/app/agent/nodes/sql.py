from __future__ import annotations

import logging
import re
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence

from ..state import AgentState, Citation, EvidenceItem, ToolCallRecord, now_ms
from ..reliability import safe_tool_call
from contracts.types import SqlResult
from ...storage.repo import InMemoryRepo

logger = logging.getLogger(__name__)


class SqlEngine:
    def __init__(self, repo: InMemoryRepo, limit: int = 200) -> None:
        self._repo = repo
        self._limit = limit

    def query(self, query: str, params: Optional[Dict[str, Any]] = None, limit: Optional[int] = None) -> SqlResult:
        start = time.perf_counter()
        query = query.strip().rstrip(";")
        parsed = _parse_select(query)
        if not parsed:
            raise ValueError("Only simple SELECT queries are allowed")
        table_name, columns, where = parsed
        table = self._repo.get_table(table_name)
        if not table:
            raise ValueError(f"Unknown table: {table_name}")
        rows = _apply_where(table.rows, where)
        if columns != ["*"]:
            rows = [{col: row.get(col) for col in columns} for row in rows]
            columns_meta = [col for col in table.columns if col["name"] in columns]
        else:
            columns_meta = table.columns
        limit_value = min(limit or self._limit, self._limit)
        rows = rows[:limit_value]
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        return SqlResult(
            rows=rows,
            columns=columns_meta,
            stats={"row_count": len(rows), "elapsed_ms": elapsed_ms},
        )

    def introspect_schema(self) -> List[Dict[str, Any]]:
        """Return schema info for all registered in-memory tables."""
        state = self._repo.export_state()
        tables = state.get("tables", [])
        result = []
        for table in tables:
            result.append({
                "table_name": table.get("name", "?"),
                "columns": table.get("columns", []),
                "row_count": len(table.get("rows", [])),
            })
        return result


class PostgresSqlEngine:
    def __init__(
        self,
        dsn: str,
        allowed_schemas: Optional[Sequence[str]] = None,
        limit: int = 200,
        timeout_ms: int = 3000,
        pool_min_size: int = 1,
        pool_max_size: int = 5,
    ) -> None:
        try:
            from psycopg_pool import ConnectionPool
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("psycopg_pool is required for PostgresSqlEngine") from exc

        self._pool = ConnectionPool(dsn, min_size=pool_min_size, max_size=pool_max_size, open=True)
        self._limit = limit
        self._timeout_ms = timeout_ms
        self._allowed_schemas = set(allowed_schemas) if allowed_schemas else None

    def close(self) -> None:
        self._pool.close()

    def query(self, query: str, params: Optional[Dict[str, Any]] = None, limit: Optional[int] = None) -> SqlResult:
        start = time.perf_counter()
        cleaned = _clean_query(query)
        _validate_read_only_query(cleaned, self._allowed_schemas)
        limited = _ensure_limit(cleaned, min(limit or self._limit, self._limit))

        with self._pool.connection() as conn:
            with conn.transaction():
                conn.execute("SET LOCAL statement_timeout = %s", (self._timeout_ms,))
                conn.execute("SET LOCAL TRANSACTION READ ONLY")
                cursor = conn.execute(limited, params or {})
                rows = cursor.fetchall()
                columns_meta = _columns_meta(cursor.description)

        elapsed_ms = int((time.perf_counter() - start) * 1000)
        dict_rows = _rows_to_dicts(rows, columns_meta)
        return SqlResult(
            rows=dict_rows,
            columns=columns_meta,
            stats={"row_count": len(dict_rows), "elapsed_ms": elapsed_ms},
        )

    def introspect_schema(self) -> List[Dict[str, Any]]:
        """Query information_schema for table/column metadata."""
        schema_filter = tuple(self._allowed_schemas) if self._allowed_schemas else ("public",)
        sql = """
            SELECT table_schema, table_name, column_name, data_type, is_nullable,
                   column_default, ordinal_position
            FROM information_schema.columns
            WHERE table_schema = ANY(%s)
            ORDER BY table_schema, table_name, ordinal_position
        """
        with self._pool.connection() as conn:
            with conn.transaction():
                conn.execute("SET LOCAL TRANSACTION READ ONLY")
                cursor = conn.execute(sql, (list(schema_filter),))
                rows = cursor.fetchall()

        tables: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            schema, table, col_name, dtype, nullable, default, pos = row
            key = f"{schema}.{table}"
            if key not in tables:
                tables[key] = {"table_name": key, "columns": [], "schema": schema}
            tables[key]["columns"].append({
                "name": col_name,
                "type": dtype,
                "nullable": nullable == "YES",
                "default": default,
                "position": pos,
            })
        return list(tables.values())


def sql_node(state: AgentState, services: Any) -> AgentState:
    sql_query = _resolve_sql(state.query, services)
    params = {"dialect": "postgres", "query": sql_query, "timeout_ms": 3000, "limit": 200}
    start_ms = now_ms()
    try:
        result = safe_tool_call("sql_query", services.sql_engine.query, sql_query)
        citations = _rows_to_citations(result.rows)
        text = _format_sql_result(result.rows, max_rows=5)
        state.evidence.append(
            EvidenceItem(
                kind="sql_rows",
                score=1.0,
                text=text,
                citations=citations,
                metadata=result.stats,
            )
        )
        record = ToolCallRecord(
            name="sql_query",
            args=params,
            ok=True,
            started_at_ms=start_ms,
            ended_at_ms=now_ms(),
            result_preview={"row_count": result.stats.get("row_count")},
        )
    except Exception as exc:
        record = ToolCallRecord(
            name="sql_query",
            args=params,
            ok=False,
            started_at_ms=start_ms,
            ended_at_ms=now_ms(),
            error=str(exc),
        )
    state.tool_calls.append(record)
    return state


def _resolve_sql(query: str, services: Any) -> str:
    if _looks_like_sql(query):
        return query
    llm = getattr(services, "llm", None)
    if not llm or not getattr(llm, "enabled", False):
        return query
    tables_desc = _describe_tables(services)
    if not tables_desc:
        return query
    try:
        return _llm_nl2sql(llm, query, tables_desc)
    except Exception as exc:
        logger.warning("NL2SQL failed, using raw query: %s", exc)
        return query


def _looks_like_sql(query: str) -> bool:
    return bool(re.match(r"^\s*select\b", query, flags=re.IGNORECASE))


_NL2SQL_SCHEMA = {
    "type": "object",
    "properties": {
        "sql": {"type": "string"},
        "explanation": {"type": "string"},
    },
    "required": ["sql", "explanation"],
    "additionalProperties": False,
}


def _llm_nl2sql(llm: Any, question: str, tables_desc: str) -> str:
    system = (
        "You are a SQL expert. Convert the user's natural language question into a valid "
        "SELECT SQL query. Only output read-only SELECT queries. Return JSON with "
        "fields: sql (the SQL string), explanation (brief explanation)."
    )
    user = f"Tables:\n{tables_desc}\n\nQuestion: {question}"
    result = llm.chat_json(system=system, user=user, schema=_NL2SQL_SCHEMA)
    sql = result.get("sql", "").strip()
    if not sql:
        raise ValueError("LLM returned empty SQL")
    return sql


def _describe_tables(services: Any) -> str:
    sql_engine = getattr(services, "sql_engine", None)

    # Try introspection first (works for both in-memory and Postgres)
    if sql_engine and hasattr(sql_engine, "introspect_schema"):
        try:
            tables = sql_engine.introspect_schema()
            if tables:
                lines = []
                for table in tables:
                    name = table.get("table_name", "?")
                    cols = table.get("columns", [])
                    col_desc = ", ".join(f"{c.get('name', '?')} {c.get('type', '?')}" for c in cols)
                    lines.append(f"  {name}({col_desc})")
                return "\n".join(lines)
        except Exception as exc:
            logger.debug("introspect_schema failed, falling back: %s", exc)

    # Fall back to in-memory repo tables
    repo = getattr(services, "repo", None)
    if not repo:
        return ""
    state = getattr(repo, "export_state", None)
    if not state:
        return ""
    exported = state()
    tables = exported.get("tables", [])
    if not tables:
        return ""
    lines = []
    for table in tables:
        name = table.get("name", "?")
        cols = table.get("columns", [])
        col_desc = ", ".join(f"{c.get('name', '?')} {c.get('type', '?')}" for c in cols)
        lines.append(f"  {name}({col_desc})")
    return "\n".join(lines)


def _rows_to_citations(rows: List[Dict[str, Any]]) -> List[Citation]:
    citations: List[Citation] = []
    for idx, _row in enumerate(rows[:30], start=1):
        citations.append(Citation(kind="row", row_ref=f"row:{idx}"))
    return citations


def _format_sql_result(rows: List[Dict[str, Any]], max_rows: int = 5) -> str:
    if not rows:
        return "SQL 返回 0 行。"
    preview = rows[:max_rows]
    lines = ["SQL 返回 {} 行，示例如下：".format(len(rows))]
    for row in preview:
        pairs = ", ".join(f"{key}={value}" for key, value in row.items())
        lines.append(pairs)
    return " ".join(lines)


def _clean_query(query: str) -> str:
    cleaned = query.strip()
    if cleaned.endswith(";"):
        cleaned = cleaned[:-1]
    if ";" in cleaned:
        raise ValueError("Multiple statements are not allowed")
    return cleaned


def _validate_read_only_query(query: str, allowed_schemas: Optional[Iterable[str]]) -> None:
    if not re.match(r"^select\b", query, flags=re.IGNORECASE):
        raise ValueError("Only SELECT queries are allowed")
    disallowed = (
        "insert",
        "update",
        "delete",
        "drop",
        "alter",
        "create",
        "truncate",
        "grant",
        "revoke",
        "comment",
        "vacuum",
        "call",
        "execute",
        "copy",
        "commit",
        "rollback",
    )
    if re.search(r"\b(" + "|".join(disallowed) + r")\b", query, flags=re.IGNORECASE):
        raise ValueError("Read-only queries only")
    if allowed_schemas is None:
        return
    for table in _extract_tables(query):
        if "." in table:
            schema = table.split(".")[0]
            if schema not in allowed_schemas:
                raise ValueError(f"Schema not allowed: {schema}")


def _extract_tables(query: str) -> List[str]:
    tables: List[str] = []
    for match in re.finditer(r"\b(from|join)\s+([\w\.]+)", query, flags=re.IGNORECASE):
        tables.append(match.group(2))
    return tables


def _ensure_limit(query: str, limit: int) -> str:
    match = re.search(r"\blimit\s+(\d+)\b", query, flags=re.IGNORECASE)
    if match:
        existing = int(match.group(1))
        if existing > limit:
            return re.sub(r"\blimit\s+\d+\b", f"LIMIT {limit}", query, flags=re.IGNORECASE)
        return query
    return f"{query} LIMIT {limit}"


def _columns_meta(description: Any) -> List[Dict[str, str]]:
    columns: List[Dict[str, str]] = []
    if not description:
        return columns
    for col in description:
        columns.append({"name": col.name, "type": str(col.type_code)})
    return columns


def _rows_to_dicts(rows: List[Any], columns: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    if not rows:
        return []
    col_names = [col["name"] for col in columns]
    return [dict(zip(col_names, row)) for row in rows]


def _parse_select(query: str):
    pattern = re.compile(
        r"^select\s+(?P<cols>[\w\*,\s]+)\s+from\s+(?P<table>[\w\-]+)(?:\s+where\s+(?P<where>.+))?$",
        re.IGNORECASE,
    )
    match = pattern.match(query)
    if not match:
        return None
    cols_raw = match.group("cols")
    columns = [col.strip() for col in cols_raw.split(",")]
    table_name = match.group("table")
    where_raw = match.group("where")
    where = _parse_where(where_raw) if where_raw else None
    return table_name, columns, where


def _parse_where(where_raw: str) -> Optional[Dict[str, Any]]:
    if not where_raw:
        return None
    pattern = re.compile(r"^(?P<col>[\w\-]+)\s*=\s*'(?P<val>[^']*)'$", re.IGNORECASE)
    match = pattern.match(where_raw.strip())
    if not match:
        return None
    return {"column": match.group("col"), "value": match.group("val")}


def _apply_where(rows: List[Dict[str, Any]], where: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not where:
        return list(rows)
    col = where["column"]
    val = where["value"]
    return [row for row in rows if str(row.get(col)) == val]

