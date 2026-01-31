from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Optional

from contracts.types import AgentState, SqlResult
from ...storage.repo import InMemoryRepo


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


def sql_node(state: AgentState, services: Any) -> AgentState:
    params = {"dialect": "postgres", "query": state.query}
    state.add_tool_call("sql_query", params)
    try:
        result = services.sql_engine.query(state.query)
    except ValueError as exc:
        state.add_evidence(
            "sql_error",
            {"error": str(exc)},
            ["sql:validation"],
        )
        return state
    payload = {"rows": result.rows, "columns": result.columns, "stats": result.stats}
    state.add_evidence("rows", payload, ["sql:result"])
    return state


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

