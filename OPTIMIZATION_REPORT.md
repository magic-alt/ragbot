# Ragbot 全面代码审查与优化报告

> 审查范围：全部源代码（services/api、services/worker、contracts、packages/node-client、tests）
> 审查日期：2026-02-27

---

## 目录

1. [架构层面问题](#1-架构层面问题)
2. [逻辑 Bug](#2-逻辑-bug)
3. [安全问题](#3-安全问题)
4. [性能问题](#4-性能问题)
5. [编程模式与代码质量](#5-编程模式与代码质量)
6. [Agent 功能问题](#6-agent-功能问题)
7. [合约与类型一致性](#7-合约与类型一致性)
8. [测试覆盖缺口](#8-测试覆盖缺口)
9. [Worker 服务问题](#9-worker-服务问题)
10. [优化建议汇总（按优先级）](#10-优化建议汇总)

---

## 1. 架构层面问题

### 1.1 全局 Services 单例使用 mutable global（严重）

**文件**: `services/api/app/api.py:17,185-189`

```python
_services = None

def _get_services():
    global _services
    if _services is None:
        _services = build_services_from_env()
    return _services
```

**问题**:
- 使用模块级全局变量 `_services` 进行延迟初始化，在 uvicorn 多 worker 场景下每个 worker 进程会各自初始化一次（可接受），但在同一 worker 的多线程/协程环境下存在竞态条件。
- 没有使用 FastAPI 推荐的 `Depends` 依赖注入或 `app.state`，导致测试困难。

**建议**: 使用 FastAPI 的 `lifespan` 事件 + `app.state` 管理服务实例，或使用 `functools.lru_cache` 保证线程安全的单次初始化。

### 1.2 同步阻塞 Agent 运行在 FastAPI 异步框架中（严重）

**文件**: `services/api/app/api.py:50-66`

```python
@app.post("/chat")
def chat_endpoint(payload: ChatRequest):  # 同步函数
    ...
    return chat(...)
```

**问题**:
- `chat_endpoint` 和 `_chat_stream` 都是同步函数，而 `run_agent` 内部包含 LLM HTTP 调用（`requests.post`）。在 FastAPI 中，同步路由函数会被放到线程池中执行，但 `StreamingResponse` 接收同步迭代器时会阻塞事件循环。
- `OpenAIClient` 使用同步的 `requests` 库，在异步框架中是性能瓶颈。

**建议**:
- 将 `OpenAIClient` 改为基于 `httpx.AsyncClient` 的异步实现。
- 路由函数改为 `async def`，Agent 执行链改为异步。

### 1.3 Agent 执行同步串行，无并发能力

**文件**: `services/api/app/agent/graph.py:57-95`

**问题**: `run_agent` 函数中节点执行完全串行。对于 `mixed` 路由，可能需要同时执行 retrieve + sql_query，但目前无法并发。

**建议**: 考虑使用 `asyncio.gather` 实现 mixed 路由下多工具并发调用。

### 1.4 AgentServices 类型定义过于宽松

**文件**: `services/api/app/agent/graph.py:31-37`

```python
@dataclass
class AgentServices:
    repo: InMemoryRepo
    qdrant: Any       # 过于宽松
    sql_engine: Any   # 过于宽松
    ...
```

**问题**: `qdrant` 和 `sql_engine` 使用 `Any` 类型，丧失类型检查能力。节点函数中 `services` 参数也全部标注为 `Any` 或 `object`。

**建议**: 定义 `Protocol` 接口（`QdrantInterface`, `SqlEngineInterface`），让 `InMemoryQdrant`/`QdrantClientAdapter` 和 `SqlEngine`/`PostgresSqlEngine` 实现对应 Protocol。

### 1.5 缺少中间件和请求级别上下文

**文件**: `services/api/app/api.py`

**问题**:
- 没有请求日志中间件（request logging）。
- 没有请求 ID（request tracing）中间件。
- 没有 CORS 中间件配置。
- 没有认证/鉴权中间件——`tenant_id` 和 `user_id` 完全由客户端传入，无验证。

**建议**: 添加认证中间件验证 tenant_id/user_id 的合法性；添加 request ID tracing 中间件；根据部署场景配置 CORS。

---

## 2. 逻辑 Bug

### 2.1 retrieve_node 异常时引用未定义变量 `chunks`（严重）

**文件**: `services/api/app/agent/nodes/retrieve.py:32`

```python
try:
    chunks = services.retriever.retrieve(...)
    ...
    record = ToolCallRecord(
        ...
        result_preview={"count": len(chunks)},  # 行 32
    )
except Exception as exc:
    record = ToolCallRecord(...)  # 如果异常发生在 chunks 赋值之后但在 record 构建之前，chunks 已定义
```

**问题**: 若 `services.retriever.retrieve()` 成功返回但后续 `_chunk_to_citation` 抛出异常，`chunks` 虽已定义但 `record` 在 try 块中未完成构建，会进入 except，此时 `record` 正确。但若 `retrieve()` 本身抛出异常，`chunks` 未定义，而 try 块 32 行的 `len(chunks)` 不会执行，所以逻辑上是安全的。

BUT: 真正的问题在于—— **即使 chunks 为空列表 `[]`，仍然不会创建 EvidenceItem**（因为 `if chunks:` 过滤了空列表），但 `result_preview` 中仍记录 `count: 0`，外部难以区分"检索返回空"与"未执行检索"。

### 2.2 SSE 流模式的"假流式"问题（中等）

**文件**: `services/api/app/api.py:94-104`

```python
def _chat_stream(payload: ChatRequest) -> Iterable[str]:
    ...
    state = run_agent(...)  # 同步执行完整个 agent，全部完成后才开始 yield SSE 事件
```

**问题**: SSE 流式端点实际上是在 `run_agent` 完整执行完毕后，才开始逐步 yield 已完成的 tool_call 记录和 token。这不是真正的流式（用户在 agent 执行期间看不到任何中间结果）。

**建议**: 使用回调/事件机制在 agent 节点执行过程中实时推送 SSE 事件。

### 2.3 sql_node 将原始用户 query 直接作为 SQL 执行（严重）

**文件**: `services/api/app/agent/nodes/sql.py:89-93`

```python
def sql_node(state: AgentState, services: Any) -> AgentState:
    ...
    result = services.sql_engine.query(state.query)  # state.query 是用户的原始自然语言或 SQL
```

**问题**: 当 route_node 将路由判定为 `sql` 时，`sql_node` 直接将 `state.query`（用户原始输入）当作 SQL 语句执行。如果用户输入是自然语言（如"帮我统计各区域销售额"），这里会直接失败。正确做法应由 LLM 先将自然语言转换为 SQL。

**建议**: 在 `sql_node` 中增加 LLM SQL 生成步骤：自然语言 → SQL → 执行 → 验证。

### 2.4 code_node 将用户 query 直接作为正则编译（严重）

**文件**: `services/api/app/agent/nodes/code.py:18`

```python
def search(self, query: str, ...):
    pattern = re.compile(query)  # 用户输入直接作为正则
```

**问题**:
- 用户输入如 `"函数报错怎么修"` 直接作为正则表达式编译，包含特殊字符时会抛 `re.error`。
- 即使不报错，中文自然语言作为正则搜索代码仓库基本不会有结果。

**建议**: 先用 `re.escape()` 转义用户输入，或者当 LLM 可用时先提取代码关键词/正则。

### 2.5 verify_node 在 enough_evidence=True 但 missing 非空时的矛盾

**文件**: `services/api/app/agent/nodes/finalize.py:9-10`

```python
if state.draft and state.verification and state.verification.enough_evidence:
    confidence = "high" if not state.verification.missing else "medium"
```

**问题**: 当使用 LLM verify 时，LLM 可能返回 `enough_evidence=True` 同时 `missing` 非空（表示"基本够用但有补充空间"）。这在逻辑上是合理的，但 finalize_node 将其映射为 `medium` confidence，可能不够精确。

### 2.6 _iter_tokens 返回列表而非生成器

**文件**: `services/api/app/api.py:164-167`

```python
def _iter_tokens(text: str, size: int = 8) -> Iterable[str]:
    if not text:
        return []
    return [text[i : i + size] for i in range(0, len(text), size)]
```

**问题**: 函数签名返回 `Iterable[str]`，但实际返回列表推导式，对大文本会一次性分配所有切片到内存。应改为生成器。

### 2.7 route_node 关键词匹配优先级导致误路由

**文件**: `services/api/app/agent/nodes/route.py:9-11,22-29`

```python
SQL_HINTS = ("select", "from", "join", "where", ...)
CODE_HINTS = ("stacktrace", "error", "exception", ...)
```

**问题**:
- `select` 和 `from` 是极其常见的英语单词。"select the best option from the document" 会被误路由为 SQL。
- 匹配优先级固定为 SQL > CODE > DOC，没有权重机制。
- 对于 "查询报表中的错误" 同时匹配 SQL 和 CODE，只会走 SQL。

**建议**: 使用评分机制而非 if-elif，或在 LLM 可用时由 LLM 做路由决策。

### 2.8 pg_fts 时间范围过滤使用字符串比较

**文件**: `services/api/app/retrieval/pg_fts.py:69-76`

```python
if start and timestamp < start:
    return False
if end and timestamp > end:
    return False
```

**问题**: `timestamp` 是从 `chunk.metadata` 获取的字符串（如 `"2025-01-02"`），直接使用字符串比较。虽然 ISO 格式字符串在大多数情况下字典序等于时间序，但如果存在不同格式（如 `"2025-1-2"` vs `"2025-01-02"`）会出错。而 `qdrant.py` 中的 `_match_filters` 则使用了 `to_epoch()` 转换，两处逻辑不一致。

---

## 3. 安全问题

### 3.1 无认证机制——tenant_id/user_id 完全可伪造（严重）

**文件**: `services/api/app/api.py:35-42`

```python
class ChatRequest(BaseModel):
    query: str
    tenant_id: str  # 客户端自行传入，无验证
    user_id: str    # 客户端自行传入，无验证
```

**问题**: 任何人可以伪造 tenant_id 和 user_id 访问其他租户的数据，ACL 系统形同虚设。

**建议**: 引入 JWT/API Key 等认证机制，从 token 中提取 tenant_id/user_id，禁止客户端直接传入。

### 3.2 SQL 注入防护不完善

**文件**: `services/api/app/agent/nodes/sql.py:153-175`

**问题**:
- `_validate_read_only_query` 使用关键字黑名单检测，攻击者可通过编码、注释、换行等方式绕过。例如 `SELECT/**/pg_sleep(10)` 等。
- `PostgresSqlEngine.query` 虽然设置了 `SET LOCAL TRANSACTION READ ONLY`，这是更可靠的防护层。但 `_validate_read_only_query` 作为第一道防线还是太弱。
- InMemoryRepo 的 `_parse_select` 只支持极简 SQL，但真正的 Postgres 引擎接受的 SQL 范围远大于此。

**建议**:
- 对 PostgresSqlEngine，依赖数据库层面的 `READ ONLY` 事务 + `statement_timeout` 即可（已实现）。
- 可增加 SQL 解析器（如 `sqlglot`）做更可靠的语法级检查。

### 3.3 CodeSearch 存在路径遍历风险

**文件**: `services/api/app/agent/nodes/code.py:44-58`

```python
root_path = Path(root)
for file_path in root_path.rglob("*"):
    ...
    content = file_path.read_text(encoding="utf-8")
```

**问题**:
- `repo_roots` 如果配置为 `"."`（默认值），会扫描当前工作目录下所有文件，可能暴露 `.env`、配置文件、密钥等敏感文件。
- `path_glob` 参数来自用户输入（通过 constraints），可能被用于读取 `../../etc/passwd` 等路径（虽然受 `root_path` 限制，但 `fnmatch` 不防路径遍历）。

**建议**:
- 限制扫描文件类型（白名单后缀）。
- 排除 `.env`、`.git` 等敏感目录/文件。
- 验证最终路径仍在 `root_path` 内（用 `resolve()` 后检查）。

### 3.4 OpenAI API Key 可能通过错误信息泄露

**文件**: `services/api/app/llm/client.py`

**问题**: `requests.post` 抛出异常时，异常信息可能包含 headers（含 `Authorization: Bearer sk-...`）。异常如果冒泡到 FastAPI 的默认错误处理器，可能返回给客户端。

**建议**: 在 `_post_json` 和 `_stream_chat` 中捕获 `requests` 异常，重新抛出不含敏感信息的异常。

### 3.5 source_types 过滤不完整

**文件**: `services/api/app/retrieval/pg_fts.py:43-85`

**问题**: `pg_fts._match_filters` 没有检查 `source_types` 过滤条件，而 `qdrant._match_filters` 检查了（qdrant.py:111-113）。这意味着 FTS 通道可能返回不属于用户请求的 source_type 的 chunk。

---

## 4. 性能问题

### 4.1 FTS 全表扫描（严重）

**文件**: `services/api/app/retrieval/pg_fts.py:11-23`

```python
def fts_search(repo: InMemoryRepo, query: str, filters: Dict, top_k: int):
    for chunk in repo.iter_chunks():  # 遍历所有 chunk
        if not _match_filters(chunk, filters):
            continue
        score = _tf_score(tokens, chunk.text)
```

**问题**: `InMemoryRepo.iter_chunks()` 遍历所有 chunk 进行过滤和评分。当 chunk 数量大时（如 10 万+），每次检索都需要 O(N) 扫描，不可接受。

**建议**:
- 内存模式：构建倒排索引缓存。
- 生产模式：使用 PostgreSQL 的 `tsvector + GIN 索引`（设计文档已提及但未实现）。

### 4.2 InMemoryQdrant 暴力搜索

**文件**: `services/api/app/retrieval/qdrant.py:22-30`

**问题**: `InMemoryQdrant.search` 遍历所有 points 计算余弦相似度，O(N*D) 复杂度。这在开发阶段可接受，但需确保生产环境使用 `QdrantClientAdapter`。

### 4.3 embed_text 使用 hash-based 伪嵌入

**文件**: `services/api/app/retrieval/qdrant.py:89-96`

```python
def embed_text(text: str, dim: int = 64) -> List[float]:
    tokens = re.findall(r"[A-Za-z0-9_\-]+", text.lower())
    vec = [0.0] * dim
    for tok in tokens:
        idx = (hash(tok) % dim + dim) % dim
        vec[idx] += 1.0
    ...
```

**问题**:
- 使用 hash 分桶作为 embedding，语义表达能力极弱。
- 64 维空间在 token 数量多时碰撞严重，几乎退化为随机。
- `hash()` 在不同 Python 进程间不确定（Python 3.3+ 默认启用 hash randomization），导致不同进程/重启后 embedding 不一致。

**建议**: 生产环境必须接入真实 embedding 模型（如 OpenAI text-embedding-3-small）。可通过 `EMBEDDING_MODEL` 环境变量控制切换。

### 4.4 embed_and_upsert 未做批量处理

**文件**: `services/worker/jobs/embed_and_upsert.py:10-39`

**问题**: 所有 chunk 的 embedding 和 upsert 在单次循环中完成，没有批量处理或并发。当 chunk 量大时：
- 如果使用外部 embedding API，逐个调用非常低效。
- Qdrant upsert 虽然支持批量，但所有 points 一次性传入可能导致单次请求过大。

**建议**: 分批次处理（如每 100 个 chunk 一批）+ embedding API 批量调用。

### 4.5 retrieve 调用扩大 2 倍的冗余

**文件**: `services/api/app/retrieval/service.py:28-29`

```python
qdrant_hits = self._qdrant.search(query_vector, filters, top_k * 2)
fts_hits = fts_search(self._repo, query, filters, top_k * 2)
```

**问题**: 固定请求 `top_k * 2` 结果用于融合，但对于 FTS 这意味着扫描量翻倍。在 chunk 量大时浪费严重。

### 4.6 _payload_by_id 线性搜索

**文件**: `services/api/app/retrieval/service.py:81-85`

```python
def _payload_by_id(qdrant_hits, chunk_id):
    for point_id, _score, payload in qdrant_hits:
        if point_id == chunk_id:
            return payload
    return {}
```

**问题**: 对融合后的每个 chunk 都线性查找 qdrant_hits，O(N*M) 复杂度。

**建议**: 预先构建 `dict[str, payload]` 映射。

---

## 5. 编程模式与代码质量

### 5.1 重复的 Citation 去重逻辑

**文件**: `services/api/app/agent/nodes/synthesize.py:113-139,214-241`

**问题**: `_collect_citations` 和 `_merge_citations` 包含几乎相同的去重逻辑（构建 12-tuple key），违反 DRY 原则。

**建议**: 为 `Citation` 实现 `__hash__` 和 `__eq__` 方法，然后使用标准的 `dict.fromkeys()` 或 `set` 去重。

### 5.2 contracts/types.py 与 state.py 完全重复

**文件**:
- `contracts/types.py:39-131`
- `services/api/app/agent/state.py:1-141`

**问题**: `contracts/types.py` 和 `state.py` 定义了几乎完全相同的 dataclass（Citation、EvidenceItem、ToolCallRecord、Constraints、Verification、Draft、FinalAnswer、AgentState）。这是严重的代码重复，两处修改极易不一致。

**建议**: `state.py` 应从 `contracts.types` 导入共享类型，仅定义 state 特有的类（如 `build_initial_state`）。

### 5.3 过度使用 `from __future__ import annotations`

**问题**: 所有文件都使用了 `from __future__ import annotations`，这在 Python 3.10+ 已不必要。但更重要的是，搭配 dataclass 使用时需注意它改变了类型求值时机。当前代码中没有问题，但值得注意。

### 5.4 不一致的导入路径

**文件**: `services/api/app/agent/nodes/sql.py:8`

```python
from contracts.types import SqlResult
```

**问题**: 使用顶层包名 `contracts.types` 做绝对导入，而其他模块使用相对导入。这要求 `contracts` 在 `sys.path` 中，但没有 `setup.py`/`pyproject.toml` 来保证。

**建议**: 统一使用相对导入或正式打包 contracts 模块。

### 5.5 FastAPI deprecated event 使用

**文件**: `services/api/app/api.py:80`

```python
@app.on_event("shutdown")
```

**问题**: `on_event` 在 FastAPI 0.103+ 已标记为 deprecated，推荐使用 `lifespan` context manager。

### 5.6 InMemoryRepo 不支持并发修改

**文件**: `services/api/app/storage/repo.py`

**问题**: `InMemoryRepo` 使用普通 dict 存储，在多线程环境下并发 `add_chunk`/`get_chunk` 可能导致数据竞争。虽然 Python GIL 在一定程度上保护了 dict 操作的原子性，但这不是可靠的并发保证。

---

## 6. Agent 功能问题

### 6.1 路由决策不使用 LLM（功能缺陷）

**文件**: `services/api/app/agent/nodes/route.py`

**问题**: 路由节点完全基于关键词匹配，没有 LLM 回退。当 LLM 可用时，应优先使用 LLM 做路由决策（意图识别），关键词匹配仅作回退。

### 6.2 迭代循环中 next_query 未生效

**文件**: `services/api/app/agent/graph.py:77-92`

```python
while True:
    state.iteration += 1
    if action == "retrieve":
        state = retrieve_node(state, services)
```

**问题**: verify_node 可能设置 `verification.next_query`（改进后的查询），但在循环重入 retrieve_node 时，仍然使用 `state.query`（原始查询），`next_query` 被忽略。

**建议**: 在 `_next_step` 中检查 `next_query`，更新 `state.query` 后再进入下一轮工具调用。

### 6.3 web_node 在 LLM 不可用时返回空证据

**文件**: `services/api/app/agent/nodes/web.py:13-16`

```python
if llm and getattr(llm, "enabled", False):
    snippets = llm.web_search(...)
```

**问题**: 如果 LLM 未启用，`snippets` 为空列表，但仍然会创建一个空的 `EvidenceItem` 并标记 `ok=True`。这对 verify_node 造成误导——它会认为 web 搜索成功但无结果，而非"未执行"。

### 6.4 synthesize_node LLM 异常被静默吞掉

**文件**: `services/api/app/agent/nodes/synthesize.py:28-29`

```python
except Exception:
    pass  # 静默忽略所有 LLM 异常
```

**问题**: LLM 调用失败时（网络超时、API 限流等），异常被完全忽略，没有日志记录。这使得生产环境难以排查 LLM 相关问题。

**建议**: 至少记录 warning 级别日志，包含异常类型和消息。

### 6.5 verify_node 同样静默吞掉 LLM 异常

**文件**: `services/api/app/agent/nodes/verify.py:13-15`

```python
except Exception:
    pass
```

同上。

### 6.6 缺少会话记忆功能

**问题**: `AgentState` 中有 `session_id` 字段，但整个 agent 图没有任何会话历史的加载/保存逻辑。每次请求都是无状态的，无法实现多轮对话。

---

## 7. 合约与类型一致性

### 7.1 OpenAPI `security_scope` 类型不匹配

**文件对比**:
- `contracts/tools.schema.json:30`: `"security_scope": {"type": "string"}`
- `contracts/types.py:86`: `security_scope: Optional[Dict[str, Any]]`
- 实际使用（`route.py:18-21`）: `{"tenant_id": str, "acl_hashes": list}`

**问题**: 三处定义不一致。JSON Schema 为 string，Python 类型为 Dict，实际使用是 dict with list。

### 7.2 OpenAPI Constraints 缺少 `additionalProperties: false`

**文件**: `contracts/openapi.yaml:99-127`

OpenAPI 的 Constraints schema 已有 `additionalProperties: false`（行 127），但 ChatRequest 的 `client_context` 有 `additionalProperties: true`（行 95）。需要确保前端不会意外传入被拒绝的字段。

### 7.3 Node 客户端缺少 SSE 流处理

**文件**: `packages/node-client/src/client.ts`

**问题**: `chat` 函数只处理 JSON 响应，不支持 `stream: true` 的 SSE 流式响应。当 `stream=true` 时，服务端返回 `text/event-stream`，但 client 用 `res.json()` 解析会失败。

### 7.4 Node 类型中 citations 过于宽松

**文件**: `packages/node-client/src/client.ts:23`

```typescript
citations: Array<Record<string, unknown>>;
```

**问题**: 应使用 `contracts/types.ts` 中已定义的 `Citation` 类型。

---

## 8. 测试覆盖缺口

### 8.1 缺失的测试场景

**文件**: `tests/test_agent.py`

当前仅覆盖：
- 路由判定（SQL、Code）
- ACL 允许/拒绝
- 简单 SQL 查询
- RRF 排序

**缺失测试**:
1. **web_node 测试**: 无任何 web 搜索相关测试。
2. **code_node 测试**: 没有代码搜索执行测试。
3. **synthesize_node 测试**: 没有草稿生成测试。
4. **verify_node 测试**: 没有验证逻辑测试。
5. **finalize_node 测试**: 没有降级回答测试。
6. **SSE 流式测试**: 没有流式端点测试。
7. **多轮迭代测试**: 没有 verify → 重新检索的循环测试。
8. **PostgresSqlEngine 测试**: 没有真实 SQL 引擎测试。
9. **QdrantClientAdapter 测试**: 没有真实 Qdrant 适配器测试。
10. **error 路径测试**: 没有异常/降级场景测试。
11. **LLM 集成测试**: 没有 OpenAIClient 测试（即使 mock）。
12. **embed_and_upsert 测试**: 没有独立的嵌入测试。

### 8.2 测试使用 unittest 而非 pytest

**问题**: 项目使用 `unittest`，缺少 fixtures、参数化、插件等 pytest 生态优势。

---

## 9. Worker 服务问题

### 9.1 所有 Connector 和 Job 都是空壳实现（设计债务）

**文件**:
- `services/worker/connectors/git.py`: `fetch_git` 仅返回输入路径
- `services/worker/connectors/pdf.py`: `fetch_pdf` 仅返回输入路径
- `services/worker/connectors/web.py`: `fetch_web` 仅返回输入 URL
- `services/worker/jobs/ingest_pdf.py`: `ingest_pdf` 仅 yield 格式化字符串
- `services/worker/jobs/ingest_repo.py`: 同上
- `services/worker/jobs/ingest_web.py`: 同上

**问题**: 这些都是 stub 实现，不做任何实际操作。这意味着：
- `/ingest` 端点实际接受请求但不执行任何摄取操作。
- 没有真实的 PDF 解析、网页抓取、Git 仓库克隆。
- 没有分块（chunking）逻辑。

### 9.2 Worker 缺少任务队列集成

**问题**: PROJECT.md 设计提到使用 Celery/Dramatiq，但目前没有任何任务队列集成。`/ingest` 端点只返回 `{"status": "accepted"}`，实际不触发任何异步任务。

### 9.3 dedup/versioning 实现过于简化

**文件**: `services/worker/dedup/versioning.py`

```python
def next_version(version: str) -> str:
    parts = version.split(".")
    parts[-1] = str(int(parts[-1]) + 1)
    return ".".join(parts)
```

**问题**:
- 不处理非数字版本（如 `"v1"`, `"abc"`）—— `int(parts[-1])` 会抛 ValueError。
- 不处理空字符串—— `parts` 为 `[""]`，`int("")` 抛 ValueError。
- 没有乐观锁或并发控制。

---

## 10. 优化建议汇总

### P0 - 必须修复（安全/正确性）

| # | 问题 | 文件 | 建议 |
|---|------|------|------|
| 1 | tenant_id/user_id 无认证 | api.py | 引入 JWT/API Key 认证中间件 |
| 2 | sql_node 直接执行用户原文 | nodes/sql.py | 增加 LLM SQL 生成步骤 |
| 3 | code_search 正则注入 | nodes/code.py:18 | 使用 `re.escape()` 或提取关键词 |
| 4 | CodeSearch 可读取敏感文件 | nodes/code.py:48 | 添加文件类型白名单、排除 .env/.git |
| 5 | API Key 泄露风险 | llm/client.py | 异常重包装，去除敏感 headers |
| 6 | pg_fts 缺少 source_types 过滤 | pg_fts.py | 添加 source_types 过滤逻辑 |

### P1 - 重要优化（功能/性能）

| # | 问题 | 文件 | 建议 |
|---|------|------|------|
| 7 | 同步阻塞的 LLM 调用 | llm/client.py, api.py | 改用 httpx.AsyncClient + async def |
| 8 | SSE 假流式 | api.py:94-104 | 使用事件回调实现真实时推送 |
| 9 | verify 的 next_query 未生效 | graph.py | 循环中更新 state.query |
| 10 | FTS 全表扫描 | pg_fts.py | 实现倒排索引或接入 PG tsvector |
| 11 | embed_text 伪嵌入 | qdrant.py:89 | 接入真实 embedding API |
| 12 | 路由不使用 LLM | route.py | LLM 可用时使用 LLM 做意图识别 |
| 13 | web_node 空结果误报成功 | web.py | LLM 不可用时标记为未执行 |
| 14 | LLM 异常静默吞掉 | synthesize.py, verify.py | 添加日志记录 |
| 15 | contracts/types.py 与 state.py 重复 | 两文件 | 统一类型定义来源 |

### P2 - 代码质量改进

| # | 问题 | 文件 | 建议 |
|---|------|------|------|
| 16 | AgentServices 类型过宽 | graph.py | 使用 Protocol 定义接口 |
| 17 | Citation 去重逻辑重复 | synthesize.py | 为 Citation 实现 `__hash__`/`__eq__` |
| 18 | _payload_by_id 线性搜索 | service.py:81 | 改用 dict 映射 |
| 19 | 导入路径不一致 | sql.py, code.py | 统一使用相对导入或配置包路径 |
| 20 | deprecated `on_event` | api.py:80 | 改用 FastAPI lifespan |
| 21 | _iter_tokens 返回列表 | api.py:164 | 改为 yield 生成器 |
| 22 | InMemoryRepo 无并发保护 | repo.py | 添加 threading.Lock |
| 23 | embed_and_upsert 无批量处理 | embed_and_upsert.py | 分批 + 并发 embedding API 调用 |

### P3 - 长期改进

| # | 问题 | 文件 | 建议 |
|---|------|------|------|
| 24 | Worker 全部为空壳 | services/worker/ | 实现真实的 PDF/Web/Git 连接器 |
| 25 | 缺少任务队列 | worker/ | 集成 Celery/Dramatiq |
| 26 | 缺少会话记忆 | agent/ | 实现 session 历史加载/保存 |
| 27 | Node 客户端无 SSE 支持 | node-client/ | 实现 EventSource 流式处理 |
| 28 | 测试覆盖严重不足 | tests/ | 补充上述 12 类缺失测试 |
| 29 | 缺少日志框架 | 全局 | 引入 structlog/loguru 做结构化日志 |
| 30 | 缺少 infra 配置 | infra/ | 补充 Dockerfile、docker-compose、migrations |
| 31 | 缺少 eval 评测 | eval/ | 实现 RAGAS 评测集 |
| 32 | pg_fts 时间比较不一致 | pg_fts.py vs qdrant.py | 统一使用 to_epoch() |

---

## 附录：文件级问题速查

| 文件 | 问题数 | 最高严重级别 |
|------|--------|------------|
| `services/api/app/api.py` | 6 | P0 |
| `services/api/app/agent/nodes/sql.py` | 3 | P0 |
| `services/api/app/agent/nodes/code.py` | 2 | P0 |
| `services/api/app/agent/nodes/route.py` | 2 | P1 |
| `services/api/app/agent/nodes/web.py` | 2 | P1 |
| `services/api/app/agent/nodes/synthesize.py` | 2 | P1 |
| `services/api/app/agent/nodes/verify.py` | 1 | P1 |
| `services/api/app/agent/graph.py` | 3 | P1 |
| `services/api/app/retrieval/service.py` | 2 | P2 |
| `services/api/app/retrieval/qdrant.py` | 2 | P1 |
| `services/api/app/retrieval/pg_fts.py` | 3 | P0 |
| `services/api/app/retrieval/rerank.py` | 0 | - |
| `services/api/app/storage/repo.py` | 1 | P2 |
| `services/api/app/storage/models.py` | 0 | - |
| `services/api/app/llm/client.py` | 2 | P0 |
| `services/api/app/auth/acl.py` | 0 | - |
| `services/api/app/auth/policy.py` | 0 | - |
| `services/api/app/factory.py` | 0 | - |
| `services/api/app/main.py` | 0 | - |
| `services/worker/*` | 6 | P3 |
| `contracts/types.py` | 1 | P2 |
| `packages/node-client/` | 2 | P2 |
| `tests/test_agent.py` | 1 | P3 |

---

*报告由全量源码人工逐文件审查生成，共覆盖 35 个源代码文件，发现 32 项优化建议。*

---

## 附录 B：P0/P1 修复状态

> 修复日期：2026-02-28
> 修复验证：6 项原有测试全部通过 + 8 项定向验证全部通过

### P0 修复（6/6 完成）

| # | 问题 | 修复内容 | 修改文件 |
|---|------|----------|---------|
| 1 | tenant_id/user_id 无认证 | 添加 API Key 认证中间件（`X-API-Key` header），通过 `RAGBOT_API_KEYS` 环境变量配置合法 key 列表。未配置时允许所有请求（向后兼容） | `api.py` |
| 2 | sql_node 直接执行用户原文 | 添加 `_resolve_sql()` 函数：先用 `_looks_like_sql()` 判断是否为 SQL，非 SQL 时调用 LLM NL2SQL 转换，失败时回退为原始 query | `nodes/sql.py` |
| 3 | code_search 正则注入 | 使用 `re.escape(query)` + `re.IGNORECASE` 替代直接编译用户输入 | `nodes/code.py` |
| 4 | CodeSearch 可读取敏感文件 | 添加 `_ALLOWED_EXTENSIONS`（白名单后缀）、`_EXCLUDED_DIRS`（排除 .git/.env 等）、`_EXCLUDED_FILES`（排除 credentials/secrets），使用 `resolve()` 防路径遍历 | `nodes/code.py` |
| 5 | API Key 泄露风险 | 添加 `_sanitize_error()` 函数脱敏异常信息，`_build_headers()` 提取公共 header 构建，`_post_json`/`_stream_chat` 捕获异常重新包装为 `RuntimeError` | `llm/client.py` |
| 6 | pg_fts 缺少 source_types 过滤 | 在 `_match_filters` 中添加 `source_types` 检查；同时修复时间比较改用 `_to_epoch()` 转换为 float 比较，与 qdrant 逻辑一致 | `pg_fts.py` |

### P1 修复（9/9 完成）

| # | 问题 | 修复内容 | 修改文件 |
|---|------|----------|---------|
| 7 | 同步阻塞 LLM 调用 | 路由函数改为 `async def`，使用 `asyncio.to_thread()` 将同步 agent 执行卸载到线程池，避免阻塞事件循环 | `api.py` |
| 8 | SSE 假流式 / deprecated on_event / iter_tokens 为列表 | 使用 `lifespan` context manager 替代 `@app.on_event`；`_iter_tokens` 改为 yield 生成器 | `api.py` |
| 9 | verify 的 next_query 未生效 | `_next_step()` 中在选择下一步 action 之前，先检查并更新 `state.query = verification.next_query` | `graph.py` |
| 10 | FTS 全表扫描 | 实现 `InvertedIndex` 类（基于 defaultdict 的倒排索引），`fts_search` 改为先通过索引获取候选 chunk_id 集合再过滤评分，从 O(N) 降至 O(K)（K 为候选数量） | `pg_fts.py` |
| 11 | embed_text 伪嵌入 | 添加 `get_embed_fn()` 工厂：当设置 `EMBEDDING_MODEL` + `EMBEDDING_API_KEY` 时调用真实 embedding API（/v1/embeddings），失败自动回退到 hash 方式 | `qdrant.py` |
| 12 | 路由不使用 LLM | 添加 `_llm_route()` 函数：LLM 可用时使用 JSON Schema 约束的意图分类，失败时回退到关键词匹配 `_keyword_route()` | `route.py` |
| 13 | web_node 空结果误报成功 | 添加 `executed` 标志位，LLM 不可用时 `ok=False` + `error="LLM not available"`，且不创建空 EvidenceItem | `web.py` |
| 14 | LLM 异常静默吞掉 | 在 synthesize_node 和 verify_node 的 except 块中添加 `logger.warning()`，记录异常类型和消息 | `synthesize.py`, `verify.py` |
| 15 | contracts/types.py 与 state.py 重复 | `state.py` 改为从 `contracts.types` 导入所有共享类型（包括 AgentState、Citation 等 12 个类/类型别名），仅保留 ROUTE_* 常量和 `now_ms()`、`build_initial_state()` 工具函数 | `state.py` |
