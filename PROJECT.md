下面给你一套可直接落地的 **“Postgres 事务/JOIN + Qdrant 向量检索”** 的 **Agentic RAG** 设计：包含工程目录、LangGraph 状态机（节点/边）、工具 schema、以及 Qdrant payload / Postgres 表结构与过滤字段设计；同时兼容 **Python 为主 + Node 侧调用/扩展**。

---

## 总体架构

**核心思路：**

* PostgreSQL 负责：

  * 文档/Chunk 元数据、版本、权限 ACL、数据源连接信息、任务队列状态
  * 结构化数据查询（SQL 工具）、复杂 JOIN、审计日志、会话与记忆（可选）
  * （可选）关键词检索：Postgres Full-Text Search（tsvector）当 BM25 通道
* Qdrant 负责：

  * chunk embedding + payload（用于过滤/追溯）
  * 向量召回（可加稀疏/混合，但你也可以先用 Postgres FTS 做关键词通道）

**Agent 层：**

* 用 LangGraph 编排：router → 多工具调用 → 验证/补检索 → 最终回答（强制引用证据）

**服务边界建议：**

* `api-gateway`（Python）统一对外：/chat、/ingest、/admin
* `ingestion-worker`（Python）异步摄取 PDF/网页/代码仓库
* `retrieval-service`（Python）封装 Qdrant + Postgres FTS + rerank
* Node 侧：做“调用方 + 可插拔工具”（或做前端/工作流），通过 HTTP 工具协议接入

---

## 可复制的工程目录结构（monorepo）

```
rag-agent/
  contracts/                  # 跨语言共享契约：OpenAPI + JSON Schema + types
    openapi.yaml
    tools.schema.json
    types.ts
    types.py
  services/
    api/                       # Python API 网关（FastAPI）
      app/
        main.py
        routes/
          chat.py
          ingest.py
          admin.py
        agent/                 # LangGraph 图 + 节点实现
          graph.py
          state.py
          nodes/
            route.py
            retrieve.py
            sql.py
            code.py
            web.py
            synthesize.py
            verify.py
            finalize.py
        retrieval/             # 检索封装：Qdrant + FTS + rerank
          qdrant.py
          pg_fts.py
          rerank.py
        storage/               # Postgres DAO / repository
          models.py
          repo.py
        auth/                  # ACL/tenant/filter 计算
          acl.py
          policy.py
    worker/                    # Python 摄取/索引任务（Celery/Dramatiq）
      jobs/
        ingest_pdf.py
        ingest_web.py
        ingest_repo.py
        embed_and_upsert.py
      connectors/
        pdf.py
        web.py
        git.py
      dedup/
        hashing.py
        versioning.py
  packages/
    node-client/               # Node SDK：调用 /chat；也可实现自定义工具
      src/
        index.ts
        tools.ts
        client.ts
  infra/
    docker/
    migrations/                # Postgres migrations（sql / alembic）
    qdrant/
    helm/                      # 可选
  eval/
    datasets/
    ragas/                     # 评测脚本、回归
  README.md
```

> 关键点：`contracts/` 是 Python/Node 一致性的根：工具 schema、事件流格式、引用格式、metadata 字段名都在这里定死。

---

## LangGraph 状态机设计（节点/边）

### 状态 State（最小但够用）

* `query`: 用户问题
* `tenant_id`, `user_id`
* `constraints`: {time_range?, sources?, tags?, repo?, db?}
* `route`: 选择的策略（doc_rag / sql / code / mixed / web_fallback）
* `tool_calls`: 已执行工具记录（用于审计/调试）
* `evidence`: 证据列表（chunks/rows/code_snippets + citations）
* `draft`: 草稿
* `verdict`: 验证结果（enough_evidence? missing_what?）
* `final`: 最终回答（含引用）

### 节点 Nodes

1. **route_node（路由/规划）**

   * 输出：`route` + 初始 `constraints` + 候选工具序列
   * 规则：

     * 需要数值/报表/业务数据 → 优先 SQL 工具
     * 需要定位实现/函数/报错 → 优先 code_search
     * 需要文档解释/规范 → RAG（向量 + 关键词）
     * 不确定 → mixed（先轻量检索，再决定是否 SQL/代码）

2. **retrieve_node（文档检索）**

   * 调用 `retrieve()`：Qdrant 召回 + Postgres FTS 召回 + 融合 + rerank
   * 产出：`evidence += doc_chunks`

3. **sql_node（结构化查询）**

   * 调用 `sql_query()`：只允许白名单 schema、强制 LIMIT、超时、只读事务
   * 产出：`evidence += table_rows`（并带 query_hash/时间）

4. **code_node（代码检索）**

   * 调用 `code_search()`：ripgrep/索引服务（支持路径过滤、仓库/分支）
   * 产出：`evidence += code_snippets`（path + line_range + commit-ish）

5. **web_node（外部补充，可选）**

   * 内部证据不足时 fallback：`web_search()` / `web_fetch()`
   * 产出：`evidence += web_snippets`

6. **synthesize_node（基于证据写草稿）**

   * 强制：每个关键结论必须能映射到 evidence citation

7. **verify_node（证据充分性检查）**

   * 判断：是否有“可回答所需的最小证据集合”
   * 如果不足：生成“缺口查询”-> 回到 retrieve/sql/code（最多 N 轮）

8. **finalize_node（最终答案）**

   * 输出：结构化回答 + 引用列表 +（可选）可执行 next steps

### 边 Edges（条件流）

* route → retrieve / sql / code / mixed（mixed 先 retrieve，再由 verifier 决定是否 sql/code）
* synthesize → verify
* verify → finalize（enough_evidence = true）
* verify → retrieve/sql/code（need_more = true，带 reformulated query）
* 超过轮数/工具失败 → finalize（降级：说明缺口 + 给出下一步需要的数据/权限）

---

## 工具（Tool) Schema 建议（跨语言一致）

下面是**建议你固定下来的工具集合**（JSON Schema 风格；你可以直接放进 `contracts/tools.schema.json`）：

### 1) retrieve（RAG 检索）

**入参**

* `query: string`
* `top_k: number`（默认 20）
* `filters`：

  * `tenant_id`
  * `source_types: ["pdf"|"web"|"repo"|"db_doc"]`
  * `doc_ids?`, `tags?`
  * `path_prefix?`（repo）
  * `url_prefix?`（web）
  * `time_range?`（ingested_at / doc_updated_at）
  * `security_scope`（由后端计算/注入，前端不可伪造）
    **出参**
* `chunks: [{chunk_id, doc_id, text, score, citations, metadata}]`

### 2) sql_query（结构化数据）

**入参**

* `dialect: "postgres"`
* `query: string`
* `params?: object`
* `limit?: number`（硬上限，例如 200）
* `timeout_ms?: number`（例如 3000）
  **出参**
* `rows: array<object>`
* `columns: [{name,type}]`
* `stats: {row_count, elapsed_ms}`

### 3) code_search（代码）

**入参**

* `query: string`（支持关键词/正则）
* `repo: string`
* `ref?: string`（branch/commit/tag）
* `path_glob?: string`
* `max_hits?: number`
  **出参**
* `snippets: [{path, ref, line_start, line_end, content}]`

### 4) web_search / web_fetch（可选）

* `web_search(query, recency_days?, domains?)`
* `web_fetch(url)`

### 5) respond（最终输出，或由模型直接输出）

* `answer: string`
* `citations: [...]`
* `confidence: "high"|"medium"|"low"`
* `followups?: [...]`

> 重要：**security_scope 由服务端根据 user/tenant/ACL 计算并注入**，Node/Python 客户端都不允许直接传任意 ACL。

---

## Qdrant Payload 与过滤字段设计（强烈建议照这个定）

### Collection：`rag_chunks`

**point**

* `id = chunk_id`（UUID 或 snowflake）
* `vector = embedding`
* `payload`（用于过滤/审计/引用）：

建议 payload 字段（全部可索引过滤）：

* `tenant_id`
* `source_type`: `"pdf" | "web" | "repo" | "db_doc"`
* `doc_id`
* `chunk_index`
* `title?`
* `path?`（repo 文件路径）
* `url?`（网页 URL）
* `page?`（PDF 页）
* `section?`
* `language?`
* `ingested_at`（ISO time）
* `doc_updated_at?`
* `version`（文档版本号或 hash）
* `checksum`（chunk 内容 hash，用于去重）
* `acl_hash`（权限集合摘要，或 policy id）
* `tags: string[]`
* `embedding_model`（便于迁移/回滚）

> 过滤最常用的是：`tenant_id + source_type + (doc_id/path/url) + time_range + acl_hash`
> 这样路由节点能精准缩小召回范围，也便于多租户隔离。

---

## Postgres 表设计（最关键的 4 张表）

### 1) `documents`

* `doc_id (pk)`
* `tenant_id`
* `source_type`
* `title`
* `uri`（file:// / https:// / repo://）
* `version`
* `doc_updated_at`
* `ingested_at`
* `tags jsonb`
* `acl_policy_id`（或直接 acl 结构）
* `status`（active/deleted）

### 2) `chunks`

* `chunk_id (pk)`
* `doc_id (fk)`
* `tenant_id`
* `chunk_index`
* `text`（可存可不存；至少存摘要/preview 以便审计）
* `page/section/path/url`（冗余一份便于 JOIN）
* `checksum`
* `qdrant_point_id`（若不同于 chunk_id）
* `created_at`

### 3) `acl_policies` / `doc_acl`

* `acl_policy_id`
* `tenant_id`
* `rules jsonb`（用户/组/角色/标签/行级策略）
* `policy_hash`（对应 qdrant 的 acl_hash）

### 4) `ingestion_jobs`

* `job_id`
* `tenant_id`
* `source_type`
* `source_config jsonb`（抓取规则、repo ref、db conn）
* `status`（queued/running/failed/done）
* `stats jsonb`（文档数、chunk 数、耗时、失败原因）

**可选（关键词检索通道）**

* 在 `chunks` 上加：

  * `tsv tsvector generated always as (...) stored`
  * GIN 索引用于 FTS
* 检索时：FTS TopK + Qdrant TopK 融合（RRF 或加权）

---

## 检索融合与重排（给 Agent “稳证据”）

**推荐流程：**

1. Qdrant：向量 TopK（例如 30）
2. Postgres FTS：关键词 TopK（例如 30）
3. 融合：RRF（Reciprocal Rank Fusion）或简单加权（语义 0.6 / 关键词 0.4）
4. Rerank：cross-encoder 或轻量 reranker（可选但强烈建议）
5. 输出给 LLM：只给 top 8~12 个 chunk，确保每个 chunk 带 citation

---

## Node 支持方式（两种都给你）

### 方案 1：Node 作为调用方（推荐先做）

* Node 通过 `/chat` 调用 Python API（SSE/WS 流式）
* Node 也可以实现“外部工具服务”，由 Python agent 通过 HTTP 调用（tool proxy）

优点：最快落地、Python 侧保持一致性。

### 方案 2：Node 实现自定义工具（例如内部系统/工单/CI）

* 定一个工具网关协议：`POST /tools/{name}` 入参/出参严格遵循 `contracts/tools.schema.json`
* Python agent 把 tool call 转发给 Node 工具服务（带签名与 tenant/user 上下文）

优点：Node 团队可以独立交付工具，不动 Python agent 内核。

---

## 安全与治理（别省，否则很快踩坑）

* **SQL 工具必须只读**：只读事务 + schema 白名单 + 自动加 LIMIT + 超时 + 禁止危险语句
* **ACL 过滤必须前置**：检索前就按 `acl_hash/security_scope` 过滤，避免“先召回再遮罩”泄漏
* **引用强制**：final answer 的每个关键结论必须能追溯到 `chunk_id/path/page/row_id`
* **审计日志**：记录 tool_calls、用到的 doc_id/chunk_id、sql_hash、响应版本

---

## 你可以直接照抄的“最小落地顺序”

1. 先做 `documents/chunks` + Qdrant upsert（只接 PDF）
2. 上 LangGraph：route → retrieve → synthesize → verify → finalize
3. 接 Postgres FTS 融合（提升术语/代码符号命中）
4. 接 `sql_query`（结构化问题闭环）
5. 接 `code_search`（repo 问题闭环）
6. 加评测集与回归（至少 30 条真实问题）

---

