# Ragbot 案例运行说明

> 本文档说明 `examples/example_case.py` 的运行方式、执行过程和预期结果。

---

## 1. 前置条件

| 依赖 | 说明 |
|------|------|
| Python >= 3.10 | 运行环境 |
| PyPDF2 >= 3.0 | PDF 文本提取（`pip install PyPDF2`） |
| 其他核心依赖 | FastAPI, requests, pydantic（见 `requirements.txt`） |
| **无需外部服务** | 案例使用全内存模式，不需要 Qdrant、PostgreSQL 或 OpenAI API Key |

## 2. 运行方式

```bash
# 从项目根目录运行（推荐）
cd e:\work\Project\ragbot
python examples/example_case.py

# 或从任意位置运行（脚本自动修正 sys.path）
python e:\work\Project\ragbot\examples\example_case.py
```

## 3. 案例做了什么

案例程序演示了 ragbot 的完整 RAG Agent 管线，包含**数据摄取**和**多路由查询**两个阶段。

### 阶段一：数据准备

```
Step 1  初始化服务
   └─ 创建全内存服务实例（InMemoryRepo、InMemoryQdrant、Retriever、SqlEngine、CodeSearch）
   └─ 覆盖 CodeSearch 为内存模式，注入一个示例 Python 文件

Step 2  创建 ACL 策略
   └─ 创建租户 "demo-tenant" 的访问控制策略
   └─ 仅允许 "demo-user" 访问

Step 3  摄取 PDF
   └─ 读取 examples/OpenVLA_AnOpen-Source Vision-Language-Action_Model.pdf（37 页论文）
   └─ 使用 PyPDF2 提取全文
   └─ 按 800 字符/块、100 字符重叠 切分为 ~194 个 chunks
   └─ 对每个 chunk 计算 hash embedding 并写入内存向量库

Step 4  注册 SQL 表
   └─ 创建 model_benchmarks 表（6 行模型评测数据）
```

### 阶段二：查询演示（4 条查询覆盖 3 种路由 + ACL 测试）

| # | 查询 | 路由 | 执行过程 | 预期结果 |
|---|------|------|----------|----------|
| 1 | `"What is OpenVLA and what does it do?"` | mixed (Doc RAG) | route_node 判定 → retrieve_node 检索向量库 → synthesize_node 摘要 → verify_node 验证 → finalize_node | 返回从 PDF 中提取的 OpenVLA 相关摘要，confidence=high，12 条 citations |
| 2 | `"SELECT model, success_rate FROM model_benchmarks WHERE task = 'pick-and-place'"` | sql | route_node 识别 SELECT 关键字 → sql_node 执行 SQL → synthesize → verify → finalize | 返回 4 行匹配结果（OpenVLA, RT-2-X, Octo-Base, Diffusion Policy），confidence=high |
| 3 | `"class OpenVLAController"` | code | route_node 识别 class 关键字 → code_node 在内存代码中搜索 → synthesize → verify → finalize | 找到 robot_controller.py 中的代码片段，confidence=high |
| 4 | `"What is OpenVLA?"` (unauthorized-user) | mixed | route_node → retrieve_node 检索（ACL 过滤掉所有 chunk）→ 循环 3 次均无证据 → finalize 降级 | 返回"证据不足"提示，confidence=low |

## 4. Agent 执行流程图

```
用户查询
   │
   ▼
┌─────────────┐
│  route_node │  关键词/LLM 路由判定
└──────┬──────┘
       │ route = sql / code / doc_rag / mixed
       ▼
┌─────────────────────────────────────────┐
│  循环（最多 3 轮迭代）                    │
│                                         │
│  ┌───────────┐  ┌───────────┐           │
│  │ sql_node  │  │ code_node │           │
│  └─────┬─────┘  └─────┬─────┘           │
│        │               │                │
│  ┌─────┴─────┐  ┌──────┴──────┐         │
│  │retrieve   │  │  web_node   │         │
│  │  _node    │  │             │         │
│  └─────┬─────┘  └──────┬──────┘         │
│        └───────┬───────┘                │
│                ▼                        │
│       ┌────────────────┐                │
│       │ synthesize_node│  生成草稿回答    │
│       └───────┬────────┘                │
│               ▼                         │
│       ┌────────────────┐                │
│       │  verify_node   │  验证证据充分性  │
│       └───────┬────────┘                │
│               │                         │
│          enough? ──Yes──► 跳出循环       │
│               │No                       │
│          下一轮迭代                      │
└─────────────────────────────────────────┘
       │
       ▼
┌──────────────┐
│ finalize_node│  生成最终回答 + confidence
└──────────────┘
       │
       ▼
   返回结果
```

## 5. 运行输出示例

```
============================================================
  Step 1: Initialize Services
============================================================
  Services initialized (InMemory mode, no external DB required)

============================================================
  Step 2: Setup ACL Policy
============================================================
  Policy created: allow_users=[demo-user]
  Policy hash  : e7376262dfbf...

============================================================
  Step 3: Ingest PDF
============================================================
  PDF extracted: 194 chunks from OpenVLA_AnOpen-Source Vision-Language-Action_Model.pdf
  First chunk preview (800 chars):
    "OpenVLA: An Open-Source Vision-Language-Action Model Moo Jin Kim..."
  Embedded and indexed: 194 chunks

============================================================
  Step 4: Register SQL Table
============================================================
  Table 'model_benchmarks' registered with 6 rows

============================================================
  Query 1: Doc RAG - 'What is OpenVLA?'
============================================================
  Route      : mixed
  Confidence : high
  Answer     :
    根据检索内容，总结如下：How important is OpenX training ...
  Tool calls : 1
    - retrieve [OK]
  Citations  : 12

============================================================
  Query 2: SQL - 'Pick-and-place success rates'
============================================================
  Route      : sql
  Confidence : high
  Answer     :
    SQL 返回 4 行。
  Tool calls : 1
    - sql_query [OK]
  Citations  : 4

============================================================
  Query 3: Code Search - 'class OpenVLAController'
============================================================
  Route      : code
  Confidence : high
  Answer     :
    已检索到相关代码片段。
  Tool calls : 1
    - code_search [OK]
  Citations  : 1
    [1] code: robot_controller.py

============================================================
  Query 4: ACL Block - unauthorized user
============================================================
  Route      : mixed
  Confidence : low
  Answer     :
    当前证据不足，缺少: doc_chunks。建议补充相关文档或权限后重试。
  Tool calls : 3
    - retrieve [OK]  (3 轮迭代均无法检索到数据)

============================================================
  Summary
============================================================
  4 queries executed across 3 routes + 1 ACL test:
    1. Doc RAG   -> confidence=high
    2. SQL       -> confidence=high
    3. Code      -> confidence=high
    4. ACL Block -> confidence=low
```

## 6. 结果分析

### Query 1 — Doc RAG（文档检索）

- **路由**: `mixed`（查询为纯自然语言，无 SQL/Code 关键词，fallback 到 mixed）
- **过程**: retrieve_node 使用 hash embedding 在 194 个 chunk 中进行向量检索 + FTS 全文检索，RRF 融合排序后取 top-30 chunk 作为证据
- **结果**: 从 PDF 中提取了包含 "OpenVLA" 和 "performance" 等关键词的段落，synthesize_node 基于关键词评分抽取前 3 个最相关句子拼接为摘要
- **注意**: 由于未配置真实 LLM（无 OPENAI_API_KEY），synthesize 使用的是基于关键词重叠的启发式摘要而非 LLM 生成，所以回答质量有限。配置 LLM 后回答质量会显著提升

### Query 2 — SQL 查询

- **路由**: `sql`（检测到 SELECT 关键词）
- **过程**: sql_node 使用内存 SQL 引擎解析 `SELECT ... FROM model_benchmarks WHERE task = 'pick-and-place'`，在内存表中过滤并返回匹配行
- **结果**: 返回 4 行（OpenVLA, RT-2-X, Octo-Base, Diffusion Policy 在 pick-and-place 任务上的成功率）

### Query 3 — Code 搜索

- **路由**: `code`（检测到 "class" 关键词）
- **过程**: code_node 使用 `re.escape("class OpenVLAController")` 在内存代码文件中进行正则搜索
- **结果**: 在 `robot_controller.py` 中找到 `class OpenVLAController` 定义，返回包含上下文的代码片段

### Query 4 — ACL 拦截

- **路由**: `mixed`
- **过程**: route_node 计算 `unauthorized-user` 的 ACL hash，该用户不在策略的 `allow_users` 列表中。retrieve_node 在检索时过滤掉所有不匹配 ACL 的 chunk，导致连续 3 轮迭代均无证据。verify_node 判定证据不足
- **结果**: finalize_node 生成降级回答"证据不足"，confidence=low。这证明了 ACL 权限控制正常工作

## 7. 关键概念映射

| 概念 | 案例中的体现 |
|------|------------|
| PDF 摄取 | `ingest_pdf()` 提取 PDF → 切分 chunk → `embed_and_upsert()` 嵌入 |
| 向量检索 | InMemoryQdrant 余弦相似度搜索 + pg_fts 全文搜索 → RRF 融合 |
| SQL 执行 | InMemory SqlEngine 解析简单 SELECT 并在内存表中执行 |
| 代码搜索 | CodeSearch 正则匹配内存中的代码文件 |
| ACL 控制 | build_policy → acl_hash → 检索时按 hash 过滤 chunk |
| Agent 循环 | route → tool → synthesize → verify → (重试或结束) → finalize |
| 降级回答 | 证据不足时 finalize_node 输出低置信度回答 |
