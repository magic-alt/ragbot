# Ragbot CLI 部署与操作手册

本文档面向“clone 后立即可用”的本地开发与单机测试场景，覆盖 Windows PowerShell、Linux/macOS、Docker Compose 和无 Docker 纯 Python 两种路径。

> 推荐入口：`python scripts/ragbot.py ...`。这个脚本只依赖 Python 标准库，即使 `.venv` 中没有 `pip`，也会优先尝试通过 `ensurepip` 修复后再安装 Ragbot。

## 1. 最短路径：一条命令启动

要求：Python 3.10+。

```bash
git clone https://github.com/magic-alt/ragbot.git
cd ragbot
python scripts/ragbot.py up --mode auto
```

`--mode auto` 的规则：

- Docker CLI、Compose 和 Docker daemon 均可用：启动完整 Docker Compose 栈；
- Docker 不可用：自动退回纯 Python local 模式；
- 两种模式都会创建 `.env`、`data/`、`.venv/`，安装所需依赖并执行 readiness 检查。

启动完成后：

```text
API       http://127.0.0.1:8000
Admin UI  http://127.0.0.1:8000/admin/ui
```

检查状态：

```bash
python scripts/ragbot.py status
python scripts/ragbot.py doctor
```

## 2. Windows PowerShell

推荐直接使用 Python 入口，可避免 PowerShell ExecutionPolicy 对 `.ps1` 的限制：

```powershell
cd D:\Project\ragbot
python .\scripts\ragbot.py up --mode auto
```

也可以使用包装器：

```powershell
.\scripts\ragbot.ps1 up --mode auto
```

如果系统禁止执行 PowerShell 脚本，继续使用 `python .\scripts\ragbot.py ...` 即可。

### 强制无 Docker 模式

```powershell
python .\scripts\ragbot.py up --mode local
```

### 强制 Docker 模式

```powershell
python .\scripts\ragbot.py up --mode docker
```

### 常见环境修复

如果虚拟环境存在但没有 pip：

```powershell
.\.venv\Scripts\python.exe -m ensurepip --upgrade
```

正常情况下不需要手工执行；`scripts/ragbot.py` 会自动检测并修复。

如果依赖损坏或环境混乱：

```powershell
python .\scripts\ragbot.py setup --mode local --force-install
```

或：

```powershell
python .\scripts\ragbot.py up --mode local --force-install
```

## 3. Linux / macOS

直接运行：

```bash
python3 scripts/ragbot.py up --mode auto
```

或者：

```bash
bash scripts/ragbot.sh up --mode auto
```

## 4. 两种部署模式的区别

| 模式 | 启动命令 | PostgreSQL | Qdrant | Worker | 数据持久化 | 用途 |
| --- | --- | --- | --- | --- | --- | --- |
| local | `up --mode local` | InMemoryRepo | InMemoryQdrant | inline | 仅 API 进程生命周期 | 快速开发、功能验证 |
| docker | `up --mode docker` | PostgreSQL 16 | Qdrant | 独立 worker | Docker volumes | 长期本地知识库、接近生产拓扑 |

local 模式会强制：

```text
RAGBOT_ENV=development
RAGBOT_INGESTION_MODE=inline
POSTGRES_DSN=<unset>
QDRANT_URL=<unset>
RAGBOT_ALLOWED_LOCAL_SOURCE_ROOTS=<repo>/data
```

因此无需 PostgreSQL、Qdrant Server 或 worker 进程。

注意：local 模式的数据存储在 API 进程内存中。执行 `down` 或进程退出后需要重新 ingest。

## 5. 配置真实 Embedding / LLM

首次运行会从 `.env.example` 创建 `.env`。

要测试真实语义检索，至少配置 Embedding：

```dotenv
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_API_KEY=<your-key>
EMBEDDING_BASE_URL=https://api.openai.com
QDRANT_DIM=1536
```

要使用 `ask`，再配置 LLM：

```dotenv
RAGBOT_LLM_PROVIDER=openai
OPENAI_API_KEY=<your-key>
OPENAI_BASE_URL=https://api.openai.com
OPENAI_MODEL=gpt-4o-mini
```

修改 `.env` 后重启：

```bash
python scripts/ragbot.py restart --mode local
```

或：

```bash
python scripts/ragbot.py restart --mode docker
```

如果 `EMBEDDING_API_KEY` 为空，development 模式会使用 HashEmbedder fallback。它适合验证 pipeline 是否跑通，不适合评估中文或真实 semantic retrieval 质量。

## 6. 准备本地文档

所有本地文件建议放在仓库 `data/` 下：

```text
ragbot/
├─ data/
│  ├─ manuals/
│  │  ├─ ethercat.md
│  │  ├─ motor_control.txt
│  │  └─ notes.csv
│  └─ pdf/
│     └─ product_manual.pdf
```

Docker 模式下，脚本会自动把：

```text
./data/manuals
```

转换成容器路径：

```text
/data/manuals
```

因此用户不需要手工处理 Windows `D:\...` 与容器 `/data/...` 的路径映射。

### local_fs 当前支持的目录文件类型

```text
.txt
.md
.markdown
.rst
.csv
.log
```

PDF 使用独立 `pdf` connector，应按文件 ingest；扫描型 PDF 目前仍需先 OCR。

## 7. 导入本地文本目录

```bash
python scripts/ragbot.py ingest data/manuals \
  --tenant engineering \
  --name "Engineering manuals" \
  --tag manuals
```

Windows PowerShell：

```powershell
python .\scripts\ragbot.py ingest .\data\manuals `
  --tenant engineering `
  --name "Engineering manuals" `
  --tag manuals
```

默认会等待 ingestion 完成。若只提交不等待：

```bash
python scripts/ragbot.py ingest data/manuals --tenant engineering --no-wait
```

## 8. 导入 PDF

```bash
python scripts/ragbot.py ingest data/pdf/product_manual.pdf \
  --tenant engineering \
  --type pdf
```

远程 PDF：

```bash
python scripts/ragbot.py ingest https://example.com/manual.pdf \
  --tenant engineering \
  --type pdf
```

## 9. 导入 Git / Web

Git：

```bash
python scripts/ragbot.py ingest https://github.com/magic-alt/ragbot \
  --tenant engineering \
  --type repo \
  --ref main \
  --tag code
```

Web：

```bash
python scripts/ragbot.py ingest https://example.com/knowledge-base/ \
  --tenant engineering \
  --type web
```

## 10. Manifest 批量导入

```bash
python scripts/ragbot.py import examples/ragbot-manifest.json \
  --tenant engineering
```

注意：Docker 模式的 manifest 中，本地 Source 路径应使用容器可见路径 `/data/...`。单 Source 的 `scripts/ragbot.py ingest ...` 会自动转换路径，但 manifest 内容不会被脚本重写。

## 11. Search

```bash
python scripts/ragbot.py search "EtherCAT Distributed Clock 如何工作？" \
  --tenant engineering \
  --top-k 5
```

Windows：

```powershell
python .\scripts\ragbot.py search "EtherCAT Distributed Clock 如何工作？" --tenant engineering --top-k 5
```

## 12. Ask / Agentic RAG

```bash
python scripts/ragbot.py ask "根据文档总结 EtherCAT DC 的同步机制，并引用来源" \
  --tenant engineering
```

`ask` 需要有效 LLM provider 配置；`search` 只需要 retrieval/embedding 链路。

## 13. 日常运维命令

### 状态

```bash
python scripts/ragbot.py status
```

### Doctor

```bash
python scripts/ragbot.py doctor
```

### 日志

```bash
python scripts/ragbot.py logs
python scripts/ragbot.py logs --lines 300
python scripts/ragbot.py logs -f
```

local 模式日志：

```text
logs/ragbot-local.log
```

Docker 模式会代理：

```text
docker compose logs
```

### 重启

```bash
python scripts/ragbot.py restart --mode local
python scripts/ragbot.py restart --mode docker
```

### 停止

```bash
python scripts/ragbot.py down
```

Docker 模式默认保留 PostgreSQL/Qdrant volumes。

只有明确需要清空全部 Docker 数据时使用：

```bash
python scripts/ragbot.py down --mode docker --volumes
```

这是破坏性操作。

## 14. 仍然可以直接使用原生 `rag` CLI

Bootstrap helper 不替代正式 CLI，它负责安装、部署和常见运维；底层 ingest/search/ask 仍调用仓库原生 `cli.rag`。

安装成功后：

```bash
.venv/bin/python -m cli.rag --server http://127.0.0.1:8000 doctor
```

Windows：

```powershell
.\.venv\Scripts\python.exe -m cli.rag --server http://127.0.0.1:8000 doctor
```

或者虚拟环境已激活时：

```bash
rag --server http://127.0.0.1:8000 --tenant engineering search "query" --top-k 5
```

## 15. 常见故障

### `No module named pip`

控制脚本会自动执行：

```text
<venv-python> -m ensurepip --upgrade
```

如果系统 Python 自身没有 `ensurepip`，请安装完整的 Python 3.10+ 发行版。

### `No module named uvicorn` / `No module named requests`

执行：

```bash
python scripts/ragbot.py setup --mode local --force-install
```

脚本会使用 `.venv` 中的 Python 安装 Ragbot editable package 和 `worker` extra，避免系统 Python / Conda / uv / venv 混装。

### PowerShell 无法执行 `Activate.ps1`

无需激活虚拟环境。直接使用：

```powershell
python .\scripts\ragbot.py up --mode local
```

控制脚本始终显式调用 `.venv\Scripts\python.exe`。

### Docker 不可用

```bash
python scripts/ragbot.py up --mode local
```

不需要 Docker、PostgreSQL、Qdrant Server 或独立 worker。

### Docker 本地路径不可访问

把文件放入：

```text
./data
```

然后通过 helper ingest：

```bash
python scripts/ragbot.py ingest data/...
```

helper 会自动转换为 `/data/...`。

## 16. 推荐首次验收流程

```bash
python scripts/ragbot.py up --mode auto
python scripts/ragbot.py doctor
python scripts/ragbot.py ingest data/manuals --tenant local-test
python scripts/ragbot.py ingest data/pdf/manual.pdf --tenant local-test --type pdf
python scripts/ragbot.py search "一个文档中明确存在的问题" --tenant local-test --top-k 5
python scripts/ragbot.py ask "根据文档回答，并引用来源" --tenant local-test
python scripts/ragbot.py status
```

如果目标是长期保存知识库，优先选择 Docker 模式；如果只是快速验证 parser/chunk/retrieval/agent pipeline，local 模式更轻量。

## 17. 命令速查

| 目标 | 命令 |
| --- | --- |
| 自动选择模式并启动 | `python scripts/ragbot.py up --mode auto` |
| 无 Docker 启动 | `python scripts/ragbot.py up --mode local` |
| Docker 启动 | `python scripts/ragbot.py up --mode docker` |
| 重装依赖 | `python scripts/ragbot.py setup --mode local --force-install` |
| 状态 | `python scripts/ragbot.py status` |
| 健康检查 | `python scripts/ragbot.py doctor` |
| 导入 | `python scripts/ragbot.py ingest data/manuals --tenant default` |
| Search | `python scripts/ragbot.py search "query" --tenant default` |
| Ask | `python scripts/ragbot.py ask "question" --tenant default` |
| 日志 | `python scripts/ragbot.py logs -f` |
| 重启 | `python scripts/ragbot.py restart --mode local` |
| 停止 | `python scripts/ragbot.py down` |

生产环境仍应遵循 [`DEPLOYMENT.md`](DEPLOYMENT.md)、[`CONFIGURATION.md`](CONFIGURATION.md)、[`DISASTER_RECOVERY.md`](DISASTER_RECOVERY.md) 和 v1 release gate；本文的 local 模式定位是开发与真实文档功能验证，不替代生产持久化架构。
