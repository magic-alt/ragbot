# CLI paths containing spaces

Ragbot accepts local ingest paths containing spaces in both the product CLI and
the repository bootstrap controller. Quoting is still recommended because it is
portable across shells, but ordinary unquoted paths are normalized before
argument parsing.

Windows PowerShell examples:

```powershell
python .\scripts\ragbot.py ingest .\data\DeepSeek in Action LLM Deployment.pdf.pdf `
  --tenant engineering `
  --type pdf

.\.venv\Scripts\python.exe -m cli.rag `
  --server http://127.0.0.1:8000 `
  --tenant engineering `
  ingest .\data\DeepSeek in Action LLM Deployment.pdf.pdf `
  --type pdf `
  --wait
```

The quoted forms remain valid and are preferred when a path contains shell
metacharacters:

```powershell
python .\scripts\ragbot.py ingest ".\data\DeepSeek in Action LLM Deployment.pdf.pdf" `
  --tenant engineering `
  --type pdf
```

Normalization applies only to positional tokens after `ingest` and before the
first `--option`; option values are not merged into the path.
