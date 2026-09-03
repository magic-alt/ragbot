# Recursive PDF corpus ingestion

Ragbot's `local_fs` connector intentionally scans text-like files such as Markdown, TXT, RST, CSV and logs. PDF files use the dedicated `pdf` connector, so a directory Source does not implicitly parse PDFs.

For a real document corpus, use the repository helper to recursively discover every PDF below `./data`, create one stable PDF Source per file, submit them in batches, wait for indexing, and reuse the normal Ragbot ingestion pipeline.

## Fastest path

Start Ragbot first:

```powershell
python .\scripts\ragbot.py up --mode auto
```

Put PDFs anywhere below the repository `data` directory, for example:

```text
data/
├─ manuals/
│  ├─ motor-control.pdf
│  └─ ethercat/
│     └─ dc-guide.pdf
├─ papers/
│  └─ rag-survey.PDF
└─ notes.md
```

Then ingest every PDF recursively:

```powershell
python .\scripts\ingest_pdfs.py .\data --tenant engineering
```

The directory argument is optional. This is equivalent:

```powershell
python .\scripts\ingest_pdfs.py --tenant engineering
```

Linux/macOS:

```bash
python scripts/ingest_pdfs.py data --tenant engineering
```

## What the helper does

1. reads `tmp/ragbot-runtime.json` written by `scripts/ragbot.py up`;
2. discovers all files whose extension is `.pdf`, case-insensitively;
3. keeps every source below the repository `./data` security boundary;
4. maps host paths to `/data/...` automatically when the active runtime is Docker;
5. creates one Ragbot `pdf` Source per PDF;
6. groups Sources into manifests of at most 100 items because the batch API limit is 100 Sources per request;
7. delegates each manifest to `scripts/ragbot.py import`;
8. waits for indexing by default and reports the normal Ragbot document/chunk counts.

Each PDF remains a distinct Source, so nested files with the same basename do not collapse into one directory Source identity.

## Useful options

Preview discovery without ingesting anything:

```powershell
python .\scripts\ingest_pdfs.py .\data --dry-run
```

Only scan the selected directory, not its children:

```powershell
python .\scripts\ingest_pdfs.py .\data\manuals --no-recursive
```

Apply tags to every PDF:

```powershell
python .\scripts\ingest_pdfs.py .\data `
  --tenant engineering `
  --tag manuals `
  --tag pdf
```

Override chunking:

```powershell
python .\scripts\ingest_pdfs.py .\data `
  --tenant engineering `
  --chunk-size 900 `
  --chunk-overlap 120
```

Test only the first 20 discovered PDFs:

```powershell
python .\scripts\ingest_pdfs.py .\data `
  --tenant engineering `
  --max-files 20
```

Submit jobs without waiting for indexing:

```powershell
python .\scripts\ingest_pdfs.py .\data `
  --tenant engineering `
  --no-wait
```

Continue with later batches if one batch fails:

```powershell
python .\scripts\ingest_pdfs.py .\data `
  --tenant engineering `
  --continue-on-error
```

## Verify the corpus

After ingestion:

```powershell
python .\scripts\ragbot.py search `
  "EtherCAT Distributed Clock" `
  --tenant engineering `
  --top-k 5
```

Then test Agentic RAG:

```powershell
python .\scripts\ragbot.py ask `
  "根据文档总结 EtherCAT Distributed Clock 的同步机制，并引用来源" `
  --tenant engineering
```

## Re-running the command

The normal Quick Import path uses stable Source identity derived from tenant + source type + normalized location. Re-running the PDF corpus helper therefore reuses the same PDF Sources instead of intentionally creating a fresh Source for every execution.

This makes the command suitable for repeatable local corpus refreshes. The ingestion pipeline still decides which chunks can be reused versus re-embedded according to the current Source content and representation contract.

## Important PDF limitation

The current PDF connector extracts text with `PyPDF2`. It does not perform OCR.

Therefore:

- searchable/native-text PDFs are appropriate for direct ingestion;
- scanned/image-only PDFs should be OCR-processed before Ragbot ingestion;
- changing the embedding model or vector dimension requires a compatible collection/reindex strategy.

## Mixed `data` corpus

For text files, keep using the directory Source:

```powershell
python .\scripts\ragbot.py ingest .\data `
  --tenant engineering `
  --type local_fs
```

Then ingest all PDFs:

```powershell
python .\scripts\ingest_pdfs.py .\data `
  --tenant engineering
```

Together these two commands index the currently supported text-like local files plus all PDFs below the same `data` tree.
