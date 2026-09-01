export interface RetrievalChunk {
  chunk_id: string;
  doc_id: string;
  text: string;
  score: number;
  citations: string[];
  metadata: Record<string, unknown>;
}

export interface SqlResult {
  rows: Record<string, unknown>[];
  columns: { name: string; type: string }[];
  stats: Record<string, unknown>;
}

export interface CodeSnippet {
  path: string;
  ref: string;
  line_start: number;
  line_end: number;
  content: string;
}

export interface PatchResult {
  path: string;
  diff: string;
  original_lines: number;
  modified_lines: number;
}

// Document/vector retrieval source types. SQL database access is configured
// separately through POSTGRES_DSN and is not an ingestible source.
export type SourceType = "pdf" | "web" | "repo" | "local_fs";
export type RouteType = "doc_rag" | "sql" | "code" | "mixed" | "web_fallback";
export type Confidence = "high" | "medium" | "low";
export type ToolName =
  | "retrieve"
  | "sql_query"
  | "code_search"
  | "web_search"
  | "web_fetch"
  | "open_file"
  | "apply_patch"
  | "explain_error";

export interface Citation {
  kind: "chunk" | "row" | "code" | "web";
  chunk_id?: string;
  doc_id?: string;
  page?: number;
  section?: string;
  url?: string;
  path?: string;
  ref?: string;
  line_start?: number;
  line_end?: number;
  row_ref?: string;
  title?: string;
}

export interface EvidenceItem {
  kind:
    | "doc_chunk"
    | "sql_rows"
    | "code_snippets"
    | "web_snippets"
    | "file_content"
    | "patch"
    | "error_analysis";
  score?: number;
  text?: string;
  citations: Citation[];
  metadata?: Record<string, unknown>;
}

export interface ToolCallRecord {
  name: ToolName;
  args: Record<string, unknown>;
  ok: boolean;
  started_at_ms: number;
  ended_at_ms: number;
  error?: string;
  result_preview?: Record<string, unknown>;
}

export interface Constraints {
  source_types?: SourceType[];
  doc_ids?: string[];
  tags?: string[];
  repo?: string;
  ref?: string;
  path_prefix?: string;
  url_prefix?: string;
  time_from?: string;
  time_to?: string;
  security_scope?: Record<string, unknown>;
}

export interface Verification {
  enough_evidence: boolean;
  missing: string[];
  next_query?: string;
  next_action?:
    | "retrieve"
    | "sql_query"
    | "code_search"
    | "web_search"
    | "open_file"
    | "apply_patch"
    | "explain_error";
}

export interface Draft {
  answer_outline: string[];
  answer_text: string;
  used_citations: Citation[];
}

export interface FinalAnswer {
  answer: string;
  citations: Citation[];
  confidence: Confidence;
  followups: string[];
}

export interface AgentState {
  request_id: string;
  tenant_id: string;
  user_id: string;
  session_id?: string;
  query: string;
  constraints: Constraints;
  route?: RouteType;
  plan: string[];
  tool_calls: ToolCallRecord[];
  evidence: EvidenceItem[];
  draft?: Draft;
  verification?: Verification;
  iteration: number;
  max_iterations: number;
  hard_fail: boolean;
  final?: FinalAnswer;
}
