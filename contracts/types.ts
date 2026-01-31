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

export interface Evidence {
  kind: string;
  payload: Record<string, unknown>;
  citations: string[];
}

export interface AgentVerdict {
  enough_evidence: boolean;
  missing_what?: string;
}

export interface AgentState {
  query: string;
  tenant_id: string;
  user_id: string;
  constraints: Record<string, unknown>;
  route?: string;
  tool_calls: Record<string, unknown>[];
  evidence: Evidence[];
  draft?: string;
  verdict?: AgentVerdict;
  final?: string;
}
