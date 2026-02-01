export interface ChatRequest {
  query: string;
  tenant_id: string;
  user_id: string;
  session_id?: string;
  stream?: boolean;
  constraints?: {
    source_types?: Array<"pdf" | "web" | "repo" | "db_doc">;
    doc_ids?: string[];
    tags?: string[];
    repo?: string;
    ref?: string;
    path_prefix?: string;
    url_prefix?: string;
    time_from?: string;
    time_to?: string;
  };
}

export interface ChatResponse {
  request_id: string;
  answer: string;
  citations: Array<Record<string, unknown>>;
  confidence: string;
  followups?: string[];
  debug?: Record<string, unknown>;
}

export async function chat(baseUrl: string, payload: ChatRequest): Promise<ChatResponse> {
  const res = await fetch(`${baseUrl}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    throw new Error(`Chat request failed: ${res.status}`);
  }
  return (await res.json()) as ChatResponse;
}
