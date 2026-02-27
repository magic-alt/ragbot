import type { Citation } from "./types";

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
  citations: Citation[];
  confidence: string;
  followups?: string[];
  debug?: Record<string, unknown>;
}

export interface SSEToolCallEvent {
  request_id: string;
  name: string;
  args: Record<string, unknown>;
}

export interface SSEToolResultEvent {
  request_id: string;
  name: string;
  ok: boolean;
  meta?: Record<string, unknown>;
  error?: string;
}

export interface SSETokenEvent {
  request_id: string;
  delta: string;
}

export interface SSECitationEvent {
  request_id: string;
  citations: Citation[];
}

export interface SSEFinalEvent {
  request_id: string;
  answer: string;
  citations: Citation[];
  confidence: string;
  followups: string[];
}

export type SSEEvent =
  | { event: "tool_call"; data: SSEToolCallEvent }
  | { event: "tool_result"; data: SSEToolResultEvent }
  | { event: "token"; data: SSETokenEvent }
  | { event: "citation"; data: SSECitationEvent }
  | { event: "final"; data: SSEFinalEvent };

export type SSEEventHandler = (event: SSEEvent) => void;

export async function chat(
  baseUrl: string,
  payload: ChatRequest,
  apiKey?: string,
): Promise<ChatResponse> {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
  };
  if (apiKey) {
    headers["X-API-Key"] = apiKey;
  }
  const res = await fetch(`${baseUrl}/chat`, {
    method: "POST",
    headers,
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    throw new Error(`Chat request failed: ${res.status}`);
  }
  return (await res.json()) as ChatResponse;
}

export async function chatStream(
  baseUrl: string,
  payload: ChatRequest,
  onEvent: SSEEventHandler,
  apiKey?: string,
): Promise<void> {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
  };
  if (apiKey) {
    headers["X-API-Key"] = apiKey;
  }
  const res = await fetch(`${baseUrl}/chat`, {
    method: "POST",
    headers,
    body: JSON.stringify({ ...payload, stream: true }),
  });
  if (!res.ok) {
    throw new Error(`Chat stream request failed: ${res.status}`);
  }
  if (!res.body) {
    throw new Error("Response body is null");
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";

    let currentEvent = "";
    for (const line of lines) {
      if (line.startsWith("event: ")) {
        currentEvent = line.slice(7).trim();
      } else if (line.startsWith("data: ") && currentEvent) {
        try {
          const data = JSON.parse(line.slice(6));
          onEvent({ event: currentEvent, data } as SSEEvent);
        } catch {
          // skip malformed JSON
        }
        currentEvent = "";
      }
    }
  }
}
