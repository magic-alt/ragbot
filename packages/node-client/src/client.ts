export interface ChatRequest {
  query: string;
  tenant_id: string;
  user_id: string;
}

export interface ChatResponse {
  answer: string;
  citations: string[];
  confidence: string;
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
