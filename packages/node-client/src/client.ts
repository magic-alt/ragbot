import type { Citation, SourceType } from "./types";

export type RetrievalPlan = "dense" | "lexical" | "hybrid_rrf" | "qdrant_dense_sparse";

export interface ChatRequest {
  query: string;
  tenant_id: string;
  user_id: string;
  session_id?: string;
  stream?: boolean;
  constraints?: {
    source_types?: SourceType[];
    doc_ids?: string[];
    tags?: string[];
    repo?: string;
    ref?: string;
    path_prefix?: string;
    url_prefix?: string;
    time_from?: string;
    time_to?: string;
  };
  client_context?: Record<string, unknown>;
}

export interface ChatResponse {
  request_id: string;
  answer: string;
  citations: Citation[];
  confidence: string;
  followups?: string[];
  debug?: Record<string, unknown>;
}

export interface SearchRequest {
  query: string;
  tenant_id: string;
  user_id: string;
  top_k?: number;
  plan?: RetrievalPlan;
  candidate_pool?: number;
  rerank?: boolean;
  diversity?: boolean;
  deadline_ms?: number;
  explain?: boolean;
  filters?: Record<string, unknown>;
}

export interface SearchChunk {
  chunk_id: string;
  doc_id: string;
  text: string;
  score: number;
  citations: string[];
  metadata: Record<string, unknown>;
}

export interface SearchResponse {
  request_id: string;
  chunks: SearchChunk[];
  total: number;
  diagnostics: Record<string, unknown>;
}

export interface SourceRecord {
  source_id: string;
  tenant_id: string;
  source_type: string;
  name: string;
  config: Record<string, unknown>;
  acl_policy_id?: string | null;
  tags?: string[];
  status?: string;
  created_at?: string | null;
  updated_at?: string | null;
  [key: string]: unknown;
}

export interface JobRecord {
  job_id: string;
  tenant_id: string;
  source_id: string;
  source_type: string;
  status: string;
  stats?: Record<string, unknown>;
  [key: string]: unknown;
}

export interface SourcePage {
  total: number;
  next_cursor: string | null;
  sources: SourceRecord[];
}

export interface JobPage {
  total: number;
  next_cursor: string | null;
  jobs: JobRecord[];
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

export interface SSEErrorEvent {
  request_id?: string;
  error: string;
}

export type SSEEvent =
  | { event: "tool_call"; data: SSEToolCallEvent }
  | { event: "tool_result"; data: SSEToolResultEvent }
  | { event: "token"; data: SSETokenEvent }
  | { event: "citation"; data: SSECitationEvent }
  | { event: "final"; data: SSEFinalEvent }
  | { event: "error"; data: SSEErrorEvent }
  | { event: string; data: Record<string, unknown> };

export type SSEEventHandler = (event: SSEEvent) => void;

export interface RequestOptions {
  signal?: AbortSignal;
  timeoutMs?: number;
}

export interface RagbotClientOptions {
  apiKey?: string;
  timeoutMs?: number;
  maxRetries?: number;
  fetchImpl?: typeof fetch;
}

export class RagbotApiError extends Error {
  readonly statusCode: number;
  readonly code: string;
  readonly requestId?: string;
  readonly retryable: boolean;
  readonly details?: unknown;

  constructor(args: {
    statusCode: number;
    code: string;
    message: string;
    requestId?: string;
    retryable?: boolean;
    details?: unknown;
  }) {
    super(args.message);
    this.name = "RagbotApiError";
    this.statusCode = args.statusCode;
    this.code = args.code;
    this.requestId = args.requestId;
    this.retryable = Boolean(args.retryable);
    this.details = args.details;
  }
}

export class RagbotClient {
  readonly baseUrl: string;
  readonly apiKey?: string;
  readonly timeoutMs: number;
  readonly maxRetries: number;
  private readonly fetchImpl: typeof fetch;

  constructor(baseUrl: string, options: RagbotClientOptions = {}) {
    this.baseUrl = baseUrl.replace(/\/$/, "");
    this.apiKey = options.apiKey;
    this.timeoutMs = options.timeoutMs ?? 30_000;
    this.maxRetries = Math.max(0, options.maxRetries ?? 2);
    this.fetchImpl = options.fetchImpl ?? fetch;
  }

  search(payload: SearchRequest, options: RequestOptions = {}): Promise<SearchResponse> {
    return this.request<SearchResponse>("POST", "/v1/search", {
      body: payload,
      retryable: true,
      ...options,
    });
  }

  chat(payload: ChatRequest, options: RequestOptions = {}): Promise<ChatResponse> {
    return this.request<ChatResponse>("POST", "/v1/chat", {
      body: payload,
      retryable: true,
      ...options,
    });
  }

  getSource(sourceId: string, options: RequestOptions = {}): Promise<SourceRecord> {
    return this.request<SourceRecord>("GET", `/v1/sources/${encodeURIComponent(sourceId)}`, options);
  }

  listSources(
    args: { tenantId?: string; limit?: number; cursor?: string } = {},
    options: RequestOptions = {},
  ): Promise<SourcePage> {
    const query = queryString({
      tenant_id: args.tenantId,
      limit: args.limit ?? 100,
      cursor: args.cursor,
    });
    return this.request<SourcePage>("GET", `/v1/sources${query}`, options);
  }

  async *iterateSources(
    args: { tenantId?: string; pageSize?: number } = {},
    options: RequestOptions = {},
  ): AsyncGenerator<SourceRecord> {
    let cursor: string | undefined;
    do {
      const page = await this.listSources(
        { tenantId: args.tenantId, limit: args.pageSize ?? 100, cursor },
        options,
      );
      for (const source of page.sources) yield source;
      cursor = page.next_cursor ?? undefined;
    } while (cursor);
  }

  listJobs(
    args: {
      tenantId?: string;
      sourceId?: string;
      status?: string;
      limit?: number;
      cursor?: string;
    } = {},
    options: RequestOptions = {},
  ): Promise<JobPage> {
    const query = queryString({
      tenant_id: args.tenantId,
      source_id: args.sourceId,
      status: args.status,
      limit: args.limit ?? 100,
      cursor: args.cursor,
    });
    return this.request<JobPage>("GET", `/v1/catalog/jobs${query}`, options);
  }

  async chatStream(
    payload: ChatRequest,
    onEvent: SSEEventHandler,
    options: RequestOptions = {},
  ): Promise<void> {
    const { signal, cleanup } = timeoutSignal(options.signal, options.timeoutMs ?? this.timeoutMs);
    try {
      const response = await this.fetchImpl(`${this.baseUrl}/v1/chat`, {
        method: "POST",
        headers: this.headers(),
        body: JSON.stringify({ ...payload, stream: true }),
        signal,
      });
      if (!response.ok) throw await apiError(response);
      if (!response.body) throw new Error("Response body is null");
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let currentEvent = "message";
      let dataLines: string[] = [];

      const dispatch = () => {
        if (dataLines.length === 0) return;
        const raw = dataLines.join("\n");
        let data: Record<string, unknown>;
        try {
          const parsed = JSON.parse(raw) as unknown;
          data = isRecord(parsed) ? parsed : { value: parsed };
        } catch {
          data = { raw };
        }
        onEvent({ event: currentEvent, data } as SSEEvent);
        currentEvent = "message";
        dataLines = [];
      };

      const consumeLine = (rawLine: string) => {
        const line = rawLine.replace(/\r$/, "");
        if (line === "") {
          dispatch();
        } else if (line.startsWith("event:")) {
          currentEvent = line.slice(6).trim() || "message";
        } else if (line.startsWith("data:")) {
          dataLines.push(line.slice(5).trimStart());
        }
      };

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";
        for (const line of lines) consumeLine(line);
      }
      buffer += decoder.decode();
      for (const line of buffer.split("\n")) consumeLine(line);
      dispatch();
    } finally {
      cleanup();
    }
  }

  private async request<T>(
    method: string,
    path: string,
    options: RequestOptions & { body?: unknown; retryable?: boolean } = {},
  ): Promise<T> {
    const attempts = (options.retryable ?? method === "GET") ? this.maxRetries + 1 : 1;
    let lastError: unknown;
    for (let attempt = 0; attempt < attempts; attempt += 1) {
      const { signal, cleanup } = timeoutSignal(options.signal, options.timeoutMs ?? this.timeoutMs);
      try {
        const response = await this.fetchImpl(`${this.baseUrl}${path}`, {
          method,
          headers: this.headers(),
          body: options.body === undefined ? undefined : JSON.stringify(options.body),
          signal,
        });
        if (response.ok) {
          if (response.status === 204) return undefined as T;
          return (await response.json()) as T;
        }
        const error = await apiError(response);
        lastError = error;
        if (!error.retryable || attempt + 1 >= attempts) throw error;
        await sleep(backoffMs(attempt, response.headers.get("Retry-After")), options.signal);
      } catch (error) {
        lastError = error;
        if (error instanceof RagbotApiError) throw error;
        if (options.signal?.aborted) throw error;
        if (attempt + 1 >= attempts) throw error;
        await sleep(backoffMs(attempt), options.signal);
      } finally {
        cleanup();
      }
    }
    throw lastError;
  }

  private headers(): Record<string, string> {
    const headers: Record<string, string> = {
      Accept: "application/json",
      "Content-Type": "application/json",
    };
    if (this.apiKey) headers["X-API-Key"] = this.apiKey;
    return headers;
  }
}

export async function chat(
  baseUrl: string,
  payload: ChatRequest,
  apiKey?: string,
): Promise<ChatResponse> {
  return new RagbotClient(baseUrl, { apiKey }).chat(payload);
}

export async function chatStream(
  baseUrl: string,
  payload: ChatRequest,
  onEvent: SSEEventHandler,
  apiKey?: string,
): Promise<void> {
  return new RagbotClient(baseUrl, { apiKey }).chatStream(payload, onEvent);
}

async function apiError(response: Response): Promise<RagbotApiError> {
  let payload: unknown;
  try {
    payload = await response.json();
  } catch {
    payload = undefined;
  }
  const envelope = isRecord(payload) && isRecord(payload.error) ? payload.error : undefined;
  if (envelope) {
    return new RagbotApiError({
      statusCode: response.status,
      code: String(envelope.code ?? "http_error"),
      message: String(envelope.message ?? `HTTP ${response.status}`),
      requestId: typeof envelope.request_id === "string" ? envelope.request_id : response.headers.get("X-Request-ID") ?? undefined,
      retryable: Boolean(envelope.retryable ?? retryableStatus(response.status)),
      details: envelope.details,
    });
  }
  const detail = isRecord(payload) ? payload.detail : undefined;
  return new RagbotApiError({
    statusCode: response.status,
    code: "http_error",
    message: String(detail ?? `HTTP ${response.status}`),
    requestId: response.headers.get("X-Request-ID") ?? undefined,
    retryable: retryableStatus(response.status),
    details: detail,
  });
}

function timeoutSignal(parent: AbortSignal | undefined, timeoutMs: number): {
  signal: AbortSignal;
  cleanup: () => void;
} {
  const controller = new AbortController();
  const abort = () => controller.abort(parent?.reason);
  if (parent?.aborted) controller.abort(parent.reason);
  else parent?.addEventListener("abort", abort, { once: true });
  const timer = setTimeout(() => controller.abort(new Error("Ragbot request timed out")), timeoutMs);
  return {
    signal: controller.signal,
    cleanup: () => {
      clearTimeout(timer);
      parent?.removeEventListener("abort", abort);
    },
  };
}

function retryableStatus(status: number): boolean {
  return [429, 502, 503, 504].includes(status);
}

function backoffMs(attempt: number, retryAfter?: string | null): number {
  if (retryAfter) {
    const parsed = Number(retryAfter);
    if (Number.isFinite(parsed)) return Math.max(0, Math.min(parsed * 1000, 30_000));
  }
  return Math.min(5_000, 250 * 2 ** attempt) + Math.random() * 50;
}

async function sleep(ms: number, signal?: AbortSignal): Promise<void> {
  if (signal?.aborted) throw signal.reason;
  await new Promise<void>((resolve, reject) => {
    const timer = setTimeout(resolve, ms);
    const abort = () => {
      clearTimeout(timer);
      reject(signal?.reason ?? new Error("aborted"));
    };
    signal?.addEventListener("abort", abort, { once: true });
  });
}

function queryString(values: Record<string, string | number | undefined>): string {
  const params = new URLSearchParams();
  for (const [key, value] of Object.entries(values)) {
    if (value !== undefined) params.set(key, String(value));
  }
  const encoded = params.toString();
  return encoded ? `?${encoded}` : "";
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
