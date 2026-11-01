import type {
  ChatEvent,
  ChatRequestBody,
  FilingsResponse,
  HealthResponse,
  QuestionResponse,
  StoredMessage,
  ThreadSummary,
} from '@/types';
import { getAccessToken } from '@/lib/supabase';

// Calls go to this app's own server-side proxy (app/api/[...path]/route.ts),
// which adds the backend API key. No secret is ever shipped to the browser.
const API_BASE = '/api';

/**
 * Custom error class for API errors
 */
export class APIError extends Error {
  constructor(
    message: string,
    public status: number,
    public detail?: string,
    public retryAfter?: number
  ) {
    super(message);
    this.name = 'APIError';
  }
}

/**
 * Remaining request quota, from the backend's X-RateLimit-* headers
 */
export interface Quota {
  limit: number;
  remaining: number;
  receivedAt: number;
}

function readQuota(headers: Headers): Quota | null {
  const limit = Number(headers.get('x-ratelimit-limit'));
  const remaining = Number(headers.get('x-ratelimit-remaining'));
  if (!headers.has('x-ratelimit-limit') || Number.isNaN(limit) || Number.isNaN(remaining)) return null;
  return { limit, remaining, receivedAt: Date.now() };
}

/**
 * fetch() with the JSON content type and the user's JWT; network failures become APIError(0)
 */
async function send(endpoint: string, options?: RequestInit): Promise<Response> {
  // Get JWT token if user is authenticated
  const token = await getAccessToken();
  try {
    return await fetch(`${API_BASE}${endpoint}`, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...(token && { Authorization: `Bearer ${token}` }),
        ...options?.headers,
      },
    });
  } catch (e) {
    if (e instanceof DOMException && e.name === 'AbortError') throw e;
    throw new APIError(
      'Failed to connect to server. Please ensure the backend is running.',
      0,
      'Network error'
    );
  }
}

async function errorFrom(response: Response): Promise<APIError> {
  let detail = 'Request failed';
  let retryAfter: number | undefined;
  try {
    const body = await response.json();
    const d = body?.detail;
    if (typeof d === 'string') {
      detail = d;
    } else if (Array.isArray(d) && typeof d[0]?.msg === 'string') {
      detail = d[0].msg;
    } else if (d && typeof d === 'object') {
      detail = d.error ?? detail;
      retryAfter = typeof d.retry_after === 'number' ? d.retry_after : undefined;
    }
  } catch {
    // Response wasn't JSON
  }
  retryAfter ??= Number(response.headers.get('retry-after')) || undefined;
  return new APIError(detail, response.status, detail, retryAfter);
}

/**
 * Generic fetch wrapper with error handling
 */
async function fetchAPI<T>(
  endpoint: string,
  options?: RequestInit
): Promise<{ data: T; headers: Headers }> {
  const response = await send(endpoint, options);
  if (!response.ok) throw await errorFrom(response);
  return { data: (await response.json()) as T, headers: response.headers };
}

/**
 * Turn any thrown error into a message that is safe and useful to show users
 */
export function friendlyError(e: unknown): { message: string; retryAfter?: number; status: number } {
  if (!(e instanceof APIError)) return { message: 'Something went wrong. Please try again.', status: -1 };
  switch (e.status) {
    case 401:
      return { message: 'Your session has expired. Please sign in again.', status: 401 };
    case 429:
      return {
        message: e.retryAfter
          ? `You have reached the question limit. Try again in ${e.retryAfter}s.`
          : 'You have reached the question limit. Please wait a moment and try again.',
        retryAfter: e.retryAfter,
        status: 429,
      };
    case 0:
      return { message: e.message, status: 0 };
    default:
      return { message: e.status >= 500 ? e.message : e.detail || e.message, status: e.status };
  }
}

/**
 * Check API health
 */
export async function getHealth(): Promise<HealthResponse> {
  return (await fetchAPI<HealthResponse>('/health')).data;
}

/**
 * Get list of available filings
 */
export async function getFilings(): Promise<FilingsResponse> {
  return (await fetchAPI<FilingsResponse>('/filings')).data;
}

/**
 * Ask a question about a filing
 */
export async function askQuestion(
  question: string,
  doc_id: string,
  context_k?: number
): Promise<{ response: QuestionResponse; quota: Quota | null }> {
  const { data, headers } = await fetchAPI<QuestionResponse>('/questions', {
    method: 'POST',
    body: JSON.stringify({
      question,
      doc_id,
      ...(context_k !== undefined && { context_k }),
    }),
  });
  return { response: data, quota: readQuota(headers) };
}

/**
 * Parse one SSE block ("event: x\ndata: {...}") into a chat event; comments are ignored
 */
export function parseSSEBlock(block: string): ChatEvent | null {
  let event: string | null = null;
  const data: string[] = [];
  for (const line of block.split('\n')) {
    if (line.startsWith(':')) continue;
    const colon = line.indexOf(':');
    const field = colon === -1 ? line : line.slice(0, colon);
    const value = colon === -1 ? '' : line.slice(colon + 1).replace(/^ /, '');
    if (field === 'event') event = value;
    else if (field === 'data') data.push(value);
  }
  if (!event || data.length === 0) return null;
  try {
    return { event, data: JSON.parse(data.join('\n')) } as ChatEvent;
  } catch {
    return null;
  }
}

/**
 * Stream one research-chat turn. Calls onEvent for every server event, in order.
 * Browser EventSource is GET-only, so this reads the POST response body directly.
 * Abort `signal` to stop; the backend then cancels the run.
 */
export async function streamChat(
  body: ChatRequestBody,
  onEvent: (event: ChatEvent) => void,
  signal?: AbortSignal
): Promise<Quota | null> {
  const response = await send('/chat', { method: 'POST', body: JSON.stringify(body), signal });
  if (!response.ok) throw await errorFrom(response);
  if (!response.body) throw new APIError('The server returned an empty response.', 502);

  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = '';
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += value.replace(/\r\n?/g, '\n');
    let end: number;
    while ((end = buffer.indexOf('\n\n')) !== -1) {
      const event = parseSSEBlock(buffer.slice(0, end));
      buffer = buffer.slice(end + 2);
      if (event) onEvent(event);
    }
  }
  const tail = parseSSEBlock(buffer.trim());
  if (tail) onEvent(tail);
  return readQuota(response.headers);
}

/**
 * The signed-in user's saved threads. Throws APIError(404) when server history is off.
 */
export async function getThreads(): Promise<ThreadSummary[]> {
  return (await fetchAPI<{ threads: ThreadSummary[] }>('/threads')).data.threads;
}

/**
 * One saved thread's turns, oldest first
 */
export async function getThreadMessages(threadId: string): Promise<StoredMessage[]> {
  return (await fetchAPI<{ messages: StoredMessage[] }>(`/threads/${threadId}/messages`)).data
    .messages;
}
