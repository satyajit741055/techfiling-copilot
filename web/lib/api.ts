import type { FilingsResponse, QuestionResponse, HealthResponse } from '@/types';
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
 * Generic fetch wrapper with error handling
 */
async function fetchAPI<T>(
  endpoint: string,
  options?: RequestInit
): Promise<{ data: T; headers: Headers }> {
  const url = `${API_BASE}${endpoint}`;

  // Get JWT token if user is authenticated
  const token = await getAccessToken();

  let response: Response;
  try {
    response = await fetch(url, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...(token && { Authorization: `Bearer ${token}` }),
        ...options?.headers,
      },
    });
  } catch {
    throw new APIError(
      'Failed to connect to server. Please ensure the backend is running.',
      0,
      'Network error'
    );
  }

  if (!response.ok) {
    let detail = 'Request failed';
    let retryAfter: number | undefined;
    try {
      const body = await response.json();
      const d = body?.detail;
      if (typeof d === 'string') {
        detail = d;
      } else if (d && typeof d === 'object' && !Array.isArray(d)) {
        detail = d.error ?? detail;
        retryAfter = typeof d.retry_after === 'number' ? d.retry_after : undefined;
      }
    } catch {
      // Response wasn't JSON
    }
    retryAfter ??= Number(response.headers.get('retry-after')) || undefined;
    throw new APIError(detail, response.status, detail, retryAfter);
  }

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
