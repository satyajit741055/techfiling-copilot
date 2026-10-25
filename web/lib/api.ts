import type { FilingsResponse, QuestionResponse, HealthResponse, ErrorResponse } from '@/types';

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
    public detail?: string
  ) {
    super(message);
    this.name = 'APIError';
  }
}

/**
 * Generic fetch wrapper with error handling
 */
async function fetchAPI<T>(
  endpoint: string,
  options?: RequestInit
): Promise<T> {
  const url = `${API_BASE}${endpoint}`;
  
  try {
    const response = await fetch(url, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...options?.headers,
      },
    });

    if (!response.ok) {
      let detail = 'Request failed';
      try {
        const errorData: ErrorResponse = await response.json();
        detail = errorData.detail || detail;
      } catch {
        // Response wasn't JSON
      }
      throw new APIError(detail, response.status, detail);
    }

    return response.json();
  } catch (error) {
    if (error instanceof APIError) {
      throw error;
    }
    // Network error or other fetch failure
    throw new APIError(
      'Failed to connect to server. Please ensure the backend is running.',
      0,
      'Network error'
    );
  }
}

/**
 * Check API health
 */
export async function getHealth(): Promise<HealthResponse> {
  return fetchAPI<HealthResponse>('/health');
}

/**
 * Get list of available filings
 */
export async function getFilings(): Promise<FilingsResponse> {
  return fetchAPI<FilingsResponse>('/filings');
}

/**
 * Ask a question about a filing
 */
export async function askQuestion(
  question: string,
  doc_id: string,
  context_k?: number
): Promise<QuestionResponse> {
  return fetchAPI<QuestionResponse>('/questions', {
    method: 'POST',
    body: JSON.stringify({
      question,
      doc_id,
      ...(context_k !== undefined && { context_k }),
    }),
  });
}
