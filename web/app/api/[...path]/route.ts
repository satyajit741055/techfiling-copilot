import type { NextRequest } from 'next/server';

// Server-side proxy to the FastAPI backend. The browser only talks to this
// origin; the backend API key (API_ACCESS_KEY) is added here and never reaches
// the client.

export const dynamic = 'force-dynamic';

const UPSTREAM_TIMEOUT_MS = 120_000;

// Only the endpoints the UI needs are forwarded.
const ALLOWED: Record<string, string[]> = {
  GET: ['health', 'filings'],
  POST: ['questions'],
};

// Backend response headers the UI needs (quota display, retry timing, support reference).
const FORWARDED_RESPONSE_HEADERS = [
  'retry-after',
  'x-ratelimit-limit',
  'x-ratelimit-remaining',
  'x-ratelimit-reset',
  'x-correlation-id',
];

function errorResponse(status: number, detail: string): Response {
  return Response.json({ detail }, { status });
}

async function proxy(
  request: NextRequest,
  { params }: { params: Promise<{ path: string[] }> }
): Promise<Response> {
  const { path } = await params;
  const target = path.join('/');

  if (!ALLOWED[request.method]?.includes(target)) {
    return errorResponse(404, 'Not found');
  }

  const backendUrl = (process.env.BACKEND_URL || 'http://localhost:8000').replace(/\/+$/, '');
  const apiKey = process.env.API_ACCESS_KEY;

  const headers: Record<string, string> = {};
  const contentType = request.headers.get('content-type');
  if (contentType) headers['Content-Type'] = contentType;
  if (apiKey) headers['X-API-Key'] = apiKey;

  // Forward Authorization header if present (for JWT authentication)
  const authHeader = request.headers.get('authorization');
  if (authHeader) headers['Authorization'] = authHeader;

  try {
    const upstream = await fetch(`${backendUrl}/${target}`, {
      method: request.method,
      headers,
      body: request.method === 'POST' ? await request.text() : undefined,
      cache: 'no-store',
      signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS),
    });

    const responseHeaders: Record<string, string> = {
      'Content-Type': upstream.headers.get('content-type') || 'application/json',
    };
    for (const name of FORWARDED_RESPONSE_HEADERS) {
      const value = upstream.headers.get(name);
      if (value) responseHeaders[name] = value;
    }

    return new Response(await upstream.text(), {
      status: upstream.status,
      headers: responseHeaders,
    });
  } catch {
    return errorResponse(502, 'Backend unavailable');
  }
}

export const GET = proxy;
export const POST = proxy;
