import type { NextRequest } from 'next/server';

// Server-side proxy to the FastAPI backend. The browser only talks to this
// origin; the backend API key (API_ACCESS_KEY) is added here and never reaches
// the client.

export const dynamic = 'force-dynamic';

const UPSTREAM_TIMEOUT_MS = 120_000;
// /chat streams a whole research turn: the backend's AGENT_TIMEOUT_SECONDS plus a margin.
const CHAT_TIMEOUT_MS = (Number(process.env.AGENT_TIMEOUT_SECONDS) || 180) * 1000 + 30_000;

// Only the endpoints the UI needs are forwarded.
const ALLOWED: Record<string, RegExp[]> = {
  GET: [
    /^health$/,
    /^filings$/,
    /^threads$/,
    /^threads\/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\/messages$/i,
  ],
  POST: [/^questions$/, /^chat$/],
};
const STREAMED = new Set(['chat']);

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

  if (!ALLOWED[request.method]?.some((pattern) => pattern.test(target))) {
    return errorResponse(404, 'Not found');
  }

  const backendUrl = (process.env.BACKEND_URL || 'http://localhost:8000').replace(/\/+$/, '');
  const apiKey = process.env.API_ACCESS_KEY;
  const streamed = STREAMED.has(target);

  const headers: Record<string, string> = {};
  const contentType = request.headers.get('content-type');
  if (contentType) headers['Content-Type'] = contentType;
  if (apiKey) headers['X-API-Key'] = apiKey;

  // Forward Authorization header if present (for JWT authentication)
  const authHeader = request.headers.get('authorization');
  if (authHeader) headers['Authorization'] = authHeader;

  const search = request.method === 'GET' ? request.nextUrl.search : '';
  // A browser that stops reading (closed tab, "Stop") aborts the upstream request, so the
  // backend sees the disconnect and cancels the research run.
  const signal = AbortSignal.any([
    request.signal,
    AbortSignal.timeout(streamed ? CHAT_TIMEOUT_MS : UPSTREAM_TIMEOUT_MS),
  ]);

  try {
    const upstream = await fetch(`${backendUrl}/${target}${search}`, {
      method: request.method,
      headers,
      body: request.method === 'POST' ? await request.text() : undefined,
      cache: 'no-store',
      signal,
    });

    const responseHeaders: Record<string, string> = {
      'Content-Type': upstream.headers.get('content-type') || 'application/json',
    };
    for (const name of FORWARDED_RESPONSE_HEADERS) {
      const value = upstream.headers.get(name);
      if (value) responseHeaders[name] = value;
    }

    const isEventStream = responseHeaders['Content-Type'].startsWith('text/event-stream');
    if (streamed && isEventStream && upstream.body) {
      return new Response(upstream.body, {
        status: upstream.status,
        headers: {
          ...responseHeaders,
          'Cache-Control': 'no-cache, no-transform',
          'X-Accel-Buffering': 'no',
        },
      });
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
