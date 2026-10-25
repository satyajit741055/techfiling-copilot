# Docker

Runs the FastAPI backend and the Next.js frontend with Docker Compose.

The backend container always runs in **Hugging Face API mode**: embedding, reranking and answer
generation all call Hugging Face APIs. The image does not include torch, sentence-transformers,
transformers or CUDA, so `local` mode, Ollama and GPUs are not supported in Docker.

## Setup

The backend reads the same `.env` as a local run (`env_file: .env`). Compose forces
`MODEL_EXECUTION_MODE=huggingface_api` and clears any per-stage overrides, so the only required
value is the API key:

```bash
HUGGINGFACE_API_KEY=hf_...
```

`.env.huggingface.example` shows the full set of API-mode settings (models, provider, reranker URL).

The chunk and embedding data in `./data` is baked into the backend image, so rebuild the backend
after re-chunking or changing `RUN_DIRECTORY`.

## Run

```bash
docker compose up --build -d
docker compose logs -f backend      # wait for "API ready to serve requests!"
```

- Frontend: http://localhost:3000
- API docs (dev only): http://localhost:8000/docs
- Readiness (dependencies): http://localhost:8000/health/ready

Stop with `docker compose down`.

## Security

**How requests flow:** the browser calls only the frontend (`/api/*`). The frontend's server-side
proxy calls the backend at `http://backend:8000` over the Docker network and adds the
`X-API-Key` header. The key is never sent to the browser. Only `/health`, `/filings` and
`/questions` are forwarded.

Set the same key for both services in `.env` (Compose passes `API_ACCESS_KEY` to the frontend):

```bash
API_ACCESS_KEY=<at least 24 random characters>   # e.g. openssl rand -hex 32
```

| Setting | dev | prod (`ENVIRONMENT=prod`) |
|---|---|---|
| `API_ACCESS_KEY` | optional (auth is off and a warning is logged when unset) | required, 24+ characters, otherwise the backend refuses to start |
| `CORS_ORIGINS` | defaults to `http://localhost:3000,http://127.0.0.1:3000` | must be set explicitly (comma-separated); `*` and the localhost defaults are rejected |
| `/docs`, `/redoc`, `/openapi.json` | enabled | disabled |

Compose publishes the backend on `127.0.0.1:8000` only, for debugging. Remove that `ports` entry in
production so only the frontend is reachable.

CORS only matters if a browser calls the backend directly. With the proxy it normally does not, but
`CORS_ORIGINS` is still validated in prod as defense in depth.

## Health checks

- `GET /health` is a cheap liveness check, used by the Compose healthcheck.
- `GET /health/ready` verifies that filings and the index data are present, and that the Hugging
  Face API key works (or Ollama / the provider key, depending on your mode). It returns `503` with
  per-check details if a dependency is down. It calls external services, so poll it sparingly.

