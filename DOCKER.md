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
after re-chunking or changing `RUN_DIRECTORY`. This is only needed for `VECTOR_STORE=numpy`; with
`VECTOR_STORE=qdrant` the image can ship without any data (see below).

## Qdrant (optional vector store)

Compose includes a `qdrant` service (`qdrant/qdrant:v1.20.0`, data in the `qdrant_data` volume,
published on `127.0.0.1:6333` only). The backend reaches it at `http://qdrant:6333`; compose sets
`QDRANT_URL` for you. By default the backend still uses the baked `.npy` index
(`VECTOR_STORE=numpy`). To use Qdrant instead:

```bash
# 1. in .env
VECTOR_STORE=qdrant
QDRANT_COLLECTION=filings_bge_base_v1

# 2. start Qdrant and load an existing run from the host (no re-embedding)
docker compose up -d qdrant
uv run python scripts/migrate_npy_to_qdrant.py            # uses RUN_DIRECTORY and http://127.0.0.1:6333

# 3. start the rest
docker compose up --build -d
```

Qdrant also holds the **filing catalog** (company, ticker, fiscal year, ...). The migration reads
`data/parsed/<doc_id>.document.json` and stores it on each document's chunks, and `/filings` and the
`/questions` doc-ID check are served from the collection. This makes the backend image stateless:
set `INCLUDE_DATA=false` (in `.env`, which Compose reads for substitution) and rebuild, and the
image contains no `data/` at all. Adding a filing then needs no image rebuild: ingest it into Qdrant
and restart the backend (the catalog is loaded at startup).

```bash
INCLUDE_DATA=false          # in .env; default true (numpy mode needs the baked data)
```

If a collection was loaded before the catalog existed, the backend logs a warning and falls back to
`data/parsed`; re-run the migration to store the catalog.

- The collection must exist and be non-empty before the backend starts, otherwise `/health/ready`
  reports `index_data` as failed and questions fail. Re-running the migration is safe: each
  document's old chunks are replaced.
- The embedding model is stored with the collection and checked at startup. After changing
  `EMBEDDING_MODEL`, re-embed and use a new `QDRANT_COLLECTION` (for example `filings_bge_base_v1` ->
  `filings_<model>_v2`).
- Keep the Qdrant image tag and the `qdrant-client` version (in `uv.lock`) on the same minor version.
- To secure Qdrant in production, run the server with an API key and set `QDRANT_API_KEY` in `.env`.
  This is not configured in `docker-compose.yml` yet. Also keep port 6333 unpublished.
- To ingest new filings straight into Qdrant, run `uv run src/ingestion/processor.py`
  with `VECTOR_STORE=qdrant` and Qdrant running on the host.

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
- `GET /health/ready` verifies that filings and the index data are present (with
  `VECTOR_STORE=qdrant`: that the Qdrant collection exists and has points), and that the Hugging
  Face API key works (or Ollama / the provider key, depending on your mode). It returns `503` with
  per-check details if a dependency is down. It calls external services, so poll it sparingly.

