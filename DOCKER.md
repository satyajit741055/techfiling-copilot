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
- API docs: http://localhost:8000/docs

Stop with `docker compose down`.

## Frontend API URL

The browser calls the backend directly. The URL is inlined into the frontend bundle at build time
(`NEXT_PUBLIC_API_URL`) and defaults to `http://localhost:8000`. To change it, set
`DOCKER_PUBLIC_API_URL` and rebuild with `docker compose build frontend`.

