# Evaluation Suite

Every evaluation script for TechFiling Copilot lives here. All commands run from the **repository root** with `uv run python ...`.

## What is evaluated

| Question | Script | Needs LLM judge (OpenAI)? | Results folder |
|---|---|---|---|
| Does retrieval find the right chunks? (Hit/Recall/MRR/nDCG/Precision) | `dense_retriever_eval.py`, `hybrid_rerank_eval.py` | No | `retriver_results/` |
| Does the generator answer correctly from ideal context? (rule-based) | `generation_eval.py` | No | `generation_results/` |
| Retriever quality judged by an LLM | `deepEval/retriever_eval.py` | Yes | `deepEval/retriever_results/` |
| Generator quality judged by an LLM (faithfulness, relevancy, correctness) | `deepEval/generator_eval.py` | Yes | `deepEval/generation_results/` |
| Full pipeline RAG triad (contextual relevancy, faithfulness, answer relevancy) | `deepEval/full_pipeline_eval.py` | Yes | `deepEval/full_pipeline_results/` |
| All three DeepEval evals in one go | `deepEval/run_all_evaluations.py` | Yes | the three folders above |
| Latency (P50/P95/P99), cost per query, reliability | `performance/performance_eval.py` | No | `performance/results/` |

Other folders:

- `V2_dataset_experiements/` and `old/` hold written reports and archived results from earlier experiments. They are not used by any script.
- `deepEval/inspect_*.ipynb` are notebooks for browsing saved DeepEval runs.
- Golden datasets live in `../goldens/`. Deeper explanations are in `../study/evals/`.

---

## 1. Setup (first time)

### Prerequisites

- Python 3.13+, [`uv`](https://docs.astral.sh/uv/), Git
- [Ollama](https://ollama.com) if you use local generation
- An OpenAI API key, only for the DeepEval scripts (the judge model)
- A Hugging Face API key, only if you use `MODEL_EXECUTION_MODE=huggingface_api`

### Install

```powershell
git clone <repo-url>
cd techfiling-copilot
uv sync
```

On a corporate network that intercepts TLS, add packages with `uv --system-certs add <pkg>`.

### Configure `.env`

Pick one template:

```powershell
Copy-Item .env.local.example .env          # local models: Ollama + local embedding/reranker
# or
Copy-Item .env.huggingface.example .env    # Hugging Face hosted models
```

Settings that matter for evals (schema and defaults: `src/config/settings.py`):

| Variable | Purpose |
|---|---|
| `MODEL_EXECUTION_MODE` | `local` or `huggingface_api`. Chooses where embedding, reranking and generation run. |
| `LLM_PROVIDER`, `OLLAMA_MODEL` | Generator provider and model, for example `ollama` and `qwen2.5-7b-instruct:latest`. |
| `DEVICE` | `auto`, `cpu` or `cuda` for local embedding and reranker models. |
| `VECTOR_STORE` | `numpy` (default, no extra services) or `qdrant`. |
| `RETRIEVAL_ALPHA`, `RETRIEVAL_K`, `CONTEXT_K` | Retrieval tuning used when a script flag is not given. |
| `OPENAI_API_KEY`, `DEEPEVAL_JUDGE_MODEL` | Required by DeepEval scripts (judge, default `gpt-4o-mini`). |
| `HUGGINGFACE_API_KEY` | Required only for Hugging Face mode. |
| `API_ACCESS_KEY` | Backend API key. `performance_eval.py api` reads it from `.env` automatically. |
| `METRICS_*`, `LLM_*_PRICE_PER_1M`, `LOCAL_COMPUTE_COST_PER_HOUR` | Operational eval (see section 6). |

Never print or commit `.env`; it holds secrets.

### Local models (Ollama)

```powershell
ollama pull qwen2.5-7b-instruct:latest
ollama list          # confirms Ollama is running; run `ollama serve` only if it is not
```

The first run also downloads the embedding model (`BAAI/bge-base-en-v1.5`) and reranker (`cross-encoder/ettin-reranker-150m-v1`) from Hugging Face.

### Data

The repo ships the parsed filings and chunk/embedding runs under `data/`, so no ingestion step is needed. For Qdrant mode:

```powershell
docker compose up -d qdrant
uv run python scripts/migrate_npy_to_qdrant.py
```

### Verify the setup without calling any model

```powershell
uv run python -m pytest tests -q                                      # note: tests/test_api.py loads the real pipeline and takes ~4 min
uv run python eval/deepEval/full_pipeline_eval.py --dry-run
uv run python eval/deepEval/generator_eval.py --dry-run
uv run python eval/performance/performance_eval.py direct --dry-run
```

---

## 2. Retrieval evals (no LLM judge)

Golden dataset: `goldens/golden_retriever_dataset_v2.jsonl`. Metrics: Hit Rate, Recall, MRR, nDCG and Precision at each k.

### `dense_retriever_eval.py` (dense-only baseline)

```powershell
uv run python eval/dense_retriever_eval.py
uv run python eval/dense_retriever_eval.py -k 1 3 5 10 20
uv run python eval/dense_retriever_eval.py --run-directory data/chunks/<run> --device cpu
uv run python eval/dense_retriever_eval.py --analyze-failures   # show queries that missed
uv run python eval/dense_retriever_eval.py --compare            # table of all earlier runs
```

Other flags: `--golden <path>`, `--output-dir`, `--no-save`.

### `hybrid_rerank_eval.py` (dense, BM25, hybrid, RRF, reranker, Qdrant)

`--mode` is one of `dense`, `bm25`, `hybrid` (default), `rerank`, `rrf`, `rrf_rerank`, `qdrant`, `qdrant_dense`, `qdrant_sparse`, `qdrant_rerank`.

```powershell
uv run python eval/hybrid_rerank_eval.py --mode hybrid --alpha 0.85
uv run python eval/hybrid_rerank_eval.py --mode rerank --retrieve-k 50
uv run python eval/hybrid_rerank_eval.py --mode rrf_rerank --rrf-k 60 --dense-weight 1 --bm25-weight 1
uv run python eval/hybrid_rerank_eval.py --tune-alpha            # sweep dense/BM25 weight
uv run python eval/hybrid_rerank_eval.py --mode qdrant_rerank    # needs Qdrant running
uv run python eval/hybrid_rerank_eval.py --compare               # compare all runs
```

Other flags: `--k`, `--reranker-model`, `--run-directory`, `--golden-path`, `--device`, `--dense-k`, `--bm25-k`.

Results are JSON files in `eval/retriver_results/` (the folder name is spelled that way).

---

## 3. Generator eval, rule-based (`generation_eval.py`)

Tests the generator in isolation on frozen golden contexts, scored with rules (numeric matching, fact coverage, citation validity). Golden dataset: `goldens/golden_generation_dataset.jsonl`.

```powershell
uv run python eval/generation_eval.py --limit 5 --verbose
uv run python eval/generation_eval.py --provider ollama --model qwen2.5-7b-instruct:latest
uv run python eval/generation_eval.py --provider huggingface --model Qwen/Qwen2.5-7B-Instruct
uv run python eval/generation_eval.py --prompt-version v2_complete_precise --fact-threshold 0.6
uv run python eval/generation_eval.py --case-ids <id> <id>
uv run python eval/generation_eval.py --compare
```

Each run writes `cases.jsonl`, `manifest.json` and `summary.json` to `eval/generation_results/<timestamp>_<provider>_<model>/`.

---

## 4. DeepEval LLM-judge evals (`deepEval/`)

These use an OpenAI judge model, so they need `OPENAI_API_KEY` and cost money. Start small with `--limit` and try `--dry-run` first (prints configuration, makes no model calls).

Shared flags: `--judge-model`, `--threshold`, `--strict`, `--limit`, `--dry-run`, `--verbose`, `--fail-fast`, `--compare` (compare saved runs), `--golden-path`, `--output-dir`.
Generator flags: `--provider`, `--model`, `--temperature`, `--max-tokens`, `--prompt-version`.

| Script | Metrics | Default golden dataset | Extra flags |
|---|---|---|---|
| `retriever_eval.py` | contextual precision, contextual recall | `golden_retriever_dataset_v3_enriched.jsonl` | `--context-k`, `--query-ids`, `--categories`, `--difficulties` |
| `generator_eval.py` | faithfulness, answer relevancy, answer correctness | `golden_generator_isolation_dataset.jsonl` | `--metrics`, `--case-ids` |
| `full_pipeline_eval.py` | contextual relevancy, faithfulness, answer relevancy | `golden_retriever_dataset_v3.jsonl` | `--context-k`, `--vector-store`, `--device`, `--metrics`, `--query-ids`, `--categories`, `--difficulties` |

```powershell
uv run python eval/deepEval/retriever_eval.py --limit 5 --verbose
uv run python eval/deepEval/generator_eval.py --metrics faithfulness answer_relevancy --limit 5
uv run python eval/deepEval/full_pipeline_eval.py --limit 5 --provider ollama
uv run python eval/deepEval/full_pipeline_eval.py --compare
```

### Run all three together

```powershell
uv run python eval/deepEval/run_all_evaluations.py --judge-model gpt-4o-mini
uv run python eval/deepEval/run_all_evaluations.py --skip-retriever --skip-generator
uv run python eval/deepEval/run_all_evaluations.py --compare      # compare latest runs, no new evaluation
```

Each run writes `cases.jsonl`, `manifest.json` and `summary.json` into its own timestamped folder. Open the `inspect_*.ipynb` notebooks to browse them.

---

## 5. Which eval should I use?

| Goal | Use |
|---|---|
| Tune retrieval (alpha, RRF weights, reranker) quickly and for free | `hybrid_rerank_eval.py` |
| Check the generator, offline and cheaply | `generation_eval.py` |
| Semantic quality scores for a report | `deepEval/*` |
| Compare models or prompts end to end | `deepEval/full_pipeline_eval.py --compare` |
| How fast, how costly, how reliable | `performance/performance_eval.py` |

---

## 6. Operational eval: latency, cost, reliability (`performance/`)

Measures P50/P95/P99 latency per component (retrieval, generation, total, and client end-to-end in API mode), cold versus warm requests, cost per query, and success/error/timeout/retry rates. It shares code with the API (`src/observability/`), so eval and production numbers match. The API also writes one JSONL record per `/questions` request to `logs/metrics/queries-<date>.jsonl`.

Full guide, formulas, JSONL schema and Prometheus/Grafana design: `../study/evals/06_operational_eval_latency_cost_reliability.md`.

### Modes

| Mode | What it does |
|---|---|
| `direct` | Runs the pipeline in the benchmark's own process. No HTTP, so it shows component timings and cold start. |
| `api` | Sends `POST /questions` to a running backend and shows what a real client sees. |
| `analyze` | Summarizes production logs (`logs/metrics/`) or a saved `queries.jsonl`. |

The mode does not choose the models. `.env` does, through `MODEL_EXECUTION_MODE` (for `api` mode, the backend's `.env`).

### Run

```powershell
# direct mode: stop the backend first so they do not compete for GPU/Ollama
uv run python eval/performance/performance_eval.py direct --limit 5 --repeat 2

# api mode: start the backend WITHOUT --reload in another terminal
uv run uvicorn api.main:app --host 127.0.0.1 --port 8000
uv run python eval/performance/performance_eval.py api --repeat 3 --delay 7

# summarize
uv run python eval/performance/performance_eval.py analyze
uv run python eval/performance/performance_eval.py analyze --logs eval/performance/results/<run>/queries.jsonl
```

### Options

| Flag | Meaning |
|---|---|
| `--limit N` | Use only the first N golden queries |
| `--repeat N` | Passes over the selected queries (default 3) |
| `--requests N` | Total requests (overrides `--repeat`) |
| `--warmup N` | First N requests are reported as cold (default 2) |
| `--categories ...` | Filter golden queries by category |
| `--provider`, `--model` | `direct` mode: override generator |
| `--api-url` | `api` mode: backend URL (default `http://127.0.0.1:8000`) |
| `--api-key`, `--bearer-token` | `api` mode: auth. The API key is read from `.env` if omitted. |
| `--timeout` | `api` mode: client timeout in seconds (default 120) |
| `--delay S` | `api` mode: pause between requests |
| `--dry-run` | Validate setup, call nothing |

Results go to `eval/performance/results/<timestamp>_<mode>/` as `queries.jsonl`, `summary.json` and `manifest.json`.

### Tips for trustworthy numbers

- Run the backend without `--reload`. Any file edit restarts it and causes another cold start.
- Rate limit: the question limit is `RATE_LIMIT_QUESTIONS_PER_MINUTE` (10 in `.env.local.example`, 2 if unset). Use `--delay 7` or raise the limit, otherwise 429 responses count as errors.
- Use `127.0.0.1`, not `localhost`. On Windows `localhost` can add about 2s per new connection.
- The first request after a start loads the retriever (about 30s) and the Ollama model (about 10s). These are reported as cold, not hidden.
- Percentiles need enough samples: about 20 or more warm requests for P95, 100 or more for P99.
- Cost is $0 for local models unless you set `LOCAL_COMPUTE_COST_PER_HOUR`. For paid providers, set `LLM_INPUT_PRICE_PER_1M` and `LLM_OUTPUT_PRICE_PER_1M`, or check the built-in price table against the provider's current prices.
- Requests rejected by auth (401/403) count as errors.

---

## 7. Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'api'` | Run from the repo root, using `uv run python eval/...`. |
| `uvicorn api.app:main` fails to import | The correct target is `api.main:app`. |
| HTTP 401 in API benchmark | `API_ACCESS_KEY` is set in `.env`. Restart the backend, or pass `--api-key`. |
| HTTP 429 in API benchmark | Add `--delay 7` or raise `RATE_LIMIT_QUESTIONS_PER_MINUTE`. |
| `analyze` finds no `logs/metrics` | Only the API writes it. Send requests to the API first, or use `--logs <queries.jsonl>`. |
| First request takes 30-45 s | Cold start (model loading). Expected. |
| DeepEval fails on auth | Set `OPENAI_API_KEY` in `.env`. |
| `--compare` shows nothing | No earlier runs exist yet for that script. |
| Qdrant modes fail | Start Qdrant (`docker compose up -d qdrant`) and run the migration script. |
| TLS errors on a corporate network | Use `uv --system-certs ...` for package installs. |

## 8. Conventions

- Each run is saved in its own timestamped folder or file and is never overwritten.
- Use `--dry-run` or `--limit` before a full run, especially for DeepEval, which calls a paid API.
- Do not commit `.env`.
