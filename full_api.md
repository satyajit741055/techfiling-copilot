# Full API Mode

This configuration runs all model inference through hosted APIs:

| Stage | Service |
|---|---|
| Query and document embeddings | Hugging Face feature-extraction API |
| Reranking | Hugging Face serverless or dedicated reranker endpoint |
| Answer generation | Hugging Face Inference Providers using Qwen 2.5 7B |
| DeepEval judging | OpenAI API |

BM25, vector similarity, stored-index access, context formatting, and metric aggregation continue to run locally because they are algorithms rather than model inference.

## 1. Configure `.env`

Start from the ready-to-copy hosted template:

```powershell
Copy-Item .env.huggingface.example .env
```

Then replace the placeholder Hugging Face and OpenAI keys. The relevant configuration is:

```dotenv
# =============================================================================
# EXECUTION MODE
# =============================================================================
MODEL_EXECUTION_MODE=huggingface_api

# Leave blank so every component inherits MODEL_EXECUTION_MODE
EMBEDDING_EXECUTION_MODE=
RERANKER_EXECUTION_MODE=
GENERATION_EXECUTION_MODE=

# =============================================================================
# HUGGING FACE
# =============================================================================
HUGGINGFACE_API_KEY=hf_your_actual_token

# Answer generation
HUGGINGFACE_MODEL=Qwen/Qwen2.5-7B-Instruct
HUGGINGFACE_PROVIDER=auto

# Embeddings
EMBEDDING_MODEL=BAAI/bge-base-en-v1.5
HUGGINGFACE_EMBEDDING_PROVIDER=hf-inference

# Reranking
RERANKER_MODEL=cross-encoder/ettin-reranker-150m-v1
RERANKER_MODEL_PROD=cross-encoder/ettin-reranker-150m-v1
HUGGINGFACE_RERANKER_URL=https://router.huggingface.co/hf-inference/models/{model}
HUGGINGFACE_TIMEOUT=120

# =============================================================================
# DEEPEVAL JUDGE
# =============================================================================
OPENAI_API_KEY=sk_your_actual_openai_key
DEEPEVAL_JUDGE_MODEL=gpt-4o-mini
```

Do not commit `.env` or real API keys.

## 2. Settings that are unused in full API mode

These values can remain in `.env`, but they are not used for embedding, reranking, or generation when all effective execution modes are `huggingface_api`:

```dotenv
LLM_PROVIDER=ollama
OLLAMA_MODEL=qwen2.5-7b-instruct:latest
OLLAMA_HOST=http://localhost:11434
DEVICE=auto
```

In full API mode:

- The generation provider resolves to `huggingface` automatically.
- Ollama does not need to run.
- `DEVICE` does not control embedding or reranking because those models run remotely.
- `HUGGINGFACE_PROVIDER=auto` selects a provider serving `Qwen/Qwen2.5-7B-Instruct`.

## 3. Confirm configuration without API calls

Run from the repository root:

```powershell
uv run python eval/deepEval/full_pipeline_eval.py --dry-run --limit 1
```

Expected resolved configuration:

```text
embedding_mode: huggingface_api
reranker_mode: huggingface_api
generator: huggingface/Qwen/Qwen2.5-7B-Instruct
judge_model: gpt-4o-mini
```

The dry run validates the dataset and mode resolution but does not run retrieval, generation, or DeepEval judge calls.

## 4. Run one real end-to-end case

```powershell
uv run python eval/deepEval/full_pipeline_eval.py `
  --limit 1 `
  --fail-fast `
  --verbose
```

This performs:

```text
Query embedding       -> Hugging Face API
Hybrid retrieval      -> local vector index + local BM25
Reranking             -> Hugging Face API
Answer generation     -> Hugging Face API, Qwen 2.5 7B
DeepEval judging      -> OpenAI API
```

Start with one case because it verifies all external services while limiting cost.

## 5. Existing embedding-index warning

The current active chunk run may have been embedded locally:

```dotenv
RUN_DIRECTORY=data/chunks/2026-10-20_16-33-24
```

When the active query embedder uses the Hugging Face API against a locally embedded index, the retriever warns:

```text
Index was embedded with the 'local' backend but queries use the
'huggingface_api' backend; same model, but re-embedding is recommended
for exact parity.
```

The model name and vector dimension must still match. A different model or dimension causes an error rather than silently returning invalid similarity scores.

Using the same BGE model is sufficient for an initial smoke test. For a controlled full-API experiment, regenerate the stored document embeddings through the API.

## 6. Rebuild document embeddings through the API

With full API mode enabled in `.env`, run from the repository root:

```powershell
$env:PYTHONPATH="src"
uv run python -m ingestion.processor --skip-parse
```

This creates:

```text
data/chunks/<new-timestamp>/
```

Update `.env` to point to it:

```dotenv
RUN_DIRECTORY=data/chunks/<new-timestamp>
```

The new `run_manifest.json` should include:

```json
{
  "embedding_model": {
    "name": "BAAI/bge-base-en-v1.5",
    "backend": "huggingface_api",
    "normalized": true,
    "query_prefix": "Represent this sentence for searching relevant passages: "
  }
}
```

Embedding every document uses many hosted requests and may incur usage costs. Keep the generated run so it can be reused by subsequent retrieval and evaluation runs.

## 7. Test generation without DeepEval

From the repository root:

```powershell
$env:PYTHONPATH="src"

uv run python -m generation.generator `
  "What was Apple's total net sales in fiscal 2025?" `
  --doc-id AAPL_10-K_2025
```

Alternatively:

```powershell
cd src

uv run python -m generation.generator `
  "What was Apple's total net sales in fiscal 2025?" `
  --doc-id AAPL_10-K_2025
```

With API generation active, the metadata should show:

```text
Model: Qwen/Qwen2.5-7B-Instruct
```

## 8. Reranker endpoint options

The default serverless endpoint is:

```dotenv
HUGGINGFACE_RERANKER_URL=https://router.huggingface.co/hf-inference/models/{model}
```

`{model}` is replaced with the active reranker model.

For a dedicated Text Embeddings Inference endpoint, use a URL ending in `/rerank`:

```dotenv
HUGGINGFACE_RERANKER_URL=https://your-endpoint.endpoints.huggingface.cloud/rerank
```

The runtime selects the request format from the URL:

| Endpoint | Request format |
|---|---|
| URL ending in `/rerank` | TEI: `{"query": "...", "texts": ["..."], "truncate": true}` |
| Hugging Face serverless URL | Text ranking: `{"inputs": {"query": "...", "texts": ["..."]}}` |

## 9. Component-level API modes

A global API mode is not required. Individual stages can be hosted while the others stay local.

Example: only reranking through the API:

```dotenv
MODEL_EXECUTION_MODE=local
EMBEDDING_EXECUTION_MODE=
RERANKER_EXECUTION_MODE=huggingface_api
GENERATION_EXECUTION_MODE=
HUGGINGFACE_API_KEY=hf_your_actual_token
```

Example: local embedding and reranking with hosted generation:

```dotenv
MODEL_EXECUTION_MODE=local
GENERATION_EXECUTION_MODE=huggingface_api
HUGGINGFACE_API_KEY=hf_your_actual_token
```

Resolution order is:

```text
Explicit CLI option
        -> component execution-mode override
        -> MODEL_EXECUTION_MODE
        -> code default
```

## 10. Minimum configuration for an initial smoke test

The minimum changes are:

```dotenv
MODEL_EXECUTION_MODE=huggingface_api
HUGGINGFACE_API_KEY=hf_your_actual_token
OPENAI_API_KEY=sk_your_actual_openai_key
```

All model names, providers, endpoints, timeouts, and the DeepEval judge already have defaults in `src/config/settings.py`.
