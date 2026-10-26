# TechFiling Copilot

> SEC 10-K Document Q&A with RAG (Retrieval-Augmented Generation)

A production-ready research assistant that helps equity analysts extract insights from SEC 10-K filings. Ask questions in plain English and get grounded, citation-backed answers.

## Features

- **Grounded Answers**: Every answer is backed by explicit citations from the filing
- **Citation Traceability**: Click any citation to view the exact source passage
- **Hybrid Retrieval**: BM25 + Dense embeddings + Cross-encoder reranking for high-quality results
- **Scalable Vector Store**: default in-process NumPy index, or Qdrant (dense + sparse hybrid search with server-side RRF fusion) via `VECTOR_STORE=qdrant`
- **Multi-Provider LLM**: Supports Ollama (local), OpenAI, Anthropic, Google, and Hugging Face
- **Financial Table Support**: Properly renders and extracts data from complex financial tables

## Supported Filings

| Company | Ticker | Filing | Fiscal Year |
|---------|--------|--------|-------------|
| Apple Inc. | AAPL | 10-K | FY2025 |
| Microsoft Corporation | MSFT | 10-K | FY2026 |
| Tesla, Inc. | TSLA | 10-K | FY2025 |

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Frontend (Next.js)                          │
│  Filing Selector │ Question Input │ Answer Card │ Source Cards  │
└─────────────────────────────────────────────────────────────────┘
                              │ REST API
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Backend (FastAPI)                           │
│  /health │ /filings │ /questions                                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     RAG Pipeline                                │
│  Hybrid Retriever → Cross-Encoder Reranker → LLM Generator      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Data Layer                                  │
│  numpy (default): Chunks (.jsonl) │ Embeddings (.npy) │ BM25    │
│  qdrant (optional): dense + sparse vectors, payload filters,    │
│                     filing catalog                              │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

- Python 3.13+
- Node.js 18+
- [uv](https://github.com/astral-sh/uv) (Python package manager)
- [Ollama](https://ollama.ai/) (for local LLM) or API keys for cloud providers

### 1. Clone and Setup

```bash
git clone https://github.com/satyajit741055/techfiling-copilot.git
cd techfiling-copilot

# Install Python dependencies
uv sync

# Choose one environment template
cp .env.local.example .env          # local CUDA/CPU + Ollama
# or
cp .env.huggingface.example .env    # fully hosted Hugging Face APIs

# .env.example remains the combined reference containing every option.
```

### 2. Start Ollama (if using local LLM)

```bash
# Pull the default model
ollama pull qwen2.5-7b-instruct:latest

# Ensure Ollama is running
ollama serve
```

### 3. Start the Backend

```bash
# From project root
uv run uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### 4. Start the Frontend

```bash
cd web
npm install
npm run dev
```

Open http://localhost:3000 in your browser.

## Configuration

### Environment Variables

Copy `.env.example` to `.env` and configure:

```bash
# Run embedding, reranking and generation locally, or through Hugging Face APIs
MODEL_EXECUTION_MODE=local  # local | huggingface_api

# Optional per-stage overrides (empty = follow MODEL_EXECUTION_MODE)
EMBEDDING_EXECUTION_MODE=
RERANKER_EXECUTION_MODE=
GENERATION_EXECUTION_MODE=

# Local generation uses this provider (normally Ollama + Qwen)
LLM_PROVIDER=ollama
OLLAMA_MODEL=qwen2.5-7b-instruct:latest

# Hugging Face API generation uses the same model family
HUGGINGFACE_MODEL=Qwen/Qwen2.5-7B-Instruct
# HUGGINGFACE_API_KEY=hf_...

# DeepEval judge
DEEPEVAL_JUDGE_MODEL=gpt-4o-mini
# OPENAI_API_KEY=sk-...

# Local embedding/reranker hardware
DEVICE=auto  # auto | cuda | cpu
```

### Vector store (NumPy or Qdrant)

By default retrieval uses the `.npy` embeddings and an in-memory BM25 index from `RUN_DIRECTORY`. For a scalable setup, use Qdrant (dense + sparse hybrid search with server-side RRF, indexed metadata filters, incremental upserts, and the filing catalog stored with the chunks):

```bash
docker compose up -d qdrant
uv run python scripts/migrate_npy_to_qdrant.py   # loads an existing run, no re-embedding
# in .env
VECTOR_STORE=qdrant
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=filings_bge_base_v1
```

New filings ingested with `VECTOR_STORE=qdrant` are indexed automatically. See `DOCKER.md` for the Docker setup (including a data-free backend image with `INCLUDE_DATA=false`) and `study/12_qdrant_vector_store.md` for the rationale and tradeoffs.

See `.env.example` for all options and `study/11_model_execution_modes.md` for backend selection, component overrides, reranker endpoint formats, and embedding-index compatibility.

## API Endpoints

### Health Check
```bash
GET /health
```

### List Filings
```bash
GET /filings
```

### Ask a Question
```bash
POST /questions
Content-Type: application/json

{
  "question": "What was Apple's total net sales in fiscal 2025?",
  "doc_id": "AAPL_10-K_2025"
}
```

**Response:**
```json
{
  "question": "What was Apple's total net sales in fiscal 2025?",
  "doc_id": "AAPL_10-K_2025",
  "answer": {
    "answer": "$391,035 million",
    "explanation": "The consolidated statements show total net sales for fiscal 2025.",
    "citations": ["S1"],
    "insufficient_context": false
  },
  "sources": [
    {
      "source_id": "S1",
      "doc_id": "AAPL_10-K_2025",
      "section": "Item 7",
      "text": "Total net sales decreased 4%...",
      "score": 0.89
    }
  ],
  "model": "qwen2.5-7b-instruct:latest",
  "latency_seconds": 2.34
}
```

## Project Structure

```
techfiling-copilot/
├── api/                    # FastAPI backend
│   ├── main.py            # Application entry point
│   ├── config.py          # API configuration
│   ├── dependencies.py    # Dependency injection
│   ├── routers/           # API endpoints
│   ├── schemas/           # Pydantic models
│   └── services/          # Business logic
│
├── src/                    # RAG pipeline
│   ├── config/            # Centralized settings
│   ├── generation/        # LLM generation & prompts
│   ├── ingestion/         # Document parsing & chunking
│   ├── retriever/         # Hybrid retrieval & reranking (numpy and Qdrant)
│   └── vector_store/      # Qdrant chunk store & sparse (BM25) vectors
│
├── web/                    # Next.js frontend
│   ├── app/               # Pages and layouts
│   ├── components/        # React components
│   ├── lib/               # API client
│   └── types/             # TypeScript types
│
├── scripts/                # Utilities (e.g. migrate_npy_to_qdrant.py)
├── data/                   # Parsed documents & embeddings
├── eval/                   # Evaluation scripts & results
├── goldens/               # Golden test datasets
└── tests/                 # Test suite
```

## Evaluation Results

The RAG system was systematically evaluated across all pipeline stages using both traditional metrics (human labels) and LLM-judge metrics (DeepEval with gpt-4o).

### Overall Performance (Single-Company Use Case)

| Metric | Score |
|---|---:|
| **Triad Mean** | **80.4%** |
| Contextual Relevancy | 49.2% |
| Faithfulness | 95.6% |
| Answer Relevancy | 96.5% |

### Baseline vs Final Comparison

#### Retriever (Traditional Metrics, Human Labels)

| Metric | Baseline (Dense only) | Final (Hybrid + Reranker) | Improvement |
|---|---:|---:|---:|
| Hit@5 | 94.6% | **98.2%** | +3.6% |
| MRR@5 | 88.7% | **93.5%** | +4.8% |
| Recall@5 | 74.3% | 76.1% | +1.8% |
| nDCG@5 | 75.8% | **77.5%** | +1.7% |

**Key changes**: Added BM25 with RRF, Ettin reranker, tuned hybrid alpha to 0.85, implemented pseudo-heading chunking

#### Retriever (DeepEval Metrics, LLM Judge)

| Metric | Score | Pass Rate |
|---|---:|---:|
| Contextual Precision | 81.0% | 85% |
| Contextual Recall | 65.8% | 70% |
| Mean Score | 73.4% | - |

#### Generation (Isolation, Curated Context)

| Metric | V1 (Baseline) | V2 (Final) | Improvement |
|---|---:|---:|---:|
| Faithfulness | 100% | 100% | Maintained |
| Answer Relevancy | 98.8% | 96.5% | -2.3% (false negative) |
| Answer Correctness | 88.3% | **91.5%** | **+3.2%** |
| Correctness Pass Rate | 90% | **100%** | **+10%** |

**Key changes**: V2 prompt explicitly requires answering every part of multi-part questions, preserves exact values/units, includes causes/reasons

#### Full-Pipeline (End-to-End)

| Metric | Baseline (V1 + old chunking) | Final (V2 + pseudo-heading) | Improvement |
|---|---:|---:|---:|
| Triad Mean | 71.6% | **76.4%** | **+4.8%** |
| Contextual Relevancy | 41.2% | 44.7% | +3.5% |
| Faithfulness | 93.1% | 89.2% | -3.9% |
| Answer Relevancy | 80.6% | **95.2%** | **+14.6%** |

**Key changes**: Pseudo-heading chunking for better table extraction, V2 prompt for complete answers

### Production Configuration

| Setting | Value |
|---|---|
| **Prompt version** | `v2_complete_precise` |
| **Chunk directory** | `data/chunks/2026-10-20_16-33-24` (pseudo-heading chunking) |
| **Context-k** | 5 |
| **Embedding mode** | local |
| **Reranker mode** | local |
| **Hybrid alpha** | 0.85 |
| **Retrieve-k** | 50 |
| **Reranker model** | `cross-encoder/ettin-reranker-150m-v1` |
| **Generator** | Qwen/Qwen2.5-7B-Instruct (Hugging Face via Featherless AI) |
| **DeepEval judge** | gpt-4o |

### How to Evaluate

#### Retriever Evaluation

```bash
# Traditional metrics (human labels)
uv run python eval/hybrid_rerank_eval.py --mode rerank --golden-path goldens/golden_retriever_dataset_v2.jsonl --run-directory data/chunks/2026-10-20_16-33-24

# DeepEval metrics (LLM judge)
uv run python eval/deepEval/retriever_eval.py --judge-model gpt-4o-mini
uv run python eval/deepEval/retriever_eval.py --compare
```

#### Generation Evaluation (Isolation)

```bash
# V1 baseline
uv run python eval/deepEval/generator_eval.py --prompt-version v1_strict_grounded --judge-model gpt-4o

# V2 production
uv run python eval/deepEval/generator_eval.py --prompt-version v2_complete_precise --judge-model gpt-4o

# Compare runs
uv run python eval/deepEval/generator_eval.py --compare
```

#### Full-Pipeline Evaluation

```bash
# Dry run (check configuration)
uv run python eval/deepEval/full_pipeline_eval.py --dry-run

# Run evaluation
uv run python eval/deepEval/full_pipeline_eval.py --judge-model gpt-4o --verbose

# Compare runs
uv run python eval/deepEval/full_pipeline_eval.py --compare
```

#### Run All Evaluations

```bash
# Run all three evaluations
uv run python eval/deepEval/run_all_evaluations.py --judge-model gpt-4o-mini

# Compare latest runs
uv run python eval/deepEval/run_all_evaluations.py --compare
```

### Detailed Evaluation Reports

For complete experiment documentation, case studies, and analysis:

- **Final Report**: [eval/V2_dataset_experiements/final_report.md](eval/V2_dataset_experiements/final_report.md) - Comprehensive summary of all evaluations
- **Retriever Experiments**: [eval/V2_dataset_experiements/retrivers.md](eval/V2_dataset_experiements/retrivers.md) - Retrieval tuning experiments
- **Generation Experiments**: [eval/V2_dataset_experiements/generation_isolation_result.md](eval/V2_dataset_experiements/generation_isolation_result.md) - Prompt optimization experiments
- **Full-Pipeline Summary**: [eval/V2_dataset_experiements/full_pipeline_evaluation_summary.md](eval/V2_dataset_experiements/full_pipeline_evaluation_summary.md) - End-to-end evaluation details

## Development

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=api --cov=src
```

### Code Quality

```bash
# Format code
uv run ruff format .

# Lint
uv run ruff check .

# Type check
uv run mypy src api
```

## Roadmap

- [ ] Multi-turn conversations with context
- [ ] User authentication (Supabase)
- [ ] Analytics dashboard
- [ ] Historical filing comparison
- [ ] Additional companies and filing types


## License

This project is for portfolio/demonstration purposes.

## Acknowledgments

- SEC EDGAR for public filing data
- Sentence Transformers for embedding models
- Cross-encoder models for reranking
