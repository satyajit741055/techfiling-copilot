# TechFiling Copilot

> SEC 10-K Document Q&A with RAG (Retrieval-Augmented Generation)

A production-ready research assistant that helps equity analysts extract insights from SEC 10-K filings. Ask questions in plain English and get grounded, citation-backed answers.

## Features

- **Grounded Answers**: Every answer is backed by explicit citations from the filing
- **Citation Traceability**: Click any citation to view the exact source passage
- **Hybrid Retrieval**: BM25 + Dense embeddings + Cross-encoder reranking for high-quality results
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
│  Parsed Chunks (.jsonl) │ Embeddings (.npy) │ BM25 Index        │
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

# Copy environment file and configure
cp .env.example .env
# Edit .env with your settings (see Configuration section)
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
# LLM Provider: ollama | openai | anthropic | google | huggingface
LLM_PROVIDER=ollama

# Model (provider-specific)
OLLAMA_MODEL=qwen2.5-7b-instruct:latest
OPENAI_MODEL=gpt-4o-mini
ANTHROPIC_MODEL=claude-3-5-sonnet-20241022

# API Keys (only needed for cloud providers)
# OPENAI_API_KEY=sk-...
# ANTHROPIC_API_KEY=sk-ant-...

# Hardware
DEVICE=auto  # auto | cuda | cpu
```

See `.env.example` for all available options.

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
│   └── retriever/         # Hybrid retrieval & reranking
│
├── web/                    # Next.js frontend
│   ├── app/               # Pages and layouts
│   ├── components/        # React components
│   ├── lib/               # API client
│   └── types/             # TypeScript types
│
├── data/                   # Parsed documents & embeddings
├── eval/                   # Evaluation scripts & results
├── goldens/               # Golden test datasets
└── tests/                 # Test suite
```

## Evaluation Results

The retrieval pipeline was tuned through controlled experiments:

| Configuration | Hit@5 | MRR@5 | Recall@5 | nDCG@5 |
|--------------|-------|-------|----------|--------|
| Dense baseline | 0.750 | 0.551 | 0.601 | 0.511 |
| **Final (Hybrid + Ettin)** | **0.786** | **0.587** | **0.640** | **0.546** |

See [eval/final_retriever_conf.md](eval/final_retriever_conf.md) for detailed experiment documentation.

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
