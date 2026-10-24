# Agent notes

## Commands
- Tests: `uv run python -m pytest tests -q` (`tests/test_api.py` loads the real local pipeline, ~4 min; `tests/test_model_runtime.py` is fast and offline)
- Adding dependencies: `uv --system-certs add <pkg>` (the corporate network needs system TLS certificates)
- DeepEval dry runs (no model calls): `uv run python eval/deepEval/full_pipeline_eval.py --dry-run`, `uv run python eval/deepEval/generator_eval.py --dry-run`

## Configuration
- `.env` controls everything; `src/config/settings.py` is the schema/defaults.
- `MODEL_EXECUTION_MODE=local|huggingface_api` (plus per-stage overrides) selects where embedding, reranking and generation run; backend selection lives only in `src/model_runtime/factory.py`.
- Do not print or commit `.env` contents (contains API keys).
