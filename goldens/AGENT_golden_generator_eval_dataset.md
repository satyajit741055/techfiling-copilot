# AGENT RUNBOOK — Generator Golden Evaluation Dataset

**Goal of this document:** let a brand-new person go from a fresh checkout to the
**one final dataset file** used to evaluate the answer generator in isolation,
and (optionally) run the generator evaluation against it.

> **FINAL DELIVERABLE**
> `goldens/golden_generation_dataset_claude.jsonl`
>
> This is the reviewed, evidence-backed dataset. Everything else in this runbook
> exists to produce, validate, or consume it.

> **FOR AN AI AGENT HANDED THIS FILE:** this document is self-contained. All inputs,
> paths, constants, commands, and the review rules you need are listed below. Work
> from the repository root. Execute Stages 1 → 2 → 3 in order to produce and
> validate the final dataset. For Stage 2 (the semantic review) the authoritative
> instructions are in `goldens/generator_claude.md`; read and follow it exactly.
> Do not invent data, never alter chunk text, and keep `review_status: claude_reviewed`.

---

## 0. Inputs & requirements (read before running)

**Fixed constants** (must match everywhere):

| Constant | Value |
|---|---|
| Chunk run id | `2026-10-16_17-03-28` |
| Source questions | 60 |
| Cases produced | 192 (60 × 3 variants + 12 traps) |

**Required input files** (must already exist in the repo; do NOT regenerate unless Stage 0 is run):

| Path | Role |
|---|---|
| `data/chunks/2026-10-16_17-03-28/AAPL_10-K_2025.chunks.jsonl` | Apple frozen evidence/distractor chunks |
| `data/chunks/2026-10-16_17-03-28/MSFT_10-K_2026.chunks.jsonl` | Microsoft frozen chunks |
| `data/chunks/2026-10-16_17-03-28/TSLA_10-K_2025.chunks.jsonl` | Tesla frozen chunks |
| `data/chunks/2026-10-16_17-03-28/run_manifest.json` | chunk-run metadata (config, counts) |
| `goldens/golden_retriver_dataset.jsonl` | 60 source questions: `query`, `doc_id`, `category`, `difficulty`, `expected_answer`, relevant `element_orders` |
| `goldens/generator_golden.md` | dataset design, variants, and schema (authoritative plan) |
| `goldens/generator_claude.md` | **Stage 2 semantic-review instructions** (authoritative) |
| `data/parsed/AAPL_10-K_2025.elements.jsonl` (+ MSFT, TSLA) | parsed filing elements, for verifying ambiguous evidence |
| `scripts/eval_scripts/build_generation_goldens.py` | Stage 1 builder |
| `scripts/eval_scripts/validate_generation_goldens_claude.py` | Stage 3 validator |

**Tooling:** Python ≥ 3.13 and [uv](https://docs.astral.sh/uv/) (or a venv + `pip install -r requirements.txt`).
Stages 1 and 3 use the **standard library only**; third-party packages (`requirements.txt`)
are needed only for Stage 0 (re-chunking) and Stage 4 (eval).

**Files produced by this runbook:**

| Path | Stage | Role |
|---|---|---|
| `goldens/golden_generation_dataset.jsonl` | 1 | mechanical draft (needs_review) |
| `goldens/generation_dataset_review.jsonl` | 1 | auto-flagged items |
| `goldens/golden_generation_dataset_claude.jsonl` | 2 | **FINAL dataset** |
| `goldens/generation_claude_review.jsonl` | 2 | per-question review log |

> If any **required input file** is missing, STOP — do not fabricate it. Either run
> Stage 0 to (re)generate the chunk run, or restore the file from version control.

---

## 1. What this dataset is

It evaluates the **answer generator alone** (no retrieval). Each case freezes a
question together with the exact chunks the generator should see, so a failure is
attributable to generation, not retrieval. Three context variants per question
plus adversarial traps:

| Variant | Context | Tests |
|---|---|---|
| `oracle` | minimum gold chunk(s) only | extraction/reasoning upper bound |
| `oracle_plus_distractors` | gold + plausible distractors (~5 total) | evidence selection (primary score) |
| `no_gold` | distractors only, answer removed | safe abstention / hallucination |
| `trap` | hand-written unanswerable question | wrong period/company/metric, bad premise |

Current size: **60 source questions × 3 variants + 12 traps = 192 cases.**

---

## 2. Repository map (only what matters here)

```
data/
  parsed/                                  # stable parsed 10-K elements (input)
  chunks/2026-10-16_17-03-28/              # FROZEN chunk run used by the dataset
goldens/
  golden_retriver_dataset.jsonl            # 60 human-labeled source questions (input)
  generator_golden.md                      # dataset design + build plan
  generator_claude.md                      # semantic-review prompt (Claude step)
  golden_generation_dataset.jsonl          # STAGE 1 output: auto-built draft
  generation_dataset_review.jsonl          # STAGE 1 auto-flagged items
  golden_generation_dataset_claude.jsonl   # >>> FINAL reviewed dataset <<<
  generation_claude_review.jsonl           # per-question review log
scripts/eval_scripts/
  build_generation_goldens.py              # STAGE 1 builder
  validate_generation_goldens.py           # structural validator (draft)
  validate_generation_goldens_claude.py    # STAGE 3 validator (final dataset)
eval/
  generation_eval.py                       # STAGE 4 consumer (runs the generator)
src/generation/                            # the generator being evaluated
requirements.txt                           # pip-friendly dependency list
```

---

## 3. Prerequisites

- **Python ≥ 3.13**
- **[uv](https://docs.astral.sh/uv/)** (recommended) — manages the env from `pyproject.toml` / `uv.lock`.
- For **Stage 4 (eval only)**: an LLM backend
  - `ollama` provider → a running local [Ollama](https://ollama.com) server with the model pulled (`ollama pull qwen2.5-7b-instruct:latest`), **or**
  - `openai` / `anthropic` / `google` → the matching API key env var (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`).

### Install

```powershell
# Preferred (creates .venv from pyproject + uv.lock)
uv sync

# Fallback with pip
python -m venv .venv
.venv\Scripts\activate            # Windows PowerShell
pip install -r requirements.txt
```

> Building and validating the dataset (Stages 1–3) use only the Python standard
> library. Third-party packages are needed only for re-chunking (Stage 0) and the
> eval (Stage 4). All commands below use `uv run`; drop that prefix if you
> activated a venv instead.

---

## 4. Pipeline overview

```
data/parsed  ──(Stage 0: chunk)──►  data/chunks/<run>/
                                         │
goldens/golden_retriver_dataset.jsonl    │
            └──────────(Stage 1: build)──┴──►  golden_generation_dataset.jsonl   (draft)
                                                      │
                        (Stage 2: semantic review ────┤  per goldens/generator_claude.md)
                                                      ▼
                                        golden_generation_dataset_claude.jsonl   (FINAL)
                                        generation_claude_review.jsonl
                                                      │
                           (Stage 3: validate ───────┤  validate_generation_goldens_claude.py)
                                                      ▼
                           (Stage 4: eval ───── eval/generation_eval.py)
```

---

## 5. Stage 0 — (OPTIONAL) regenerate the chunk run

**Skip this if `data/chunks/2026-10-16_17-03-28/` already exists** (it does in this
repo). Only do this if you re-parse or re-chunk the filings.

```powershell
# Windows PowerShell
$env:PYTHONPATH="src"; uv run python -m ingestion.processor --skip-parse
```
```bash
# macOS/Linux
PYTHONPATH=src uv run python -m ingestion.processor --skip-parse
```

This writes a **new** `data/chunks/<new_timestamp>/` directory.

> ⚠️ **CRITICAL — chunk-run alignment.** The run id is hard-coded in three places.
> If you create a new chunk run you MUST update all three to the new timestamp,
> or the dataset and chunks will not line up:
>
> 1. `scripts/eval_scripts/build_generation_goldens.py` → `CHUNK_RUN`
> 2. `scripts/eval_scripts/validate_generation_goldens_claude.py` → `EXPECTED_CHUNK_RUN`
> 3. `src/generation/config.py` → `GenerationConfig.run_directory`
>
> (Also update `scripts/eval_scripts/validate_generation_goldens.py` →
> `EXPECTED_CHUNK_RUN` if you validate the Stage 1 draft.)

---

## 6. Stage 1 — build the draft dataset

```powershell
uv run python scripts/eval_scripts/build_generation_goldens.py
```

Reads `goldens/golden_retriver_dataset.jsonl` + the frozen chunk run and writes:

- `goldens/golden_generation_dataset.jsonl` — 192 draft cases (all `needs_review`)
- `goldens/generation_dataset_review.jsonl` — auto-flagged ambiguous items

This draft is **mechanical** (element-order → chunk mapping). It is *not* ground
truth: it may leak answers into `no_gold`, mis-map evidence, or carry wrong
expected answers. Stage 2 fixes that.

---

## 7. Stage 2 — semantic review (produces the FINAL dataset)

This is the quality step. A strong model with filesystem access executes the
prompt in **`goldens/generator_claude.md`**, auditing every case against the
actual chunk text and correcting answers, values, evidence, and contexts.

It produces:

- `goldens/golden_generation_dataset_claude.jsonl` — **the final dataset**
- `goldens/generation_claude_review.jsonl` — one review record per source question

Each final case keeps the base schema and adds review metadata:
`labeling_method`, `review_model`, `confidence` (`high|medium|low`),
`requires_human_review`, `evidence_map` (verbatim quotes), and `corrections`.

Rules enforced by the prompt (and checked in Stage 3):
- Never rewrite/normalize chunk text; store full frozen chunk dicts + SHA-256.
- Evidence quotes must be **verbatim** substrings of their claimed chunk.
- `no_gold` must not contain the answer (numeric or narrative).
- `review_status` is `claude_reviewed` — **never** `human_approved`.
- Every `medium`/`low`-confidence case sets `requires_human_review: true`.

> These artifacts already exist in the repo (built and reconciled to chunk run
> `2026-10-16_17-03-28`, all cases `high` confidence, 0 pending human review). Re-run
> Stage 2 only if the chunk run, source questions, or review rules change.

---

## 8. Stage 3 — validate the final dataset

```powershell
uv run python scripts/eval_scripts/validate_generation_goldens_claude.py
# strict mode also fails on warnings:
uv run python scripts/eval_scripts/validate_generation_goldens_claude.py --strict
```

Runs 15 checks: parsing/unique IDs, exactly one of each variant per question, 12
traps present, chunk-id/SHA-256 integrity, gold vs `is_gold` agreement,
answerable/unanswerable invariants, every expected value present in an oracle
chunk, every `evidence_map` quote verbatim, **no** expected value leaking into
`no_gold`, doc-id membership, `review_status=claude_reviewed`, every correction
logged, and `medium/low` confidence flagged. **Expected result: `PASSED`.**

> A clean validator run means the dataset is *structurally and evidentially*
> sound. It does **not** mean a human has approved it — see §10.

---

## 9. Stage 4 — (OPTIONAL) run the generator evaluation

Point the eval at the **final** dataset (its default path is the draft, so pass
`--golden-path` explicitly):

```powershell
# Local Ollama (default provider/model)
uv run python eval/generation_eval.py `
  --golden-path goldens/golden_generation_dataset_claude.jsonl

# A hosted provider
uv run python eval/generation_eval.py `
  --golden-path goldens/golden_generation_dataset_claude.jsonl `
  --provider openai --model gpt-4o-mini

# Smoke test a few cases / one variant
uv run python eval/generation_eval.py `
  --golden-path goldens/golden_generation_dataset_claude.jsonl `
  --variants oracle --limit 5
```

Results land in `eval/results/<timestamp>_<provider>_<model>/`
(`cases.jsonl`, `manifest.json`, `summary.json`). Compare runs with:

```powershell
uv run python eval/generation_eval.py --compare
```

Key metrics: `answer_correct`, `numeric_faithfulness`, `abstention_correct`,
`hallucinated_answer`, `false_abstention`, `citation_precision`.

---

## 10. Final acceptance (human sign-off)

The validator passing is necessary but not sufficient. Before treating the
dataset as authoritative ground truth, a human should:

1. Read `goldens/generation_claude_review.jsonl` and spot-check `corrections`.
2. Review any record with `requires_human_review: true` (currently none).
3. Confirm scope on rewritten/narrowed answers.

Keep `review_status: claude_reviewed`. Only a human marks the dataset approved
(e.g. in your own records / PR description) — do not set `human_approved` in the
file.

---

## 11. One-shot checklist

```powershell
# 0. install
uv sync

# 1. (only if re-chunking) regenerate chunks and update the 3 CHUNK_RUN constants
#    $env:PYTHONPATH="src"; uv run python -m ingestion.processor --skip-parse

# 2. build the draft
uv run python scripts/eval_scripts/build_generation_goldens.py

# 3. semantic review per goldens/generator_claude.md
#    -> goldens/golden_generation_dataset_claude.jsonl (FINAL)
#    -> goldens/generation_claude_review.jsonl

# 4. validate the final dataset  (expect: PASSED)
uv run python scripts/eval_scripts/validate_generation_goldens_claude.py

# 5. (optional) evaluate the generator against the FINAL dataset
uv run python eval/generation_eval.py --golden-path goldens/golden_generation_dataset_claude.jsonl
```

---

## 12. Case schema (reference)

```json
{
  "id": "q004_oracle",
  "source_qid": "q004",
  "query": "How much did Apple spend on research and development in fiscal 2025?",
  "doc_id": "AAPL_10-K_2025",
  "category": "exact_numeric",
  "difficulty": "easy",
  "context_variant": "oracle",
  "chunk_run": "2026-10-16_17-03-28",
  "answerable": true,
  "expected_answer": "$34,550 million",
  "expected_values": [
    {"value": 34550, "unit": "million", "currency": "USD", "period": "FY2025", "metric": "research and development expense"}
  ],
  "required_facts": ["Apple's research and development expense was $34,550 million in fiscal 2025"],
  "gold_chunk_ids": ["AAPL_10-K_2025_c0166"],
  "context": [
    {"chunk_id": "AAPL_10-K_2025_c0166", "is_gold": true, "text_sha256": "<sha256 of chunk text>", "chunk": { "...full frozen chunk dict..." }}
  ],
  "unanswerable_reason": null,
  "labeling_method": "claude_semantic_review",
  "review_model": "claude-opus-4.8",
  "review_prompt_version": "v1",
  "confidence": "high",
  "requires_human_review": false,
  "evidence_map": [
    {"fact": "...", "chunk_ids": ["AAPL_10-K_2025_c0166"], "verbatim_evidence": ["| ... Research and development | 34,550 | 31,370 | 29,915 |"]}
  ],
  "corrections": [
    {"field": "expected_answer", "original": "$33,284 million", "corrected": "$34,550 million", "reason": "FY2025 income statement reports 34,550 (millions)."}
  ],
  "review_status": "claude_reviewed",
  "review_notes": []
}
```

For `no_gold`/`trap`: `answerable=false`, `gold_chunk_ids=[]`, `expected_answer=null`,
and a non-empty `unanswerable_reason`.

---

## 13. Troubleshooting

- **`wrong_chunk_run` errors from the validator** → the dataset's `chunk_run`
  doesn't match `EXPECTED_CHUNK_RUN`. Align per §5.
- **`ModuleNotFoundError: ingestion`** (Stage 0) → set `PYTHONPATH=src` as shown.
- **`ImportError` for `generation`** (Stage 4) → ensure `src/generation/__init__.py`
  exists (it must be exactly `__init__.py`). `eval/generation_eval.py` adds `src`
  to `sys.path` automatically.
- **Ollama connection refused** (Stage 4) → start the server and pull the model:
  `ollama pull qwen2.5-7b-instruct:latest`.
- **Missing API key** (Stage 4) → export `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` /
  `GOOGLE_API_KEY` for the chosen provider; install the matching SDK (see
  `requirements.txt`).
- **torch install issues** → use `uv sync` (handles the CUDA index) or install a
  specific wheel from https://pytorch.org/get-started/locally/ .
```
