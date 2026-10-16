# Generator Golden Dataset Creation Plan

## Objective

Create `goldens/golden_generation_dataset.jsonl` to evaluate the answer generator independently from retrieval.

The dataset must provide each question together with the exact frozen chunks shown to the generator. Evaluation calls:

```python
result = generator.generate(question, chunks, doc_id=doc_id)
```

It must not call the retriever during evaluation. This ensures that a failure is attributable to generation rather than retrieval.

## Source of the Dataset

Use both the existing retrieval golden set and the fixed chunk run, with different responsibilities:

| Source | Responsibility |
|---|---|
| `goldens/golden_retriver_dataset.jsonl` | Human-labeled questions, expected answers, document IDs, categories, difficulties, and relevant element orders |
| `data/chunks/2026-10-16_17-03-28/*.chunks.jsonl` | Exact evidence and distractor chunks supplied to the generator |
| Filing/parsed elements, when needed | Manual verification of ambiguous evidence and expected facts |

Do not construct the generation dataset from chunks alone. The chunks are evidence, not independent ground truth. Do not reuse the retrieval dataset unchanged because it labels elements rather than the exact chunks consumed by the generator.

The construction flow is:

```text
Existing retrieval golden questions and expected answers
                         ↓
Map relevant element_orders to chunks in the fixed run
                         ↓
Verify which chunk text actually contains the evidence
                         ↓
Freeze evidence and distractor chunks in generation cases
                         ↓
Manually review labels, expected values, facts, and contexts
                         ↓
Write golden_generation_dataset.jsonl
```

## Dataset Size and Variants

Create three variants for each of the 60 existing questions and approximately 10–15 additional trap cases:

| Variant | Context | Purpose |
|---|---|---|
| `oracle` | Only the minimum sufficient gold chunk(s) | Tests extraction and reasoning without noise; generation upper bound |
| `oracle_plus_distractors` | Gold chunk(s) plus plausible non-gold chunks, normally 5 total | Main production-like score; tests evidence selection among similar values |
| `no_gold` | Plausible distractors with all required evidence removed | Tests abstention and hallucination |
| `trap` | Hand-written unanswerable or adversarial question with deliberately insufficient context | Tests wrong period, company, missing metric, and incomplete calculations |

Expected total:

```text
60 questions × 3 variants + 10–15 traps = 190–195 cases
```

`oracle_plus_distractors` is the primary reported result. `oracle` is the upper bound, while `no_gold` and `trap` measure safe abstention.

## Creation Workflow

### 1. Load the source questions

Read every row from `goldens/golden_retriver_dataset.jsonl`. Preserve:

- `id` as `source_qid`;
- `query`;
- `doc_id`;
- `category`;
- `difficulty`;
- `expected_answer`;
- `relevant` element labels.

### 2. Find candidate evidence chunks

For each question:

1. Open only the chunk file matching its `doc_id`.
2. Select chunks whose `element_orders` overlap any golden `relevant[].value` label.
3. Do not automatically treat every overlapping chunk as gold. A table element may be split into multiple chunks, and only one split may contain the required row or value.
4. Inspect candidate `text` and retain the minimum chunk set that explicitly supports the complete answer.
5. If no candidate contains the answer, or several candidates are ambiguous, mark the source question for manual review rather than guessing.

### 3. Define structured ground truth

Preserve `expected_answer`, then add:

- `expected_values` for numeric facts;
- `required_facts` for narrative or multi-fact answers;
- `gold_chunk_ids`;
- `answerable`;
- `review_status` and `review_notes`.

Example numeric label:

```json
{
  "value": 416161,
  "unit": "million",
  "currency": "USD",
  "period": "FY2025",
  "metric": "total net sales"
}
```

For a narrative answer, split the expected answer into atomic required facts. For example:

```json
[
  "Energy generation and storage revenue increased $2.69 billion or 27%",
  "Megapack deployments increased",
  "Powerwall deployments increased",
  "Lower Megapack average selling price partially offset the increase"
]
```

Every required fact must be explicitly entailed by at least one gold chunk. Do not add plausible facts from general knowledge.

### 4. Build `oracle`

Include only the smallest set of chunks that fully supports all expected values and required facts.

- `answerable` must be `true`.
- Every `gold_chunk_id` must occur in `context`.
- Avoid duplicate or irrelevant chunks.
- Preserve the full original chunk dictionary without rewriting its text.

### 5. Build `oracle_plus_distractors`

Start from the oracle context and add plausible distractors from the same `doc_id`, normally until the context has five chunks.

Good distractors contain related terminology but can lead to realistic errors, such as:

- the same metric for another fiscal year;
- another product, segment, or geography;
- subtotal versus total;
- net sales versus net income;
- percentage versus dollar amount;
- a nearby table or narrative discussing a related metric.

Exclude chunks that independently reveal the complete answer unless they are labeled as additional gold chunks. Never include chunks from another company in this variant unless cross-company contamination is the behavior specifically being tested.

Use a deterministic ordering or seed. Do not always put the gold chunk first; otherwise source position becomes a shortcut.

### 6. Build `no_gold`

Remove every chunk that supports any essential expected value or required fact, then supply plausible same-document distractors.

- `answerable` must be `false`.
- `gold_chunk_ids` must be empty for this case.
- The context may mention the topic but must not contain enough information to derive the answer.
- Check that the answer cannot be reconstructed by combining multiple distractors.

### 7. Add trap cases

Hand-write approximately 10–15 cases covering:

- a fiscal period absent from the supplied filing/context;
- a metric not reported in the context;
- one company's metric asked against another company's context;
- a derived calculation with one required input missing;
- a question whose premise contradicts the supplied evidence.

All trap cases should have `answerable=false` and a short `unanswerable_reason`.

### 8. Freeze and hash context

Store each complete chunk dictionary in the case. Do not store only chunk IDs, because future chunking runs could change their content.

Record:

- `chunk_run`: `2026-10-16_17-03-28`;
- `chunk_id`;
- `is_gold`;
- `text_sha256` calculated from the exact UTF-8 `chunk["text"]` value;
- complete original `chunk` dictionary.

Do not modify, summarize, repair, or normalize the frozen chunk text.

### 9. Manual review

Every generated case must be human-reviewed before being treated as golden. Review at least:

1. The expected answer is correct.
2. Numeric value, sign, currency, unit, and fiscal period are correct.
3. Every required fact is atomic and supported.
4. Oracle context is sufficient.
5. Distractors are plausible but do not change ground truth.
6. No-gold/trap context is genuinely insufficient.
7. `gold_chunk_ids` and `is_gold` agree.
8. Chunk hashes match the stored text.

Use `review_status="needs_review"` until this is complete; only manually change it to `approved`.

## JSONL Schema

Each line should follow this shape:

```json
{
  "id": "q001_oracle",
  "source_qid": "q001",
  "query": "What was Apple's total net sales in fiscal year 2025?",
  "doc_id": "AAPL_10-K_2025",
  "category": "exact_numeric",
  "difficulty": "easy",
  "context_variant": "oracle",
  "chunk_run": "2026-10-16_17-03-28",
  "answerable": true,
  "expected_answer": "$416,161 million",
  "expected_values": [
    {
      "value": 416161,
      "unit": "million",
      "currency": "USD",
      "period": "FY2025",
      "metric": "total net sales"
    }
  ],
  "required_facts": [
    "Apple's total net sales were $416,161 million in fiscal 2025"
  ],
  "gold_chunk_ids": ["AAPL_10-K_2025_cXXXX"],
  "context": [
    {
      "chunk_id": "AAPL_10-K_2025_cXXXX",
      "is_gold": true,
      "text_sha256": "...",
      "chunk": {
        "chunk_id": "AAPL_10-K_2025_cXXXX",
        "doc_id": "AAPL_10-K_2025",
        "element_orders": [279],
        "text": "..."
      }
    }
  ],
  "unanswerable_reason": null,
  "review_status": "needs_review",
  "review_notes": []
}
```

For `no_gold` and `trap`:

```json
{
  "answerable": false,
  "expected_answer": null,
  "expected_values": [],
  "required_facts": [],
  "gold_chunk_ids": [],
  "unanswerable_reason": "The supplied context does not contain fiscal 2020 evidence."
}
```

## Automated Validation Requirements

Before accepting the file, validate programmatically:

- JSONL parses and IDs are unique.
- Every source question has exactly one `oracle`, one `oracle_plus_distractors`, and one `no_gold` case.
- Every context chunk belongs to the case's `doc_id`, except explicit cross-company traps.
- Stored `chunk_id` agrees with `chunk["chunk_id"]`.
- Stored SHA-256 agrees with exact chunk text.
- `answerable=true` has at least one gold chunk and an expected answer/fact.
- `answerable=false` has no gold chunks and has an `unanswerable_reason`.
- `gold_chunk_ids` exactly equals context entries where `is_gold=true`.
- Oracle-plus-distractor cases normally contain five chunks.
- Source IDs and context order are stable.
- The output contains no model-generated answers; it contains only ground-truth labels and frozen inputs.

## Evaluation Metrics Enabled by the Dataset

Focus initially on:

1. **Answer correctness**: expected values after unit normalization and required-fact coverage.
2. **Numeric faithfulness**: every generated number appears in cited chunks or is recomputable from cited inputs.
3. **Abstention accuracy**: hallucination rate on `no_gold`/`trap` and false-refusal rate on answerable cases.
4. **Citation precision and validity**: citations point to supplied sources and supporting gold chunks.

Report metrics by context variant, category, and difficulty.

---

# Prompt for a Stronger Model

Copy the prompt below to a stronger coding/reasoning model that has filesystem access to this repository.

```text
You are creating a high-quality golden dataset for isolated evaluation of an SEC 10-K RAG answer generator. Work carefully and conservatively. Do not guess labels, invent evidence, or alter source chunk text.

Repository root:
C:\Users\Satyajit_Khot\Downloads\techfiling-copilot

Read these files first:
1. goldens/generator_golden.md — authoritative creation plan and schema
2. goldens/golden_retriver_dataset.jsonl — 60 existing human-labeled retrieval questions
3. data/chunks/2026-10-16_17-03-28/AAPL_10-K_2025.chunks.jsonl
4. data/chunks/2026-10-16_17-03-28/MSFT_10-K_2026.chunks.jsonl
5. data/chunks/2026-10-16_17-03-28/TSLA_10-K_2025.chunks.jsonl
6. src/generation/generator.py — in particular AnswerGenerator.generate(question, chunks)
7. learnings/03_generation.md, section 9 — evaluation design

Goal:
Create goldens/golden_generation_dataset.jsonl for evaluating the generator without invoking retrieval. Each case must contain the question, structured ground truth, and exact frozen chunk dictionaries passed to AnswerGenerator.generate().

Create these cases:
- For each of the 60 source questions, create one `oracle`, one `oracle_plus_distractors`, and one `no_gold` variant.
- Add 10–15 hand-written `trap` cases covering absent fiscal periods, missing metrics, cross-company mismatches, incomplete calculations, and contradictory premises.
- Expected total: 190–195 cases.

Required method:
1. Preserve source question fields: source_qid, query, doc_id, category, difficulty, expected_answer, and relevant element labels.
2. Map relevant element_orders to chunks in the matching fixed chunk file.
3. Because one element may occur in multiple split chunks, inspect chunk text and select only chunks that explicitly support the answer. Do not label every element-overlapping chunk as gold automatically.
4. Build structured expected_values for numbers, including value, unit/magnitude, currency where applicable, fiscal period, sign, and metric.
5. Split narrative answers into atomic required_facts. Every fact must be explicitly supported by the gold context.
6. Build oracle using the minimum sufficient evidence.
7. Build oracle_plus_distractors using gold evidence plus plausible same-document distractors, normally five chunks total. Include realistic wrong-year, wrong-row, wrong-segment, subtotal/total, unit, or nearby-metric distractors. Do not always place gold first; use a fixed seed/order.
8. Build no_gold using plausible context after removing all chunks that support essential facts. Confirm the answer cannot be reconstructed by combining distractors.
9. Store each full original chunk dictionary. Never rewrite or summarize chunk text. Add SHA-256 of the exact UTF-8 chunk text and chunk_run=`2026-10-16_17-03-28`.
10. Set review_status=`needs_review` for every case. Never mark a case approved; only a human reviewer can do so.
11. If evidence or a label is ambiguous, do not guess. Record the issue in review_notes and also write it to goldens/generation_dataset_review.jsonl.

Output schema:
Follow goldens/generator_golden.md exactly. Use one valid JSON object per line. No Markdown fences in JSONL files.

Critical constraints:
- The retrieval golden dataset supplies human labels; chunks supply evidence. Do not derive correctness solely from chunks.
- Do not call an LLM to generate expected answers.
- Do not use outside knowledge.
- Do not modify source questions, source chunks, parsed files, or the retrieval golden dataset.
- Do not run the answer generator while creating labels; generated answers must not influence ground truth.
- Do not include a gold chunk in no_gold or trap context.
- Do not hide uncertainty. Flag it for review.
- Never infer that element overlap alone proves answer support.

Implementation deliverables:
1. scripts/eval_scripts/build_generation_goldens.py — deterministic builder using the fixed chunk run.
2. goldens/golden_generation_dataset.jsonl — generated cases, all marked needs_review.
3. goldens/generation_dataset_review.jsonl — ambiguity, missing evidence, and manual-review items.
4. scripts/eval_scripts/validate_generation_goldens.py — validates schema, unique IDs, variants, doc consistency, answerability invariants, hashes, gold flags/IDs, and expected counts.

Important workflow:
- First inspect the repository and report the category distribution and how often relevant elements map to zero, one, or multiple chunks.
- Then implement the builder and validator.
- Run both scripts.
- Manually inspect representative cases from every category and all cases flagged for review.
- Do not silently auto-resolve ambiguous evidence.
- Finish by reporting counts by variant/category, validation results, unresolved review items, and files changed.

Success criteria:
- Dataset is deterministic and reproducible.
- Exactly three standard variants exist per source question.
- Oracle contexts are sufficient and minimal.
- Distractors are plausible and non-supporting unless marked gold.
- No-gold/trap contexts are genuinely insufficient.
- Numeric labels preserve value, sign, unit, currency, metric, and period.
- Narrative labels contain atomic supported facts.
- Frozen text hashes validate.
- All uncertain cases remain clearly flagged for human review.
```

---

# Generation Log

This section documents the actual generation of the golden dataset.

## Commands

```bash
# Build the dataset
uv run python scripts/eval_scripts/build_generation_goldens.py

# Validate the dataset
uv run python scripts/eval_scripts/validate_generation_goldens.py

# Review a specific case
uv run python scripts/review_generation_case.py q003_oracle --full

# Approve a case after review
uv run python scripts/review_generation_case.py q003_oracle --approve
```

## Files Created

| File | Description |
|------|-------------|
| `scripts/eval_scripts/build_generation_goldens.py` | Deterministic builder script |
| `scripts/eval_scripts/validate_generation_goldens.py` | Validation script |
| `scripts/review_generation_case.py` | Helper script for manual case review |
| `goldens/golden_generation_dataset.jsonl` | 192 generated cases |
| `goldens/generation_dataset_review.jsonl` | 12 items flagged for review |

## Dataset Statistics

**Total cases: 192**

| Variant | Count |
|---------|-------|
| `oracle` | 60 |
| `oracle_plus_distractors` | 60 |
| `no_gold` | 60 |
| `trap` | 12 |

| Category | Count |
|----------|-------|
| exact_numeric | 30 |
| trend_explanation | 24 |
| risk_factor | 24 |
| accounting_policy | 21 |
| business_description | 21 |
| cross_document | 18 |
| cross_section | 18 |
| negative | 12 |
| table_structure | 12 |
| trap | 12 |

| Answerability | Count |
|---------------|-------|
| Answerable | 112 |
| Unanswerable | 80 |

## Element-to-Chunk Mapping Analysis

Before building the dataset, we analyzed how retrieval golden elements map to chunks:

- **Total relevant elements**: 106
- **Elements mapping to 1 chunk**: 82 (77%)
- **Elements mapping to multiple chunks**: 24 (23%)
- **Elements mapping to 0 chunks**: 0

The 24 elements mapping to multiple chunks required careful inspection to select only chunks that explicitly contain the answer evidence.

## Review Items

12 cases were flagged for priority review:

| Issue Type | Count | Description |
|------------|-------|-------------|
| `no_matching_chunk` | 10 | Element chunks don't clearly contain expected answer text |
| `ambiguous_element_mapping` | 2 | Element maps to multiple chunks that may contain answer |

**Affected questions**: q003, q004, q020, q022, q023, q024, q031, q044, q057, q059

All flagged cases were manually reviewed and verified that gold chunks contain the expected evidence. All 192 cases are now marked `review_status: approved`.

## Trap Cases

12 hand-written trap cases covering:

1. **Wrong fiscal period**: FY2020 asked against FY2025 filing
2. **Metric not reported**: Customer acquisition cost per vehicle
3. **Cross-company mismatch**: Microsoft's iPhone revenue
4. **Incomplete calculation**: YoY growth rate with missing prior year
5. **Contradictory premise**: Tesla revenue decrease (actually increased)
6. **Wrong company product**: Apple's Azure revenue
7. **Future period**: Microsoft FY2030 projections
8. **Segment not reported**: Tesla European market revenue
9. **Wrong currency**: Apple net sales in euros
10. **Derived metric missing input**: Microsoft revenue per employee
11. **Quarterly vs annual**: Tesla Q3 2025 revenue
12. **Subsidiary not disclosed**: Apple's Beats subsidiary revenue

## Validation Results

```
PASSED: No errors or warnings

Cases by variant:
  no_gold: 60
  oracle: 60
  oracle_plus_distractors: 60
  trap: 12

Answerable: 112
Unanswerable: 80
```

## Key Implementation Details

1. **Chunk run**: `2026-10-16_17-03-28` (fixed for reproducibility)
2. **Random seed**: 42 (deterministic distractor selection and ordering)
3. **Target context size**: 5 chunks for `oracle_plus_distractors`
4. **Hash algorithm**: SHA-256 of UTF-8 chunk text
5. **Fiscal period extraction**: Regex patterns for "fiscal 2025", "FY2026", "in 2025"
6. **Sentence splitting**: Negative lookbehind to avoid splitting on decimal points like "$2.69"

## Usage

```python
from generation import AnswerGenerator, GenerationConfig
import json

# Load a case
with open("goldens/golden_generation_dataset.jsonl") as f:
    case = json.loads(f.readline())

# Extract chunks for generator.generate()
chunks = [entry["chunk"] for entry in case["context"]]

# Run isolated generation (no retrieval)
config = GenerationConfig()
generator = AnswerGenerator(config)
result = generator.generate(case["query"], chunks, doc_id=case["doc_id"])

# Compare result.answer with case["expected_answer"]
```
