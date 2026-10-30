# Generation Experiments — Lab Notebook

A running log of isolated answer-generation experiments. **Newest entry on top.** Raw per-case
outputs and aggregate summaries live in `eval/generation_results/<run_id>/`:

- `cases.jsonl` — generated answer, sources, prompt, validation, and metrics per case;
- `manifest.json` — dataset, filters, model config, command, and system-prompt hash;
- `summary.json` — overall and category/difficulty/variant aggregates.

This file records the *hypothesis, controlled change, results, observation, and decision* so the
reasoning is not lost.

## How to use

1. Before a run, record the **hypothesis** and the **single variable changed**.
2. Keep the dataset, selected cases, context variant, prompt hash, temperature, maximum tokens,
   and fact threshold fixed when comparing models.
3. Run the experiments sequentially when using local Ollama models; concurrent runs compete for
   GPU/CPU/RAM and invalidate latency comparisons.
4. Copy headline metrics below, inspect per-case failures, then record the decision and next step.

Standard commands:

```powershell
# Validate the final reviewed golden set first
uv run python scripts/eval_scripts/validate_generation_goldens_claude.py

# Run one model on the primary production-like variant
uv run python eval/generation_eval.py `
  --golden-path goldens/golden_generation_dataset_claude.jsonl `
  --provider ollama `
  --model <model> `
  --variants oracle_plus_distractors

# Compare saved runs
uv run python eval/generation_eval.py --compare
```

Fixed context unless an experiment explicitly says otherwise:

- Golden set: `goldens/golden_generation_dataset_claude.jsonl` (192 cases).
- Primary variant: `oracle_plus_distractors` (60 cases).
- Chunk run: `data/chunks/2026-10-16_17-03-28`.
- Prompt version: `v1_strict_grounded`.
- Temperature: `0.0`.
- Maximum output tokens: `1024`.
- Required-fact threshold: `0.6`.

---

<!-- ===== TEMPLATE: copy this block for each new experiment =====

## GEN-EXP-NNN — <short title>
- **Date:** YYYY-MM-DD
- **Result directories:**
  - `eval/generation_results/<run_id>`
- **Hypothesis:** <what you expect and why>
- **Change vs baseline:** <the one controlled variable changed>
- **Fixed filters:** <dataset, variants, cases, prompt hash, threshold>

| Model/config | Cases | Correct | Value cov. | Fact cov. | Faithful | Abstain | Halluc. | Cite prec. | Cite recall | Latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| | | | | | | | | | | |

- **Observation:** <did the hypothesis hold? quality/safety/latency trade-offs?>
- **Decision:** <which configuration is selected and why?>
- **Failure analysis:** <which cases/categories failed and likely cause?>
- **Next step:** <one controlled follow-up experiment>

================================================================= -->

## GEN-EXP-003 — No-gold and trap safety: abstention under insufficient context

- **Date:** 2026-10-16
- **Result directories:**
  - `eval/generation_results/2026-10-16_22-35-59_ollama_qwen2.5-7b-instruct_latest`
  - `eval/generation_results/2026-10-16_22-39-09_ollama_llama3.2_latest`
- **Hypothesis:** Qwen's stronger instruction-following will make it more likely than Llama to set
  `insufficient_context=true` when plausible-looking chunks do not support an answer.
- **Change vs GEN-EXP-001:** Context slice only: 60 `no_gold` + 12 `trap` cases (all 72 are
  unanswerable) instead of 60 answerable `oracle_plus_distractors` cases.
- **Fixed filters:** Same final golden set, prompt/config, temperature `0`, max tokens `1024`,
  fact threshold `0.6`, and local Ollama provider.

| Model | Cases | Correct / abstention | Hallucinated answer | Numeric faithful | Citation validity | Avg. latency |
|---|---:|---:|---:|---:|---:|---:|
| **Qwen 2.5 7B Instruct** | 72 | **77.8%** | **22.2%** (16/72) | **100.0%** | 100.0% | 2.57s |
| Llama 3.2 | 72 | 51.4% | 48.6% (35/72) | 60.4% | 100.0% | **1.62s** |

- **Observation:** The hypothesis held, but neither model is safe enough. Qwen refused correctly on
  56/72 cases and hallucinated on 16; Llama refused on only 37/72 and hallucinated on 35. Qwen
  reduces the hallucination rate by `26.4` percentage points, but a `22.2%` unsupported-answer
  rate is still a production blocker. Llama answered nearly half of all deliberately
  unanswerable cases and is not acceptable for grounded financial QA without an additional
  sufficiency gate.
- **Metric caveats:** `citation_precision=0` is expected/not useful here because unanswerable cases
  intentionally have `gold_chunk_ids=[]`; any citation cannot be a gold citation. Qwen's 100%
  numeric faithfulness does **not** make its 16 hallucinated answers correct — it means numbers it
  stated happened to occur in cited distractor chunks. Grounded-to-a-distractor is still a wrong
  answer. Citation validity at 100% only means cited source IDs existed.
- **Decision:** Retain Qwen as the safer model, but do **not** ship the current generation prompt
  without improving context-sufficiency detection. Llama 3.2 is rejected for this use case despite
  its lower latency.
- **Failure analysis needed:** Split the 16 Qwen and 35 Llama failures by `no_gold` versus `trap`,
  category, and failure pattern: wrong-year value, nearby metric, false-premise acceptance,
  partial-calculation completion, or unsupported synthesis. Inspect `cases.jsonl` answers and
  citations; aggregate metrics alone cannot identify the prompt fix.
- **Next step:** Add an explicit evidence-sufficiency checklist before answering (requested metric,
  company, fiscal period, unit, and all calculation inputs must be present), then rerun the same 72
  cases. Optimize hallucination rate first, while checking that oracle false abstention does not
  rise.

---

## GEN-EXP-002 — Oracle upper bound and distractor penalty

- **Date:** 2026-10-16
- **Result directories:**
  - `eval/generation_results/2026-10-16_22-30-59_ollama_qwen2.5-7b-instruct_latest`
  - `eval/generation_results/2026-10-16_22-34-23_ollama_llama3.2_latest`
- **Hypothesis:** Both models should perform better with minimum sufficient oracle evidence than
  with oracle plus distractors; Qwen should retain its quality advantage over Llama.
- **Change vs GEN-EXP-001:** Context variant only: `oracle` instead of
  `oracle_plus_distractors`. Models and all generation/scoring settings are unchanged.

| Model | Cases | Correct | Value coverage | Fact coverage | Numeric faithful | Abstention | Citation precision | Citation recall | Avg. latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Qwen 2.5 7B Instruct** | 60 | **61.7%** | **67.2%** | **73.5%** | **95.3%** | **100.0%** | 100.0% | **95.3%** | 3.34s |
| Llama 3.2 | 60 | 43.3% | 56.0% | 61.6% | 91.8% | 98.3% | 100.0% | 93.4% | **1.51s** |

Oracle-to-distractor comparison:

| Model | Oracle correct | + distractors correct | Delta | Oracle fact cov. | + distractors fact cov. | Delta |
|---|---:|---:|---:|---:|---:|---:|
| Qwen 2.5 7B | 61.7% | 53.3% | **-8.4 pp** | 73.5% | 67.9% | **-5.6 pp** |
| Llama 3.2 | 43.3% | 46.7% | +3.4 pp | 61.6% | 58.8% | -2.8 pp |

- **Observation:** Qwen's oracle score establishes the stronger generation upper bound and exceeds
  Llama by `18.4` percentage points on answer correctness. Adding distractors costs Qwen `8.4`
  points, confirming a real evidence-selection weakness on top of its base generation weakness.
  Qwen still fails 38.3% even with minimum sufficient evidence, so most remaining headroom cannot
  be fixed by retrieval or distractor handling alone.
- **Unexpected Llama result:** Llama's binary correctness is 3.4 points higher with distractors,
  even though its fact coverage, citation precision/recall, and faithfulness do not improve. This
  should not be interpreted as distractors helping. With only 60 cases, thresholded
  `answer_correct` can flip when extra context changes wording; inspect the exact cases and the
  lexical threshold before drawing a causal conclusion.
- **Decision:** Qwen remains the selected model. Prioritize prompt/output improvements for
  complete numeric and multi-fact answers first, then reduce its distractor penalty. Llama's speed
  (1.51s vs 3.34s) does not compensate for the 18.4-point oracle correctness gap.
- **Next step:** Analyze Qwen's oracle failures by scoring mode (`expected_values` versus
  `required_facts`). Check whether failures are true model omissions or false negatives from the
  lexical fact-coverage threshold. Any prompt change must be tested on oracle, distractor, and
  no-gold/trap slices together to avoid trading correctness for hallucination.

---

## GEN-EXP-001 — Qwen 2.5 7B vs Llama 3.2 on oracle plus distractors

- **Date:** 2026-10-16
- **Result directories:**
  - `eval/generation_results/2026-10-16_22-23-07_ollama_qwen2.5-7b-instruct_latest`
  - `eval/generation_results/2026-10-16_22-26-59_ollama_llama3.2_latest`
- **Hypothesis:** The larger instruction-tuned Qwen 2.5 7B model will select relevant evidence,
  reproduce expected values, and cite gold chunks more reliably than the smaller Llama 3.2 model,
  while Llama may have lower latency.
- **Change vs baseline:** Model only: `qwen2.5-7b-instruct:latest` versus `llama3.2:latest`.
- **Fixed filters:** Same final Claude-reviewed golden file, all 60
  `oracle_plus_distractors` cases, same prompt hash/configuration, temperature `0`, max tokens
  `1024`, fact threshold `0.6`, local Ollama provider.

| Model | Cases | Correct | Value coverage | Fact coverage | Numeric faithful | Abstention correct | Hallucinated | Citation precision | Citation recall | Avg. latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Qwen 2.5 7B Instruct** | 60 | **53.3%** | **62.1%** | **67.9%** | **94.0%** | **96.7%** | **0.0%** | **96.2%** | **88.1%** | 3.63s |
| Llama 3.2 | 60 | 46.7% | 47.1% | 58.8% | 90.9% | 95.0% | **0.0%** | 80.1% | 81.2% | **2.21s** |

Token and runtime totals:

| Model | Prompt tokens | Completion tokens | Total latency |
|---|---:|---:|---:|
| Qwen 2.5 7B Instruct | 110,969 | 7,281 | 217.93s |
| Llama 3.2 | 100,783 | 6,481 | 132.75s |

- **Observation:** The hypothesis held. Qwen was stronger on every reported quality metric:
  answer correctness `+6.6` percentage points, expected-value coverage `+14.9`, required-fact
  coverage `+9.1`, numeric faithfulness `+3.1`, citation precision `+16.1`, and citation recall
  `+6.9`. Both models produced zero hallucinated answers on this slice. Llama was substantially
  faster: average latency was `2.21s` versus Qwen's `3.63s` (approximately 39% lower), and it
  emitted fewer completion tokens.
- **Decision:** Select **Qwen 2.5 7B Instruct** for the quality-first generation configuration.
  Select Llama 3.2 only where latency/resource use outweigh the measured losses in answer and
  citation quality.
- **Important interpretation:** `oracle_plus_distractors` is primarily an answerable evidence-
  selection test. The 96–97% abstention score comes partly from the negative source questions;
  it is not a substitute for the dedicated `no_gold + trap` safety run. The 53.3% / 46.7%
  correctness scores show that both models still struggle with multi-fact narrative coverage and
  choosing the exact expected value among plausible same-document distractors, despite high
  citation validity and numeric faithfulness.
- **Comparison warning:** `generation_eval.py --compare` printed "runs are not directly
  comparable" because the result directory also contained the earlier five-case `oracle` smoke
  test (`2026-10-16_22-18-27_...`). That smoke test has different filters. The two 60-case runs
  above **are directly comparable to each other**: their dataset, variant, case count, prompt hash,
  and metric threshold are identical. The current comparator applies one global compatibility
  check across every saved run, so one unrelated run triggers the warning for the whole table.
- **Failure analysis needed:** Inspect the failed records in each run's `cases.jsonl`, grouped by
  `category` and `difficulty`, before changing the prompt. Separate true generation failures from
  possible evaluator false negatives caused by the lexical fact threshold.
- **Next steps:**
  1. Run both models on all 60 `oracle` cases. The gap from oracle to oracle+distractors isolates
     evidence-selection failures from pure answer-generation failures.
  2. Run both models on `no_gold trap` to measure refusal safety and hallucination directly.
  3. Analyze Qwen failures by category/value/fact/citation and change only one prompt behavior at
     a time.

Commands used:

```powershell
uv run python eval/generation_eval.py `
  --golden-path goldens/golden_generation_dataset_claude.jsonl `
  --provider ollama `
  --model qwen2.5-7b-instruct:latest `
  --variants oracle_plus_distractors

uv run python eval/generation_eval.py `
  --golden-path goldens/golden_generation_dataset_claude.jsonl `
  --provider ollama `
  --model llama3.2:latest `
  --variants oracle_plus_distractors

uv run python eval/generation_eval.py --compare
```

---

## GEN-SMOKE-001 — Five-case oracle pipeline check

- **Date:** 2026-10-16
- **Result directory:**
  `eval/generation_results/2026-10-16_22-18-27_ollama_qwen2.5-7b-instruct_latest`
- **Purpose:** Confirm Ollama connectivity, output parsing/validation, frozen-context formatting,
  metrics, and result persistence before the full benchmark.
- **Configuration:** Qwen 2.5 7B, first five `oracle` cases, verbose logging.

| Cases | Correct | Value coverage | Fact coverage | Faithful | Abstention | Hallucinated | Cite precision | Cite recall | Avg. latency |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 100.0% | 100.0% | 91.1% | 100.0% | 100.0% | 0.0% | 100.0% | 100.0% | 2.21s |

- **Observation:** The full evaluation path worked with no errors. All five cases passed answer,
  numeric, citation, and faithfulness checks. This is only a smoke test over easy exact-numeric
  oracle cases; it must not be presented as the model's overall generation quality.
- **Decision:** Proceed to the full 60-case `oracle_plus_distractors` comparison (GEN-EXP-001).
