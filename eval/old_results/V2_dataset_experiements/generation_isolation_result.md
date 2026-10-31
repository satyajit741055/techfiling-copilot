# Generation Isolation and Prompt Optimization Experiment

## Executive summary

This experiment isolated the answer generator from retrieval so that prompt quality could be evaluated without embedding, retrieval, or reranking errors. Qwen received curated golden context for every question, and DeepEval measured faithfulness, answer relevancy, and answer correctness.

The initial V1 prompt was highly grounded but sometimes omitted the second part of multi-part questions. A V2 prompt was designed to require complete answers, exact table alignment, preserved units and reporting periods, and concise focus on the requested entity or metric.

A controlled 20-case comparison using `gpt-4o` as the judge showed:

| Metric | V1 | V2 | Change |
|---|---:|---:|---:|
| Faithfulness mean | 100.0% | 100.0% | No change |
| Faithfulness pass rate | 100% | 100% | No change |
| Answer relevancy mean | 98.8% | 96.5% | -2.3 points |
| Answer relevancy pass rate | 100% | 95% | -5 points |
| Answer correctness mean | 88.3% | **91.5%** | **+3.2 points** |
| Answer correctness pass rate | 90% | **100%** | **+10 points** |
| Error cases | 0 | 0 | No change |

V2 won because it preserved perfect grounding while eliminating correctness failures caused by incomplete multi-part answers. Its only failed metric was an obvious answer-relevancy false negative on a direct, exact answer. V2 is therefore the production default, while V1 remains available for reproducible experiments.

## 1. Objective

The objective was to answer four questions:

1. Can Qwen produce faithful answers when the correct evidence is supplied?
2. Does it answer every part of financial-document questions?
3. Can prompt changes improve completeness without increasing hallucination?
4. How reliable are `gpt-4o-mini` and `gpt-4o` as DeepEval judges?

## 2. Why generation was evaluated in isolation

A full RAG response depends on several stages:

```text
question -> embedding -> retrieval -> reranking -> context -> generation
```

A poor final answer can be caused either by bad retrieved context or by poor use of good context. In this experiment, retrieval was removed from the causal chain:

```text
question + curated golden context -> generation -> DeepEval judge
```

Consequently:

- embedding backend does not affect this experiment;
- retriever and reranker quality do not affect the scores;
- failures can be attributed to generation, output completeness, judge behavior, or API reliability.

## 3. Dataset and case coverage

Dataset:

```text
goldens/golden_generator_isolation_dataset.jsonl
```

The dataset contains 20 questions from Apple, Microsoft, and Tesla 10-K filings. It covers tables, exact values, accounting policies, risks, business descriptions, comparisons, multi-part questions, and narrative disclosures.

| ID | Document | Question focus |
|---|---|---|
| `q001` | Apple | Services gross margin for 2025 and 2024 |
| `q002` | Microsoft | Effects of overestimating cloud/AI demand and insufficient capacity |
| `q003` | Tesla | Energy segment revenue and year-over-year dollar/percentage change |
| `q004` | Apple | Geographic areas included in the Europe segment |
| `q005` | Tesla | Inventory valuation and cost computation policy |
| `q006` | Microsoft | R&D amounts for 2026/2025 and percentage change |
| `q007` | Tesla | Battery-cell suppliers and supply-disruption consequence |
| `q008` | Microsoft | Total employees and R&D employees |
| `q009` | Tesla | Warranty-expense classification and 2025 provision |
| `q010` | Apple | Foreign-currency effects and international-pricing consequence |
| `q011` | Microsoft | Subscription versus consumption cloud revenue recognition |
| `q012` | Apple | Effective and statutory tax rates |
| `q013` | Tesla | Powerwall location and Megapack customer types |
| `q014` | Microsoft | CFO identity and age from an executive table |
| `q015` | Apple | Net accounting for App Store sales and reason for treatment |
| `q016` | Microsoft | Share-repurchase shares and amounts for 2026 versus 2025 |
| `q017` | Tesla | AI infrastructure and net PP&E table values |
| `q018` | Apple | Highest-sales fiscal quarter and seasonal reason |
| `q019` | Tesla | Four cybersecurity incident-response stages |
| `q020` | Microsoft | Locations of largest owned international properties |

## 4. Evaluation metrics

### Faithfulness

Determines whether claims in the generated answer are supported by the supplied context. A faithful answer may still be incomplete.

### Answer relevancy

Determines whether the answer directly addresses the question without unnecessary or unrelated information.

### Answer correctness

Compares the generated answer with the expected reference answer. This metric captures factual accuracy and completeness, including whether every requested part was answered.

Each metric used a threshold of `0.7`. Strict mode was disabled.

## 5. Fixed experiment configuration

| Setting | Value |
|---|---|
| Generator provider | Hugging Face via Featherless AI |
| Generator model | `Qwen/Qwen2.5-7B-Instruct` |
| Temperature | `0.0` |
| Maximum tokens | `1024` |
| Context source | Curated golden context |
| Cases | 20 |
| Threshold | `0.7` |
| Metrics | Faithfulness, Answer Relevancy, Answer Correctness |
| Strict mode | Disabled |

The controlled final comparison changed only the prompt version. Both V1 and V2 were judged by `gpt-4o`.

## 6. Baseline: V1 with `gpt-4o-mini`

Run:

```text
2026-10-23_13-32-38_huggingface_Qwen_Qwen2.5-7B-Instruct_featherless-ai
```

| Metric | Mean | Pass rate | Passed/scored |
|---|---:|---:|---:|
| Faithfulness | 94.3% | 89.5% | 17/19 |
| Answer relevancy | 90.8% | 89.5% | 17/19 |
| Answer correctness | 82.9% | 89.5% | 17/19 |

One request, `q014`, timed out. This was an API error rather than a model-quality failure.

The initial diagnosis was that Qwen was generally grounded and relevant, but answer correctness was reduced by omitted secondary details in multi-part questions.

## 7. Initial failed-case analysis

The original failed judgments included:

| Case | Metric | Observation |
|---|---|---|
| `q003` | Faithfulness | Judge incorrectly described reported table values as projections |
| `q004` | Answer relevancy | Judge claimed the directly requested Europe locations were irrelevant |
| `q010` | Answer correctness | Answer omitted the pricing-related potential reduction in demand |
| `q017` | Faithfulness | Judge incorrectly described a reported 2024 table value as projected |
| `q018` | Answer correctness | Answer identified the quarter but omitted seasonal holiday demand |
| `q020` | Answer relevancy | Original judgment did not align with the direct property-location answer |

This review revealed two separate issues:

1. **Real generator issue:** incomplete answers to multi-part questions.
2. **Judge issue:** false negatives and unsupported actual-versus-projected interpretations from `gpt-4o-mini`.

## 8. Prompt V2 design

V2 was named:

```text
v2_complete_precise
```

It added explicit requirements to:

- answer every requested part;
- identify requested facts, comparisons, causes, consequences, periods, entities, and units;
- remain focused on the requested entity, segment, geography, metric, or property;
- avoid inverse, excluded, neighboring, or merely related information;
- preserve actual versus projected/estimated language;
- preserve exact values, signs, currencies, units, and fiscal periods;
- match table values to the correct row, column, year, and total level;
- include the cause or reason when a question asks “why”;
- perform a final completeness and grounding check before responding.

V1 remains registered as `v1_strict_grounded` for reproducibility and regression testing.

## 9. Judge reliability investigation

### Two-case prompt check

The first controlled check reran `q003` and `q017` with V1 and V2. Both prompts then received 100% faithfulness. This showed that the earlier projection-related failures were not stable.

| Prompt | Faithfulness | Relevancy | Correctness |
|---|---:|---:|---:|
| V1, two cases | 100% | 100% | 84.9% |
| V2, two cases | 100% | 100% | 88.2% |

### V2 full run with `gpt-4o-mini`

Run:

```text
2026-10-23_15-52-04_huggingface_Qwen_Qwen2.5-7B-Instruct_featherless-ai
```

| Metric | Mean | Pass rate |
|---|---:|---:|
| Faithfulness | 88.8% | 85% |
| Answer relevancy | 96.2% | 95% |
| Answer correctness | 88.6% | 95% |

The faithfulness reduction was driven by cases such as `q004` and `q019`, even though their answers were nearly verbatim restatements of the context. This indicated judge instability rather than increased hallucination.

### Seven-case check with `gpt-4o`

Run:

```text
2026-10-23_16-18-51_huggingface_Qwen_Qwen2.5-7B-Instruct_featherless-ai
```

Cases:

```text
q003 q004 q010 q017 q018 q019 q020
```

| Metric | Mean | Pass rate |
|---|---:|---:|
| Faithfulness | 100% | 100% |
| Answer relevancy | 90.7% | 85.7% |
| Answer correctness | 89.9% | 100% |

`gpt-4o` correctly recognized all seven answers as faithful. Its only failed metric was answer relevancy for `q018`, where the answer included the requested quarter and seasonal reason but continued with unnecessary product-introduction details. This was a reasonable category for the issue: supported but unnecessary information affects relevancy, not faithfulness.

## 10. Final controlled comparison: V1 versus V2 with `gpt-4o`

### V1 run

```text
2026-10-23_16-29-54_huggingface_Qwen_Qwen2.5-7B-Instruct_featherless-ai
```

Configuration:

```text
prompt: v1_strict_grounded
judge:  gpt-4o
cases:  20
```

### V2 run

```text
2026-10-23_16-36-38_huggingface_Qwen_Qwen2.5-7B-Instruct_featherless-ai
```

Configuration:

```text
prompt: v2_complete_precise
judge:  gpt-4o
cases:  20
```

### Quality results

| Metric | V1 mean | V1 pass | V2 mean | V2 pass | Interpretation |
|---|---:|---:|---:|---:|---|
| Faithfulness | 100.0% | 100% | 100.0% | 100% | V2 preserved grounding |
| Answer relevancy | 98.8% | 100% | 96.5% | 95% | V1 higher automatically; V2 failure was a false negative |
| Answer correctness | 88.3% | 90% | **91.5%** | **100%** | V2 improved completeness and eliminated failures |

### Operational results

| Measurement | V1 | V2 | Difference |
|---|---:|---:|---:|
| Error cases | 0 | 0 | No change |
| Prompt tokens | 18,020 | **16,360** | V2 used 1,660 fewer |
| Completion tokens | **2,138** | 2,256 | V2 generated 118 more |
| Total latency | 65.42 s | **62.10 s** | V2 was 3.32 s faster in this run |

V2 used approximately 9.2% fewer prompt tokens while generating slightly more complete answers.

## 11. Key case studies

### Case `q015`: App Store accounting treatment

Question:

> How does Apple account for sales of third-party applications sold through the App Store, and what reason does it give for that treatment?

Expected components:

1. Net-basis accounting, recognizing only Apple’s retained commission.
2. Reason: Apple does not obtain control before transfer to the customer.

V1 answer:

> Apple accounts for all third-party application–related sales on a net basis by recognizing in Services net sales only the commission it retains.

The treatment was correct, but the reason was omitted from the answer field. The reason appeared only in the explanation, while DeepEval evaluates `actual_output` from the answer field.

V2 answer:

> Apple accounts for sales of third-party applications sold through the App Store on a net basis by recognizing in Services net sales only the commission it retains. This is because Apple does not obtain control of the product before transferring it to the customer.

V2 answered both requested parts and converted a V1 correctness failure into a pass.

### Case `q018`: seasonal sales pattern

Question:

> In which fiscal quarter has Apple historically experienced higher net sales compared with its other quarters, and what reason does Apple give for this pattern?

Expected components:

1. First fiscal quarter.
2. Seasonal holiday demand.

V1 answer:

> Apple has historically experienced higher net sales in its first quarter compared with other quarters in its fiscal year.

V1 omitted the requested reason from the answer field and failed correctness.

V2 answer began:

> Apple has historically experienced higher net sales in its first quarter compared with other quarters in its fiscal year due to seasonal holiday demand.

V2 included both components and passed correctness. It subsequently added supported but unnecessary product-introduction details, showing a remaining opportunity to stop after fully answering the question.

### Case `q004`: judge false negative

Question:

> Which geographic areas are included in Apple’s Europe reportable segment?

V2 answer:

> The geographic areas included in Apple's Europe reportable segment are European countries, as well as India, the Middle East and Africa.

This directly matches the source and expected answer. Nevertheless, one `gpt-4o` answer-relevancy judgment failed it. Manual review classifies this as a judge false negative rather than a prompt failure.

### Cases `q003` and `q017`: actual versus projected judge errors

The earlier `gpt-4o-mini` judge described reported table values as projections. The questions, context, and answers all clearly referred to reported historical values. Reruns with `gpt-4o` passed faithfulness, confirming that the questions did not need to be rewritten.

## 12. Production decision

Production now defaults to:

```text
prompt_version = v2_complete_precise
SYSTEM_PROMPT = SYSTEM_PROMPT_V2
```

`AnswerGenerator` resolves its system prompt from `GenerationConfig.prompt_version` when a custom prompt is not supplied. This keeps the logged prompt version and actual prompt text consistent.

V1 remains available explicitly:

```python
GenerationConfig(prompt_version="v1_strict_grounded")
```

The evaluator also supports both versions:

```powershell
uv run python eval/deepEval/generator_eval.py --prompt-version v1_strict_grounded
uv run python eval/deepEval/generator_eval.py --prompt-version v2_complete_precise
```

## 13. Final conclusion

V2 is the preferred production prompt because it:

- maintained 100% faithfulness under the stronger judge;
- improved correctness from 88.3% to 91.5%;
- improved correctness pass rate from 90% to 100%;
- fixed concrete omissions in multi-part questions;
- reduced prompt-token usage by 9.2%;
- completed all 20 cases without errors.

The small automated relevancy decrease should be interpreted cautiously because its only threshold failure was a direct and correct answer. However, `q018` shows that V2 can sometimes add supported but unnecessary details. A future prompt iteration could add: “After answering every requested part, stop; do not add other context unless necessary.” This should be tested as a new version rather than silently modifying V2.

## 14. explanation


> I evaluated generation independently from retrieval by giving the model curated golden context for 20 financial-filing questions. The initial prompt was well grounded but sometimes answered only the first part of multi-part questions. I designed a second prompt that explicitly decomposes each question into requested facts, comparisons, causes, periods, and units, and asks the model to verify completeness before responding. I then ran a controlled A/B test with the same Qwen model, dataset, temperature, threshold, and GPT-4o judge. Faithfulness stayed at 100%, while answer correctness improved from 88.3% to 91.5%, and the correctness pass rate improved from 90% to 100%. The new prompt also used about 9% fewer prompt tokens. During the experiment I found that GPT-4o-mini produced obvious false-negative faithfulness judgments, so I validated suspicious cases manually and repeated the final comparison with GPT-4o. This demonstrated both prompt improvement and the importance of auditing LLM-as-judge evaluations.

Key engineering lessons:

1. Isolate pipeline stages before attributing RAG failures.
2. Use case-level judge reasons, not only aggregate scores.
3. Distinguish faithfulness from completeness and relevancy.
4. Treat LLM judges as noisy evaluators, not ground truth.
5. Keep prompt versions and hashes in manifests for reproducibility.
6. Change one experimental variable at a time.
7. Preserve old prompt versions for regression testing.

## 15. Reproduction commands

V1:

```powershell
uv run python eval/deepEval/generator_eval.py `
  --prompt-version v1_strict_grounded `
  --judge-model gpt-4o `
  --verbose
```

V2:

```powershell
uv run python eval/deepEval/generator_eval.py `
  --prompt-version v2_complete_precise `
  --judge-model gpt-4o `
  --verbose
```

Compare saved runs:

```powershell
uv run python eval/deepEval/generator_eval.py --compare
```

## 16. Result artifacts

```text
eval/deepEval/generation_results/
├── 2026-10-23_13-32-38_huggingface_Qwen_Qwen2.5-7B-Instruct_featherless-ai/  # V1, mini judge
├── 2026-10-23_15-52-04_huggingface_Qwen_Qwen2.5-7B-Instruct_featherless-ai/  # V2, mini judge
├── 2026-10-23_16-18-51_huggingface_Qwen_Qwen2.5-7B-Instruct_featherless-ai/  # V2, 7-case GPT-4o check
├── 2026-10-23_16-29-54_huggingface_Qwen_Qwen2.5-7B-Instruct_featherless-ai/  # V1, final GPT-4o run
└── 2026-10-23_16-36-38_huggingface_Qwen_Qwen2.5-7B-Instruct_featherless-ai/  # V2, final GPT-4o run
```

Each run contains:

- `manifest.json`: exact configuration, command, model, judge, prompt version, and prompt hash;
- `summary.json`: aggregate metrics and operational totals;
- `cases.jsonl`: generated answers, per-case scores, pass/fail status, and judge reasons.
