# Experiment 1: Generation Isolation Evaluation

## Objective

Evaluate the answer-generation capability of Qwen independently from retrieval, embedding, and reranking. Each question is supplied with its curated golden context, so this experiment measures whether the generator can produce grounded, relevant, and correct answers when the required evidence is already available.

## Experiment configuration

| Setting | Value |
|---|---|
| Golden dataset | `goldens/golden_generator_isolation_dataset.jsonl` |
| Cases attempted | 20 |
| Generator provider | Hugging Face via Featherless AI |
| Generator model | `Qwen/Qwen2.5-7B-Instruct` |
| Temperature | `0.0` |
| Maximum tokens | `1024` |
| Prompt version | `v1_strict_grounded` |
| Context source | Curated golden context |
| Judge model | `gpt-4o-mini` |
| Evaluation threshold | `0.7` |
| Strict mode | Disabled |
| Metrics | Faithfulness, Answer Relevancy, Answer Correctness |

Run ID:

```text
2026-10-23_13-32-38_huggingface_Qwen_Qwen2.5-7B-Instruct_featherless-ai
```

## Results

| Metric | Mean score | Pass rate | Passed / scored |
|---|---:|---:|---:|
| Faithfulness | 94.3% | 89.5% | 17 / 19 |
| Answer Relevancy | 90.8% | 89.5% | 17 / 19 |
| Answer Correctness | 82.9% | 89.5% | 17 / 19 |

### Run totals

| Measurement | Result |
|---|---:|
| Cases attempted | 20 |
| Successfully evaluated | 19 |
| Error cases | 1 |
| Prompt tokens | 17,107 |
| Completion tokens | 2,072 |
| Total generation latency | 66.86 seconds |
| Average latency per successful case | Approximately 3.52 seconds |

## Metric interpretation

### Faithfulness: 94.3%

Faithfulness measures whether claims in the generated answer are supported by the supplied context.

This is an excellent result. Qwen generally remained grounded in the filing evidence, preserved financial facts, and avoided introducing unsupported claims. The high score indicates that hallucination is not the primary generator-level problem in this experiment.

The pass rate was 89.5%, meaning 17 of 19 scored cases met the `0.7` threshold. Lower-scoring cases may contain an unsupported inference, an unnecessarily broadened statement, or a qualification not explicitly established by the context.

### Answer Relevancy: 90.8%

Answer Relevancy measures whether the response directly addresses the question.

This is also an excellent result. Responses were generally concise and focused on the requested facts rather than repeating the complete source context. The 89.5% pass rate indicates that 17 of 19 scored cases passed.

Lower relevancy can occur when a multi-part question is only partially answered—for example, when the model supplies the main value but omits a requested comparison, cause, consequence, or explanation.

### Answer Correctness: 82.9%

Answer Correctness compares the generated response with the expected reference answer.

This is a good result, but it is the clearest improvement opportunity. Although 17 of 19 cases passed, the lower mean score suggests that some passing responses contained the central answer but did not include every detail represented in the expected answer.

Probable causes include:

- omission of secondary facts in multi-part questions;
- missing comparison-year values;
- failure to include both dollar and percentage changes;
- selecting a table total instead of the requested row value;
- omitting an important reason, condition, or consequence;
- answers being too concise for the expected-answer coverage required by the judge.

The difference between high faithfulness and lower correctness is important: an answer can be fully supported by the context but still be incomplete relative to the question and reference answer.

## Error analysis

One case was not evaluated:

- Case: `q014`
- Question: Identify Microsoft's Executive Vice President and Chief Financial Officer and the officer's listed age.
- Error: `RetryError` caused by an underlying `TimeoutError`.

This was an API/infrastructure error rather than a scored model-quality failure. The case should be retried before treating this run as a complete 20-case model evaluation.

## Overall assessment

| Capability | Assessment |
|---|---|
| Grounding in supplied evidence | Excellent |
| Directness and relevance | Excellent |
| Reference-answer correctness | Good |
| Multi-part answer completeness | Good, but needs improvement |
| Hallucination control | Strong |
| API reliability | One retryable timeout |

The generator performs strongly when it receives the correct context. Its main weakness is not unsupported fabrication; it is completeness and precision when questions require several details. This indicates that generation is currently less problematic than retrieval in the complete RAG pipeline.

## Limitations

1. Only 19 of 20 cases were scored because one API request timed out.
2. The sample contains 20 questions, so one case changes a pass rate by approximately five percentage points.
3. DeepEval scores depend on the behavior of the `gpt-4o-mini` judge and may vary slightly between runs.
4. The experiment uses curated golden context and therefore does not measure retrieval quality.
5. A successful isolation result does not guarantee successful full-pipeline performance when retrieved context is incomplete or noisy.

## Recommended next experiment

1. Retry `q014` and obtain a complete 20-case result.
2. Review every case with a failed metric and its recorded judge reason.
3. Refine the generation prompt to require the model to:
   - answer every part of a multi-part question;
   - preserve exact units and reporting periods;
   - distinguish requested table rows from subtotals and totals;
   - include requested comparisons, causes, and consequences;
   - verify the final answer against the supplied context before returning it.
4. Rerun the same dataset with all other settings unchanged.
5. Compare the new run against Experiment 1 using:

```powershell
uv run python eval/deepEval/generator_eval.py --compare
```

Suggested targets for the next run:

- Faithfulness: maintain at least 90% mean score;
- Answer Relevancy: maintain at least 90% mean score;
- Answer Correctness: improve from 82.9% toward 88–90%;
- Error cases: reduce from 1 to 0.

## Result artifacts

```text
eval/deepEval/generation_results/
└── 2026-10-23_13-32-38_huggingface_Qwen_Qwen2.5-7B-Instruct_featherless-ai/
    ├── manifest.json
    ├── summary.json
    └── cases.jsonl
```
