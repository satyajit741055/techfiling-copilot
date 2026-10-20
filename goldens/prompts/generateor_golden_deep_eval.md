You are working in this repository:

C:\Users\Satyajit_Khot\Downloads\techfiling-copilot

Your task is to create a carefully verified golden dataset for evaluating a
question-answering generator in isolation using DeepEval.

The generator will receive a question and an ideal context. The dataset must
support:

1. Faithfulness evaluation
   - input: query
   - actual_output: answer produced by the generator
   - retrieval_context: ideal_context

2. Answer Relevancy evaluation
   - input: query
   - actual_output: answer produced by the generator
   - This metric is reference-free and does not use expected_answer.

3. Answer Correctness evaluation
   - input: query
   - actual_output: answer produced by the generator
   - expected_output: expected_answer

AUTHORITATIVE SOURCE FILES

Generate the dataset only from parsed filing files matching:

C:\Users\Satyajit_Khot\Downloads\techfiling-copilot\data\parsed\*.elements.jsonl

Each non-empty line in an `.elements.jsonl` file is a JSON object. Relevant
fields include:

- `doc_id`
- `order`
- `type`
- `part`
- `section`
- `section_title`
- `text`
- `table.markdown`

The parsed files are the only authoritative source. Do not use external
knowledge, web searches, existing golden answers, model knowledge, or facts
that are not explicitly present in these files.

OUTPUT FILE

Create:

C:\Users\Satyajit_Khot\Downloads\techfiling-copilot\goldens\golden_generator_isolation_dataset.jsonl

The file must contain exactly 20 valid JSONL records.

Each physical line must contain exactly one complete JSON object. Do not wrap
the records in a JSON array. Do not include Markdown fences or explanatory
text in the output file.

REQUIRED RECORD SCHEMA

{
  "source_qid": "q001",
  "doc_id": "<exact source doc_id>",
  "query": "<clear and standalone question>",
  "expected_answer": "<concise ideal answer>",
  "ideal_context": [
    "<verbatim source context>"
  ],
  "source_element_order": [
    <exact source order integer>
  ]
}

CRITICAL GENERATION WORKFLOW

Do not generate all 20 questions in one bulk model response.

Generate and validate exactly one question at a time by following this loop:

1. Select one or, only when necessary, two source elements.
2. Read the complete original JSON object for each selected element.
3. Create one question from those elements.
4. Create its expected answer.
5. Copy its ideal context directly from the selected elements.
6. Validate the record against the authoritative source files.
7. Append the record to the dataset only if every validation check passes.
8. Move to the next question only after the current question is valid.
9. Continue until exactly 20 validated records exist.

Never draft several questions together and validate them afterward.

If a candidate fails validation, discard that candidate completely. Do not
append it and do not count it toward the 20 records. Select different evidence
and create a replacement.

If necessary, keep each candidate in memory or in a temporary file while it is
being validated. Only validated records may be written to the final dataset.

FIELD REQUIREMENTS

1. `source_qid`

- Use sequential IDs from `q001` through `q020`.
- IDs must be unique.
- Do not skip or repeat an ID.
- Assign an ID only when the record is ready to be appended.

2. `doc_id`

- Copy `doc_id` exactly from the selected source element.
- All elements used by one record must have the same `doc_id`.
- Do not create cross-document questions.
- Never infer `doc_id` from a filename when it can be read from the element.

3. `query`

Each query must:

- Be natural, precise, grammatically correct, and standalone.
- Have exactly one clearly supported answer.
- Identify the company when needed.
- Identify the reporting period or date when needed.
- Identify the relevant unit when needed.
- Be answerable exclusively from `ideal_context`.
- Require no external knowledge.
- Avoid ambiguous expressions such as "the company," "that year," or "the
  table" when their meaning is not explicit.
- Avoid yes/no questions when a more informative question can be asked.
- Avoid copying a source sentence and mechanically turning it into a question.
- Avoid asking for information that is merely implied rather than stated.
- Avoid duplicate and near-duplicate questions.
- Avoid combining unrelated facts.
- Avoid unsupported calculations.

4. `expected_answer`

Each expected answer must:

- Be concise and directly answer the query.
- Be completely supported by `ideal_context`.
- Contain no facts, explanations, numbers, units, dates, qualifications, or
  conclusions that are absent from the context.
- Preserve source units and precision.
- Preserve whether a value is reported in thousands, millions, or billions.
- Preserve whether a number is a currency, percentage, count, or ratio.
- Preserve the correct fiscal year, reporting period, and date.
- Avoid adding background information from model knowledge.
- Avoid claiming causation unless the source explicitly states causation.
- Avoid silently converting units.
- Avoid silently rounding values.
- Avoid calculations unless the question explicitly asks for a simple,
  deterministic calculation and every operand is present in the context.

The expected answer does not need to be copied word-for-word, but every claim
in it must be directly entailed by the context.

5. `ideal_context`

Use the following canonical context extraction rule:

- For `heading` or `paragraph` elements:
  use the element's `text` value exactly.

- For `table` elements:
  use the element's `table.markdown` value exactly.

Requirements:

- Copy context verbatim from the authoritative source element.
- Do not summarize it.
- Do not paraphrase it.
- Do not correct spelling or punctuation.
- Do not normalize whitespace.
- Do not change capitalization.
- Do not change Unicode characters.
- Do not change currency symbols.
- Do not truncate long elements.
- Do not join multiple source elements into one context string.
- Do not insert section titles unless the section-title element is itself
  selected as evidence.
- Do not use the serialized source JSON object as context.
- Do not use null or empty source content.
- JSON escaping is allowed only as required for valid JSON. After JSON parsing,
  the resulting string must exactly equal the authoritative source value.

Use only the minimum sufficient context:

- Prefer exactly one source element when it is sufficient.
- Use two source elements only when both are genuinely necessary.
- Do not add irrelevant neighboring elements.
- Do not add distractors.
- Do not add broad sections merely for additional context.
- Do not use more than two source elements for a question.

6. `source_element_order`

- Copy the integer `order` directly from each selected element.
- Do not use the physical JSONL line number.
- Do not estimate or renumber an order.
- `source_element_order` and `ideal_context` must have equal lengths.
- They must align by array index.
- `ideal_context[i]` must be the canonical context extracted from the element
  whose `order` is `source_element_order[i]`.
- When two elements are used, list them in ascending source order.
- Do not repeat an order in one record.

QUESTION DISTRIBUTION

Create a meaningful and diverse set of 20 questions. Target approximately:

- 6 financial-table questions
- 4 business or operational questions
- 3 risk-factor questions
- 3 accounting-policy or financial-note questions
- 2 questions requiring exactly two source elements
- 2 questions from other substantive filing content

These categories may overlap slightly, but the completed dataset must not be
dominated by one section, company, question pattern, or answer type.

Include a useful mix of:

- Direct factual extraction
- Numeric financial extraction
- Period-specific values
- Comparisons explicitly shown in one source element
- Qualitative explanations
- Stated risks and consequences
- Limited two-element synthesis

TABLE QUESTION RULES

Be especially careful with tables.

A table-based question is valid only if its context unambiguously contains:

- The required row label
- The required column or reporting period
- The reported value
- The unit, when a unit is supplied by the source
- Any necessary table title or heading

Do not:

- Associate a value with the wrong year or column.
- Confuse totals and subtotals.
- Confuse percentages with absolute values.
- Confuse values in millions with values in billions.
- Infer a table heading that is absent from the selected context.
- Ask about a value when multiple cells could reasonably be the answer.
- Repair malformed or ambiguous parsed tables using outside knowledge.

If `table.markdown` does not independently preserve enough information to
answer the proposed question unambiguously, discard that candidate.

NO-HALLUCINATION REQUIREMENTS

The golden dataset itself must contain no hallucinated information.

For every factual phrase in `query` and `expected_answer`, identify the exact
source text that supports it.

Do not:

- Invent facts, values, dates, units, periods, companies, or relationships.
- Fill in missing information from general knowledge.
- Use facts remembered from SEC filings.
- infer a value from a filename.
- Combine evidence from different documents.
- Treat model confidence as evidence.
- Correct an apparent source error using outside knowledge.
- Include unsupported qualifiers such as "approximately," "primarily,"
  "significantly," or "mainly."
- State that one event caused another unless the selected context explicitly
  says so.

When evidence is incomplete, unclear, malformed, or ambiguous, discard the
candidate rather than attempting to repair it.

CREATE A VALIDATOR

Also create:

C:\Users\Satyajit_Khot\Downloads\techfiling-copilot\scripts\eval_scripts\validate_generator_isolation_dataset.py

Follow the repository's existing Python conventions and dependency setup.
Prefer the Python standard library unless an existing project dependency is
clearly appropriate.

The validator must default to validating:

goldens/golden_generator_isolation_dataset.jsonl

It should also accept an optional dataset path through a command-line
argument.

The validator must load all authoritative source files from:

data/parsed/*.elements.jsonl

Build an index using:

(doc_id, order)

For each indexed element, calculate canonical context as:

- `element["text"]` for `heading` and `paragraph`
- `element["table"]["markdown"]` for `table`

VALIDATOR CHECKS

The validator must detect and clearly report all of the following:

1. Dataset file does not exist.
2. Invalid UTF-8, if applicable.
3. Blank lines.
4. Invalid JSON.
5. A line that is not a JSON object.
6. Missing required fields.
7. Unexpected types for required fields.
8. Exactly 20 records are required.
9. Duplicate `source_qid` values.
10. IDs must be exactly `q001` through `q020`, in that order.
11. Empty `doc_id`.
12. Empty `query`.
13. Empty `expected_answer`.
14. Empty `ideal_context`.
15. Empty `source_element_order`.
16. Unequal lengths of `ideal_context` and `source_element_order`.
17. More than two context elements in a record.
18. Non-string context values.
19. Non-integer source orders; booleans must not count as integers.
20. Duplicate source orders within a record.
21. Source orders not listed in ascending order.
22. Missing `(doc_id, order)` references.
23. Context selected from a different document.
24. Unsupported source element types.
25. Null or empty canonical source context.
26. Exact context mismatch between the decoded dataset string and the
    canonical source string.
27. Duplicate normalized queries.
28. Exact duplicate records.
29. Invalid source files or duplicate `(doc_id, order)` keys in parsed data.
30. Trailing or extra content that prevents a line from being valid JSON.

For context comparisons, compare decoded Python strings exactly. Do not compare
the raw escaped JSON representation.

The validator should collect and display all discovered errors rather than
stopping at the first record error, except when the dataset or source files
cannot be loaded safely.

VALIDATOR EXIT BEHAVIOR

- Exit with status code 0 only when every check passes.
- Exit with a nonzero status code when any error is found.
- Print a concise success summary when validation passes.
- Print actionable errors containing, where available:
  - dataset line number
  - source_qid
  - doc_id
  - source order
  - failed rule
- Do not modify the dataset automatically.
- Do not silently repair invalid records.

SEMANTIC REVIEW

The deterministic validator can prove source identity and schema correctness,
but it cannot fully prove that a question is answerable or that an expected
answer is semantically entailed.

Therefore, before accepting each individual record, perform a separate
semantic review and confirm:

1. The question is unambiguous.
2. The context is minimally sufficient.
3. The expected answer directly answers the question.
4. Every answer claim is explicitly supported.
5. The correct date, period, column, row, and unit were used.
6. No outside knowledge was introduced.
7. No alternative answer is reasonably supported by the same context.
8. No irrelevant context was included.

GENERATION AND VALIDATION ORDER

Use this exact execution order:

1. Inspect repository instructions and relevant existing scripts.
2. Implement the validator.
3. Verify that the validator can run.
4. Create the dataset file.
5. Generate candidate `q001`.
6. Semantically review `q001`.
7. Append `q001`.
8. Run the validator against the current partial dataset using an explicit
   partial-generation mode, or validate the record using the same validation
   functions without enforcing the final 20-record count.
9. If validation fails, remove or replace only the current candidate.
10. Proceed to `q002` only after `q001` passes.
11. Repeat the one-question-at-a-time process through `q020`.
12. Run the complete validator with all final-count and sequence checks enabled.
13. Manually inspect the final JSONL for source fidelity and semantic quality.
14. Report the created files and the final validation result.

The validator may support a `--partial` option for steps 8–11. In partial mode:

- Permit between 1 and 20 records.
- Require IDs to be sequential from `q001` through the current final record.
- Enforce every other validation rule.
- Do not weaken exact context verification.

Do not claim completion unless:

- The dataset contains exactly 20 records.
- Every record was generated and validated individually.
- The final full validation exits successfully.
- All ideal contexts exactly match their authoritative source elements.
- Every expected answer is supported by its ideal context.
- No hallucinated information is present.