# Old Approach — HTML → Markdown → Chunk (V1 Prototype)

> **This folder is kept as a historical reference.** It shows the first pipeline
> built before the structured parser. Reading it alongside the current parser
> (`src/ingestion/parser/parser.py`) tells the full engineering story.

---

## What is in this folder

```
old_approach/
├── html_to_markdown_pipeline.ipynb   ← Full self-contained pipeline notebook
├── data/
│   ├── raw/                          ← Downloaded HTML filings (generated on run)
│   ├── processed_md_files/           ← Cleaned Markdown files (generated on run)
│   ├── chunks/                       ← Chunked JSON output (generated on run)
│   └── qdrant_storage/               ← Local Qdrant vector DB (generated on run)
└── README.md                         ← This file
```

All data produced by the notebook is written inside `data/` — nothing spills
outside this folder.

---

## How to run the notebook

1. Open `html_to_markdown_pipeline.ipynb` in Jupyter or VS Code.
2. Make sure you are in the `old_approach/` directory when the kernel starts
   (the notebook resolves paths relative to `Path().resolve()` which is the cwd).
3. Set `OPENAI_API_KEY` in your environment before running Step 4 (embedding).
4. Run cells top-to-bottom. Each step writes its output to `data/`.

**Dependencies** (install in your environment):
```
pip install requests beautifulsoup4 lxml markdownify langchain-text-splitters \
            openai qdrant-client numpy
```

---

## The Pipeline (6 steps)

| Step | What it does | Output |
|---|---|---|
| 1 | Download 10-K HTML from SEC EDGAR | `data/raw/*.html` |
| 2 | Strip iXBRL noise, promote `Item N.` headings, convert to Markdown | `data/processed_md_files/*.md` |
| 3 | Detect section boundaries by `## Item N.` regex, assign strategy, chunk | `data/chunks/apple_10K.json` |
| 4 | Embed all chunks with `text-embedding-3-small`, store in Qdrant | `data/qdrant_storage/` |
| 5 | Retrieve top-k chunks for a query | (in-memory) |
| 6 | Generate an answer with `gpt-4o-mini` | printed output |

---

## Why this approach was replaced

This prototype worked well enough to validate the overall RAG idea, but had three
fundamental problems that only showed up when running on real 10-K data:

### Problem 1 — Noisy table data

iXBRL tables use extra `<td>` cells as visual spacers and emit `$` as a separate
cell from the number. After `markdownify` conversion:

```
Raw Markdown:  | | | | $ | 307,003 | | | | | $ | 294,866 | | |
Should be:     | Products | $307,003 | $294,866 |
```

`clean_table_text()` in Step 3 partially fixes this, but can't fully recover the
structural information that was lost during HTML→Markdown flattening.

### Problem 2 — No way to verify correctness

A wrong number (e.g. wrong sign, wrong scale) looks identical to a correct one.
For a financial Q&A system this is unacceptable — there is no signal to catch
a bad parse before it reaches the user.

### Problem 3 — Brittle section detection

Section boundaries are found by regex on `## Item N.` in flat Markdown text.
This breaks silently if:
- A filing has non-standard section numbering
- The heading ends up in a table row instead of as a standalone line
- A company uses unusual formatting

### The fix — structured parsing

The current parser (`src/ingestion/parser/parser.py`) solves all three:

1. **Tables** are parsed from the HTML DOM directly — colspan/rowspan expanded,
   fragments merged, columns inferred, row hierarchy reconstructed from CSS indentation.
2. **Validation** — every number is cross-checked against the XBRL fact embedded in
   the same file (`value_match`, `sign_match`, period purity).
3. **Sections** — each element carries a structured `(part, section, section_title)`
   triple stamped at parse time, not recovered from Markdown text.

See `study/01.1_parser_appraoch.md` for the full comparison with code examples,
side-by-side output, and interview Q&A.

---

## Key lesson

> **Do structural work before serialisation, not after.**
>
> Flattening HTML to Markdown is lossy compression. You cannot fully reconstruct
> table column alignment, row hierarchy, or XBRL metadata from the compressed output.
> The information must be extracted from the DOM while it is still intact.
