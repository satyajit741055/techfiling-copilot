-- Reported XBRL figures (non-dimensional, annual or instant), each linked to the table
-- row chunk that shows it. Filled by: uv run python scripts/load_facts.py
-- The API loads the whole table into memory at startup (FACTS_STORE=postgres).

create table if not exists financial_facts (
    doc_id text not null,
    fact_id text not null,
    ticker text not null,
    filing_fiscal_year integer not null,
    fiscal_year integer not null,
    concept text not null,
    period_start date,
    period_end date not null,
    instant boolean not null,
    unit text not null,
    value double precision not null,
    shown double precision not null,
    scale integer not null,
    chunk_id text,
    table_id text,
    row_label text,
    excerpt text,
    primary key (doc_id, fact_id)
);

create index if not exists financial_facts_lookup_idx
    on financial_facts (ticker, concept, fiscal_year);

-- One row per load; the newest says which Qdrant collection the chunk ids belong to.
create table if not exists financial_facts_loads (
    id bigint generated always as identity primary key,
    loaded_at timestamptz not null default now(),
    collection text not null,
    chunk_run text not null,
    facts integer not null,
    linked integer not null
);
