-- Conversation history: plain Postgres (13+), so it runs on Docker Postgres and Supabase.
-- Supabase-only pieces (Row Level Security) live in db/supabase/.
-- Apply with: uv run python scripts/apply_migrations.py

create table if not exists threads (
    id uuid primary key,
    user_id uuid not null,
    title text not null default '',
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create index if not exists threads_user_updated_idx on threads (user_id, updated_at desc);

create table if not exists messages (
    id uuid primary key default gen_random_uuid(),
    thread_id uuid not null references threads (id) on delete cascade,
    user_id uuid not null,
    turn_id text,
    question text not null,
    status text not null check (status in ('answered', 'clarification', 'refused', 'error')),
    route text,
    answer_markdown text,
    answer jsonb,
    citations jsonb,
    assumptions jsonb,
    coverage jsonb,
    scope jsonb,
    is_valid boolean,
    validation_errors jsonb,
    trajectory jsonb,
    model text,
    prompt_tokens integer check (prompt_tokens >= 0),
    completion_tokens integer check (completion_tokens >= 0),
    latency_ms integer check (latency_ms >= 0),
    cost_usd numeric(12, 6) check (cost_usd >= 0),
    created_at timestamptz not null default now()
);

create index if not exists messages_thread_created_idx on messages (thread_id, created_at);

-- Table only; the feedback endpoint arrives in M3.
create table if not exists feedback (
    id uuid primary key default gen_random_uuid(),
    message_id uuid not null references messages (id) on delete cascade,
    user_id uuid not null,
    rating smallint not null,
    comment text,
    created_at timestamptz not null default now()
);

create index if not exists feedback_message_idx on feedback (message_id);
