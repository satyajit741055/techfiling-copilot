-- Supabase only: needs auth.uid() and the anon/authenticated roles.
-- Apply with: uv run python scripts/apply_migrations.py --supabase
-- Signed-in users may read their own rows through the Supabase client. There are no
-- insert/update/delete policies, so only the API (server-side, as the table owner over
-- HISTORY_DB_URL) can write.

-- Supabase grants anon and authenticated full privileges on new public tables, and RLS
-- does not cover TRUNCATE. Keep only SELECT for signed-in users; the policies below
-- narrow it to their own rows.
revoke all on table threads, messages, feedback, schema_migrations from anon, authenticated;
grant select on table threads, messages, feedback to authenticated;

alter table threads enable row level security;
alter table messages enable row level security;
alter table feedback enable row level security;
-- Bookkeeping for apply_migrations.py: RLS with no policies hides it from the Data API.
alter table schema_migrations enable row level security;

drop policy if exists threads_select_own on threads;
create policy threads_select_own on threads
    for select to authenticated using ((select auth.uid()) = user_id);

drop policy if exists messages_select_own on messages;
create policy messages_select_own on messages
    for select to authenticated using ((select auth.uid()) = user_id);

drop policy if exists feedback_select_own on feedback;
create policy feedback_select_own on feedback
    for select to authenticated using ((select auth.uid()) = user_id);
