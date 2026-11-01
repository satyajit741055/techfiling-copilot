'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useAuth } from '@/components/AuthProvider';
import { ChatAnswerCard } from '@/components/ChatAnswerCard';
import { ClarificationPrompt } from '@/components/ClarificationPrompt';
import { ErrorBanner } from '@/components/ErrorBanner';
import { AlertIcon, ArrowUp, ChatIcon } from '@/components/icons';
import { QuotaBadge } from '@/components/QuotaBadge';
import { PageSkeleton } from '@/components/Skeleton';
import { StepTimeline } from '@/components/StepTimeline';
import { ThreadSidebar, type ThreadItem } from '@/components/ThreadSidebar';
import {
  APIError,
  friendlyError,
  getFilings,
  getThreadMessages,
  getThreads,
  streamChat,
  type Quota,
} from '@/lib/api';
import {
  applyEvent,
  exampleQuestions,
  loadLocalThreads,
  newTurn,
  saveLocalThreads,
  threadTitle,
  turnsFromMessages,
  type ChatTurnView,
  type LocalThread,
} from '@/lib/chat';
import type { ChatRequestBody, Filing, ThreadSummary } from '@/types';

const MAX_MESSAGE_CHARS = 2000;
const MAX_TICKERS = 6;
const MAX_YEARS = 10;

function Chip({ active, onClick, children }: { active: boolean; onClick: () => void; children: React.ReactNode }) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-pressed={active}
      className={`rounded-full border px-2.5 py-0.5 text-xs font-medium transition-colors ${
        active ? 'border-brand-600 bg-brand-600 text-white' : 'border-slate-300 bg-white text-slate-600 hover:bg-slate-100'
      }`}
    >
      {children}
    </button>
  );
}

function toggle<T>(values: T[], value: T, max: number): T[] {
  if (values.includes(value)) return values.filter((v) => v !== value);
  return values.length >= max ? values : [...values, value];
}

export function ChatWorkspace() {
  const { user, signOut } = useAuth();
  const userId = user?.id ?? null;

  const [filings, setFilings] = useState<Filing[]>([]);
  const [initialLoading, setInitialLoading] = useState(true);
  const [threadId, setThreadId] = useState<string | null>(null);
  const [turns, setTurns] = useState<ChatTurnView[]>([]);
  const [input, setInput] = useState('');
  const [chipTickers, setChipTickers] = useState<string[]>([]);
  const [chipYears, setChipYears] = useState<number[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [cooldown, setCooldown] = useState(0);
  const [quota, setQuota] = useState<Quota | null>(null);

  const [historySource, setHistorySource] = useState<'server' | 'local' | 'loading'>('loading');
  const [serverThreads, setServerThreads] = useState<ThreadSummary[]>([]);
  const [localThreads, setLocalThreads] = useState<LocalThread[]>(() => loadLocalThreads(userId));
  const [historyOpen, setHistoryOpen] = useState(false);

  const abortRef = useRef<AbortController | null>(null);
  const endRef = useRef<HTMLDivElement>(null);
  // Set when a turn streams, so opening a saved local thread does not re-save it.
  const unsavedRef = useRef(false);

  const handleError = useCallback(
    (e: unknown) => {
      const { message, retryAfter, status } = friendlyError(e);
      setError(message);
      if (status === 429 && retryAfter) setCooldown(retryAfter);
      if (status === 401) signOut();
      return message;
    },
    [signOut]
  );

  const refreshThreads = useCallback(
    () =>
      getThreads().then(
        (threads) => {
          setServerThreads(threads);
          setHistorySource('server');
        },
        (e: unknown) => {
          // 404: server history is off (or no owner) — keep conversations in this browser.
          setHistorySource('local');
          if (!(e instanceof APIError) || e.status !== 404) console.warn('Could not load saved conversations');
        }
      ),
    []
  );

  useEffect(() => {
    getFilings()
      .then((response) => setFilings(response.filings))
      .catch(handleError)
      .finally(() => setInitialLoading(false));
    void refreshThreads();
  }, [handleError, refreshThreads]);

  useEffect(() => {
    if (cooldown <= 0) return;
    const id = setTimeout(() => setCooldown((c) => c - 1), 1000);
    return () => clearTimeout(id);
  }, [cooldown]);

  useEffect(() => () => abortRef.current?.abort(), []);

  useEffect(() => {
    if (busy) endRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' });
  }, [turns, busy]);

  // Browser fallback: keep finished conversations when the server does not save them.
  useEffect(() => {
    if (historySource !== 'local' || !threadId || turns.length === 0 || turns.some((t) => t.streaming)) return;
    if (!unsavedRef.current) return;
    unsavedRef.current = false;
    setLocalThreads((prev) => {
      const next = [
        { id: threadId, title: threadTitle(turns[0].question), updatedAt: new Date().toISOString(), turns },
        ...prev.filter((thread) => thread.id !== threadId),
      ];
      saveLocalThreads(userId, next);
      return next;
    });
  }, [historySource, threadId, turns, userId]);

  const tickers = useMemo(() => [...new Set(filings.map((f) => f.ticker))].sort(), [filings]);
  const years = useMemo(() => [...new Set(filings.map((f) => f.fiscal_year))].sort((a, b) => a - b), [filings]);
  const coverage = useMemo(
    () =>
      tickers.map((ticker) => {
        const own = filings.filter((f) => f.ticker === ticker);
        const fy = own.map((f) => f.fiscal_year).sort((a, b) => a - b);
        return { ticker, company: own[0]?.company ?? ticker, range: fy.length > 1 ? `FY${fy[0]}–FY${fy[fy.length - 1]}` : `FY${fy[0]}` };
      }),
    [filings, tickers]
  );
  const examples = useMemo(() => exampleQuestions(filings), [filings]);

  const updateTurn = useCallback((id: string, fn: (turn: ChatTurnView) => ChatTurnView) => {
    setTurns((prev) => prev.map((turn) => (turn.id === id ? fn(turn) : turn)));
  }, []);

  const run = useCallback(
    async (body: ChatRequestBody, turnId: string) => {
      abortRef.current?.abort();
      const controller = new AbortController();
      abortRef.current = controller;
      unsavedRef.current = true;
      setBusy(true);
      setError(null);
      try {
        const nextQuota = await streamChat(
          body,
          (event) => {
            if (event.event === 'thread') setThreadId(event.data.thread_id);
            updateTurn(turnId, (turn) => applyEvent(turn, event));
          },
          controller.signal
        );
        if (nextQuota) setQuota(nextQuota);
      } catch (e) {
        const stopped = controller.signal.aborted;
        const message = stopped ? 'Stopped.' : handleError(e);
        updateTurn(turnId, (turn) => ({
          ...turn,
          error: turn.error ?? message,
          streaming: false,
          timeline: turn.timeline.map((item) => (item.status === 'running' ? { ...item, status: 'error' } : item)),
        }));
      } finally {
        updateTurn(turnId, (turn) => (turn.streaming ? { ...turn, streaming: false } : turn));
        if (abortRef.current === controller) abortRef.current = null;
        setBusy(false);
        if (historySource === 'server') void refreshThreads();
      }
    },
    [handleError, historySource, refreshThreads, updateTurn]
  );

  const send = useCallback(
    (text: string) => {
      const message = text.trim();
      if (!message || busy) return;
      const turn = newTurn(message);
      setTurns((prev) => [...prev, turn]);
      setInput('');
      const filters =
        chipTickers.length || chipYears.length ? { tickers: chipTickers, fiscal_years: chipYears } : undefined;
      void run({ message, ...(threadId && { thread_id: threadId }), ...(filters && { filters }) }, turn.id);
    },
    [busy, chipTickers, chipYears, run, threadId]
  );

  const resume = useCallback(
    (turnId: string, choice: { option_id: string } | { text: string }, label: string) => {
      if (!threadId || busy) return;
      updateTurn(turnId, (turn) => ({ ...turn, clarificationChoice: label, error: undefined, done: undefined, streaming: true }));
      void run({ thread_id: threadId, resume: choice }, turnId);
    },
    [busy, run, threadId, updateTurn]
  );

  const openThread = useCallback(
    async (id: string) => {
      setHistoryOpen(false);
      setError(null);
      if (historySource === 'local') {
        const thread = localThreads.find((t) => t.id === id);
        setThreadId(id);
        setTurns(thread?.turns ?? []);
        return;
      }
      try {
        const messages = await getThreadMessages(id);
        setThreadId(id);
        setTurns(turnsFromMessages(messages));
      } catch (e) {
        handleError(e);
      }
    },
    [handleError, historySource, localThreads]
  );

  const newChat = useCallback(() => {
    abortRef.current?.abort();
    setThreadId(null);
    setTurns([]);
    setError(null);
    setHistoryOpen(false);
  }, []);

  const threadItems: ThreadItem[] =
    historySource === 'server'
      ? serverThreads.map((t) => ({ id: t.id, title: t.title, updatedAt: t.updated_at }))
      : localThreads.map((t) => ({ id: t.id, title: t.title, updatedAt: t.updatedAt }));

  if (initialLoading) return <PageSkeleton />;

  const rateLimited = cooldown > 0;
  const bannerMessage = rateLimited ? `You have reached the question limit. You can ask again in ${cooldown}s.` : error;
  const canSend = input.trim().length > 0 && !busy && !rateLimited;

  return (
    <div className="space-y-4">
      <div className="lg:hidden">
        <button
          onClick={() => setHistoryOpen((v) => !v)}
          className="flex items-center gap-2 rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm font-medium text-slate-700"
        >
          <ChatIcon /> Conversations ({threadItems.length})
        </button>
      </div>

      <div className="grid gap-6 lg:grid-cols-[280px_minmax(0,1fr)]">
        <aside className={`${historyOpen ? 'block' : 'hidden'} lg:sticky lg:top-24 lg:block lg:self-start`}>
          <ThreadSidebar
            threads={threadItems}
            activeId={threadId}
            source={historySource}
            disabled={busy}
            onSelect={(id) => void openThread(id)}
            onNew={newChat}
            onDelete={
              historySource === 'local'
                ? (id) => {
                    const next = localThreads.filter((t) => t.id !== id);
                    setLocalThreads(next);
                    saveLocalThreads(userId, next);
                    if (id === threadId) newChat();
                  }
                : undefined
            }
          />
        </aside>

        <section className="min-w-0 space-y-6">
          {bannerMessage && (
            <ErrorBanner message={bannerMessage} onDismiss={rateLimited ? undefined : () => setError(null)} />
          )}

          {turns.length === 0 && (
            <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
              <h2 className="text-lg font-semibold text-slate-900">Research across 10-K filings</h2>
              <p className="mt-1 text-sm text-slate-600">
                Ask about one company or compare several across years. Every claim is cited and checked against the
                filing text; what cannot be found is reported, not guessed.
              </p>
              {coverage.length > 0 ? (
                <ul className="mt-4 grid gap-2 sm:grid-cols-3">
                  {coverage.map((c) => (
                    <li key={c.ticker} className="rounded-xl border border-slate-200 bg-slate-50 px-3 py-2">
                      <p className="text-sm font-semibold text-slate-800">{c.ticker}</p>
                      <p className="text-xs text-slate-500">
                        {c.company} · {c.range}
                      </p>
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="mt-4 text-sm text-slate-500">No filings available. Please ensure the backend is running.</p>
              )}
              {examples.length > 0 && (
                <div className="mt-5">
                  <p className="mb-2 text-sm font-medium text-brand-700">Try asking</p>
                  <div className="flex flex-wrap gap-2">
                    {examples.map((q) => (
                      <button
                        key={q}
                        type="button"
                        onClick={() => send(q)}
                        disabled={busy || rateLimited}
                        className="rounded-full border border-brand-100 bg-brand-50 px-3.5 py-1.5 text-left text-sm text-brand-700 hover:bg-brand-100 disabled:opacity-50"
                      >
                        {q}
                      </button>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {turns.map((turn, index) => {
            const isLast = index === turns.length - 1;
            return (
              <article key={turn.id} className="space-y-3">
                <div className="flex justify-end">
                  <p className="max-w-[85%] whitespace-pre-wrap rounded-2xl rounded-br-sm bg-brand-600 px-4 py-2.5 text-sm text-white">
                    {turn.question}
                  </p>
                </div>
                <StepTimeline items={turn.timeline} streaming={turn.streaming} latencyMs={turn.done?.latency_ms} />
                {turn.clarification && (
                  <ClarificationPrompt
                    request={turn.clarification}
                    answered={turn.clarificationChoice}
                    disabled={busy || !isLast}
                    onOption={(optionId, label) => resume(turn.id, { option_id: optionId }, label)}
                    onText={(text) => resume(turn.id, { text }, text)}
                  />
                )}
                {turn.answer && <ChatAnswerCard answer={turn.answer} />}
                {turn.error && (
                  <div className="flex items-start justify-between gap-3 rounded-xl border border-amber-200 bg-amber-50 p-3.5">
                    <p className="flex items-start gap-2 text-sm text-amber-800">
                      <AlertIcon className="mt-0.5 h-4 w-4 shrink-0" /> {turn.error}
                    </p>
                    {isLast && !busy && (
                      <button
                        type="button"
                        onClick={() => send(turn.question)}
                        className="shrink-0 text-sm font-medium text-amber-800 underline hover:text-amber-900"
                      >
                        Ask again
                      </button>
                    )}
                  </div>
                )}
                {turn.done && turn.answer && (
                  <p className="text-xs text-slate-400">
                    {(turn.done.latency_ms / 1000).toFixed(1)}s · {turn.done.llm_calls} LLM calls ·{' '}
                    {turn.done.tool_calls} searches · {turn.done.prompt_tokens} + {turn.done.completion_tokens} tokens
                  </p>
                )}
              </article>
            );
          })}
          <div ref={endRef} />

          <div className="sticky bottom-4 rounded-2xl border border-slate-200 bg-white p-4 shadow-lg">
            {tickers.length > 0 && (
              <div className="mb-3 flex flex-wrap items-center gap-1.5">
                <span className="mr-1 text-xs font-medium text-slate-500">Scope</span>
                {tickers.map((ticker) => (
                  <Chip
                    key={ticker}
                    active={chipTickers.includes(ticker)}
                    onClick={() => setChipTickers((v) => toggle(v, ticker, MAX_TICKERS))}
                  >
                    {ticker}
                  </Chip>
                ))}
                <span className="mx-1 h-4 w-px bg-slate-200" />
                {years.map((year) => (
                  <Chip
                    key={year}
                    active={chipYears.includes(year)}
                    onClick={() => setChipYears((v) => toggle(v, year, MAX_YEARS))}
                  >
                    FY{year}
                  </Chip>
                ))}
                {(chipTickers.length > 0 || chipYears.length > 0) && (
                  <button
                    type="button"
                    onClick={() => {
                      setChipTickers([]);
                      setChipYears([]);
                    }}
                    className="ml-1 text-xs text-slate-500 underline hover:text-slate-700"
                  >
                    Clear
                  </button>
                )}
              </div>
            )}
            <div className="relative">
              <textarea
                rows={2}
                value={input}
                maxLength={MAX_MESSAGE_CHARS}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    if (canSend) send(input);
                  }
                }}
                placeholder={threadId ? 'Ask a follow-up…' : 'Ask about the indexed filings…'}
                aria-label="Message"
                disabled={rateLimited}
                className="w-full resize-none rounded-xl border border-slate-300 bg-white py-3 pl-4 pr-24 text-slate-900 placeholder-slate-400 focus:border-brand-500 focus:outline-none focus:ring-2 focus:ring-brand-500/30 disabled:bg-slate-100"
              />
              <div className="absolute bottom-2.5 right-2.5 flex gap-2">
                {busy ? (
                  <button
                    type="button"
                    onClick={() => abortRef.current?.abort()}
                    className="rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm font-medium text-slate-700 hover:bg-slate-100"
                  >
                    Stop
                  </button>
                ) : (
                  <button
                    type="button"
                    onClick={() => send(input)}
                    disabled={!canSend}
                    aria-label="Send"
                    className="flex h-9 w-9 items-center justify-center rounded-lg bg-brand-600 text-white shadow-sm hover:bg-brand-700 disabled:cursor-not-allowed disabled:bg-slate-300"
                  >
                    <ArrowUp />
                  </button>
                )}
              </div>
            </div>
            <div className="mt-1 flex items-center justify-between text-xs text-slate-500">
              <span>
                {chipTickers.length || chipYears.length
                  ? 'Scope chips set the companies/years directly.'
                  : 'Enter to send, Shift+Enter for a new line.'}
              </span>
              <QuotaBadge quota={quota} />
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}
