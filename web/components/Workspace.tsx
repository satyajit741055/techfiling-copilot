'use client';

import { useCallback, useEffect, useState } from 'react';
import { AnswerCard } from '@/components/AnswerCard';
import { useAuth } from '@/components/AuthProvider';
import { ErrorBanner } from '@/components/ErrorBanner';
import { FilingSelector } from '@/components/FilingSelector';
import { HistorySidebar } from '@/components/HistorySidebar';
import { ClockIcon } from '@/components/icons';
import { QuestionInput } from '@/components/QuestionInput';
import { QuotaBadge } from '@/components/QuotaBadge';
import { AnswerSkeleton, PageSkeleton } from '@/components/Skeleton';
import { SourceCard } from '@/components/SourceCard';
import { askQuestion, friendlyError, getFilings, type Quota } from '@/lib/api';
import { loadHistory, newHistoryEntry, saveHistory, type HistoryEntry } from '@/lib/history';
import type { Filing, QuestionResponse } from '@/types';

const SUGGESTIONS = [
  'What was total net sales?',
  'What were the main risk factors?',
  'How much was spent on R&D?',
];

interface ActiveResult {
  question: string;
  response: QuestionResponse;
  entryId: string | null;
}

export function Workspace() {
  const { user, signOut } = useAuth();
  const userId = user?.id ?? null;

  const [filings, setFilings] = useState<Filing[]>([]);
  const [selectedFiling, setSelectedFiling] = useState<string | null>(null);
  const [question, setQuestion] = useState('');
  const [active, setActive] = useState<ActiveResult | null>(null);
  const [expandedSources, setExpandedSources] = useState<Set<string>>(new Set());

  const [loading, setLoading] = useState(false);
  const [initialLoading, setInitialLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [cooldown, setCooldown] = useState(0);
  const [quota, setQuota] = useState<Quota | null>(null);

  const [history, setHistory] = useState<HistoryEntry[]>(() => loadHistory(userId));
  const [historyOpen, setHistoryOpen] = useState(false);

  const handleError = useCallback(
    (e: unknown) => {
      const { message, retryAfter, status } = friendlyError(e);
      setError(message);
      if (status === 429 && retryAfter) setCooldown(retryAfter);
      if (status === 401) signOut();
    },
    [signOut]
  );

  useEffect(() => {
    getFilings()
      .then((response) => {
        setFilings(response.filings);
        if (response.filings.length > 0) setSelectedFiling(response.filings[0].doc_id);
      })
      .catch(handleError)
      .finally(() => setInitialLoading(false));
  }, [handleError]);

  useEffect(() => {
    if (cooldown <= 0) return;
    const id = setTimeout(() => setCooldown((c) => c - 1), 1000);
    return () => clearTimeout(id);
  }, [cooldown]);

  const persistHistory = useCallback(
    (next: HistoryEntry[]) => {
      setHistory(next);
      saveHistory(userId, next);
    },
    [userId]
  );

  const handleSubmit = useCallback(
    async (q: string) => {
      if (!selectedFiling) {
        setError('Please select a filing first.');
        return;
      }
      setActive(null);
      setExpandedSources(new Set());
      setLoading(true);
      setError(null);
      setQuestion(q);

      try {
        const { response, quota: nextQuota } = await askQuestion(q, selectedFiling);
        const entry = newHistoryEntry(q, response);
        setActive({ question: q, response, entryId: entry.id });
        if (nextQuota) setQuota(nextQuota);
        persistHistory([entry, ...history]);
        setExpandedSources(new Set(response.sources.length ? [response.sources[0].source_id] : []));
      } catch (e) {
        handleError(e);
      } finally {
        setLoading(false);
      }
    },
    [selectedFiling, history, persistHistory, handleError]
  );

  const handleCitationClick = useCallback((sourceId: string) => {
    setExpandedSources((prev) => new Set([...prev, sourceId]));
    setTimeout(() => {
      document.getElementById(`source-${sourceId}`)?.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }, 100);
  }, []);

  const toggleSource = useCallback((sourceId: string) => {
    setExpandedSources((prev) => {
      const next = new Set(prev);
      if (next.has(sourceId)) next.delete(sourceId);
      else next.add(sourceId);
      return next;
    });
  }, []);

  const handleFilingChange = useCallback((doc_id: string) => {
    setSelectedFiling(doc_id);
    setActive(null);
    setError(null);
    setExpandedSources(new Set());
  }, []);

  const handleSelectHistory = useCallback(
    (entry: HistoryEntry) => {
      setActive({ question: entry.question, response: entry.response, entryId: entry.id });
      setQuestion(entry.question);
      if (filings.some((f) => f.doc_id === entry.doc_id)) setSelectedFiling(entry.doc_id);
      setExpandedSources(new Set(entry.response.sources.length ? [entry.response.sources[0].source_id] : []));
      setError(null);
      setHistoryOpen(false);
      window.scrollTo({ top: 0, behavior: 'smooth' });
    },
    [filings]
  );

  if (initialLoading) return <PageSkeleton />;

  const rateLimited = cooldown > 0;
  const bannerMessage = rateLimited
    ? `You have reached the question limit. You can ask again in ${cooldown}s.`
    : error;

  return (
    <div className="space-y-4">
      <div className="lg:hidden">
        <button
          onClick={() => setHistoryOpen((v) => !v)}
          className="flex items-center gap-2 rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm font-medium text-slate-700"
        >
          <ClockIcon /> History ({history.length})
        </button>
      </div>

      <div className="grid gap-6 lg:grid-cols-[280px_minmax(0,1fr)]">
        <aside className={`${historyOpen ? 'block' : 'hidden'} lg:sticky lg:top-24 lg:block lg:self-start`}>
          <HistorySidebar
            entries={history}
            activeId={active?.entryId ?? null}
            onSelect={handleSelectHistory}
            onDelete={(id) => persistHistory(history.filter((h) => h.id !== id))}
            onClear={() => persistHistory([])}
          />
        </aside>

        <section className="min-w-0 space-y-6">
          {bannerMessage && (
            <ErrorBanner message={bannerMessage} onDismiss={rateLimited ? undefined : () => setError(null)} />
          )}

          <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
            <div className="grid gap-5 md:grid-cols-[260px_minmax(0,1fr)]">
              <FilingSelector
                filings={filings}
                selected={selectedFiling}
                onSelect={handleFilingChange}
                disabled={loading}
              />
              <div className="space-y-2">
                <QuestionInput
                  value={question}
                  onChange={setQuestion}
                  onSubmit={handleSubmit}
                  loading={loading}
                  disabled={!selectedFiling || rateLimited}
                  placeholder={selectedFiling ? 'Ask a question about the filing...' : 'Select a filing first...'}
                />
                <div className="flex justify-end">
                  <QuotaBadge quota={quota} />
                </div>
              </div>
            </div>
          </div>

          {loading && <AnswerSkeleton question={question} />}

          {!active && !loading && selectedFiling && (
            <div className="rounded-2xl border border-brand-100 bg-brand-50 p-5">
              <p className="mb-3 text-sm font-medium text-brand-700">Try asking</p>
              <div className="flex flex-wrap gap-2">
                {SUGGESTIONS.map((s) => (
                  <button
                    key={s}
                    onClick={() => !rateLimited && handleSubmit(s)}
                    disabled={rateLimited}
                    className="rounded-full border border-brand-100 bg-white px-3.5 py-1.5 text-sm text-brand-700 transition-colors hover:bg-brand-100 disabled:opacity-50"
                  >
                    {s}
                  </button>
                ))}
              </div>
            </div>
          )}

          {active && !loading && (
            <div className="space-y-6">
              <AnswerCard
                answer={active.response.answer}
                question={active.question}
                onCitationClick={handleCitationClick}
              />

              {active.response.sources.length > 0 && (
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <h2 className="text-lg font-semibold text-slate-900">
                      Sources ({active.response.sources.length})
                    </h2>
                    <button
                      onClick={() =>
                        setExpandedSources(
                          expandedSources.size === active.response.sources.length
                            ? new Set()
                            : new Set(active.response.sources.map((s) => s.source_id))
                        )
                      }
                      className="text-sm font-medium text-brand-600 hover:text-brand-700"
                    >
                      {expandedSources.size === active.response.sources.length ? 'Collapse all' : 'Expand all'}
                    </button>
                  </div>
                  <div className="space-y-2">
                    {active.response.sources.map((source) => (
                      <SourceCard
                        key={source.source_id}
                        source={source}
                        isExpanded={expandedSources.has(source.source_id)}
                        onToggle={() => toggleSource(source.source_id)}
                      />
                    ))}
                  </div>
                </div>
              )}

              <div className="flex flex-wrap items-center justify-between gap-2 border-t border-slate-200 pt-4 text-xs text-slate-500">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="rounded-full bg-slate-100 px-2.5 py-1">{active.response.latency_seconds.toFixed(2)}s</span>
                  <span className="rounded-full bg-slate-100 px-2.5 py-1">{active.response.model}</span>
                  <span className="rounded-full bg-slate-100 px-2.5 py-1">
                    {active.response.prompt_tokens} + {active.response.completion_tokens} tokens
                  </span>
                </div>
                <span>{new Date(active.response.timestamp).toLocaleString()}</span>
              </div>
            </div>
          )}

          {!active && !loading && !bannerMessage && filings.length === 0 && (
            <p className="py-12 text-center text-slate-500">No filings available. Please ensure the backend is running.</p>
          )}
        </section>
      </div>
    </div>
  );
}
