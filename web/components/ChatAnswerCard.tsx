'use client';

import { Fragment, useCallback, useState, type ReactNode } from 'react';
import { AlertIcon, CheckIcon } from '@/components/icons';
import { SourceCard } from '@/components/SourceCard';
import type { ChatAnswer, CoverageCell, CoverageStatus } from '@/types';

const CITATION = /(\[S\d+(?:\s*,\s*S\d+)*\])/g;
const BOLD = /(\*\*[^*]+\*\*)/g;

const STATUS: Record<CoverageStatus, { label: string; className: string }> = {
  found: { label: 'Found', className: 'bg-emerald-50 text-emerald-700 ring-emerald-200' },
  not_disclosed: { label: 'Not disclosed', className: 'bg-amber-50 text-amber-700 ring-amber-200' },
  not_in_corpus: { label: 'Not in corpus', className: 'bg-slate-100 text-slate-500 ring-slate-200' },
  missing: { label: 'Missing', className: 'bg-slate-100 text-slate-500 ring-slate-200' },
  pending: { label: '—', className: 'bg-slate-100 text-slate-500 ring-slate-200' },
};

function Inline({ text, onCite }: { text: string; onCite: (id: string) => void }) {
  return (
    <>
      {text.split(CITATION).map((part, i) => {
        if (i % 2 === 1) {
          return part
            .slice(1, -1)
            .split(/\s*,\s*/)
            .map((id) => (
              <button
                key={`${i}-${id}`}
                type="button"
                onClick={() => onCite(id)}
                className="mx-0.5 rounded bg-brand-50 px-1 text-xs font-semibold text-brand-700 ring-1 ring-inset ring-brand-100 hover:bg-brand-100"
              >
                {id}
              </button>
            ));
        }
        return part.split(BOLD).map((piece, j) =>
          j % 2 === 1 ? <strong key={`${i}-${j}`}>{piece.slice(2, -2)}</strong> : <Fragment key={`${i}-${j}`}>{piece}</Fragment>
        );
      })}
    </>
  );
}

/** Paragraphs and "- " bullet lists with inline [S#] citation links. */
function AnswerText({ text, onCite }: { text: string; onCite: (id: string) => void }) {
  const blocks: ReactNode[] = [];
  let bullets: string[] = [];
  const flush = () => {
    if (!bullets.length) return;
    const items = bullets;
    blocks.push(
      <ul key={`ul-${blocks.length}`} className="list-disc space-y-1 pl-5">
        {items.map((item, i) => (
          <li key={i}>
            <Inline text={item} onCite={onCite} />
          </li>
        ))}
      </ul>
    );
    bullets = [];
  };
  for (const line of text.split('\n')) {
    const bullet = line.match(/^\s*(?:[-*•]|\d+\.)\s+(.*)$/);
    if (bullet) {
      bullets.push(bullet[1]);
      continue;
    }
    flush();
    if (line.trim()) {
      blocks.push(
        <p key={`p-${blocks.length}`}>
          <Inline text={line} onCite={onCite} />
        </p>
      );
    }
  }
  flush();
  return <div className="space-y-2 leading-relaxed text-slate-800">{blocks}</div>;
}

export function CoverageMatrix({ cells }: { cells: CoverageCell[] }) {
  if (cells.length === 0) return null;
  const tickers = [...new Set(cells.map((c) => c.ticker))];
  const years = [...new Set(cells.map((c) => c.fiscal_year))].sort((a, b) => a - b);
  const byKey = new Map(cells.map((c) => [`${c.ticker}-${c.fiscal_year}`, c]));
  return (
    <div className="overflow-x-auto">
      <table className="text-sm">
        <thead>
          <tr>
            <th className="pr-4 text-left text-xs font-semibold uppercase tracking-wide text-slate-500">Coverage</th>
            {years.map((year) => (
              <th key={year} className="px-2 pb-1 text-xs font-medium text-slate-500">
                FY{year}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {tickers.map((ticker) => (
            <tr key={ticker}>
              <td className="pr-4 font-medium text-slate-700">{ticker}</td>
              {years.map((year) => {
                const cell = byKey.get(`${ticker}-${year}`);
                if (!cell) return <td key={year} className="px-2 py-1 text-center text-slate-300">·</td>;
                const style = STATUS[cell.status];
                const title =
                  cell.status === 'not_in_corpus' && cell.available_years.length
                    ? `Indexed years: ${cell.available_years.map((y) => `FY${y}`).join(', ')}`
                    : cell.detail ?? undefined;
                return (
                  <td key={year} className="px-2 py-1 text-center">
                    <span title={title} className={`whitespace-nowrap rounded-full px-2 py-0.5 text-xs ring-1 ring-inset ${style.className}`}>
                      {style.label}
                    </span>
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function ChatAnswerCard({ answer }: { answer: ChatAnswer }) {
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  const [cardId] = useState(() => Math.random().toString(36).slice(2, 8));
  const grounded = answer.route === 'research' || answer.route === 'simple_rag';
  const abstained = grounded && !answer.answer;
  const issues = [...answer.dropped_claims.map((c) => `Removed: ${c}`), ...answer.validation_errors];

  const cite = useCallback(
    (id: string) => {
      setExpanded((prev) => new Set([...prev, id]));
      setTimeout(() => {
        document.getElementById(`source-${cardId}-${id}`)?.scrollIntoView({ behavior: 'smooth', block: 'center' });
      }, 100);
    },
    [cardId]
  );
  const toggle = (id: string) =>
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });

  return (
    <div className="space-y-4">
      <div
        className={`rounded-2xl border p-5 shadow-sm ${abstained ? 'border-amber-200 bg-amber-50' : 'border-slate-200 bg-white'}`}
      >
        {abstained ? (
          <div className="flex items-start gap-3">
            <AlertIcon className="mt-0.5 h-5 w-5 shrink-0 text-amber-600" />
            <div>
              <h3 className="font-semibold text-amber-900">Not enough evidence in the indexed filings</h3>
              <p className="mt-1 text-sm text-amber-800">
                Nothing in the retrieved passages supports an answer. The coverage table shows what was searched.
              </p>
            </div>
          </div>
        ) : (
          answer.answer && <AnswerText text={answer.answer} onCite={cite} />
        )}

        {grounded && (
          <div className="mt-4 flex flex-wrap items-center gap-2">
            {answer.is_valid && issues.length === 0 ? (
              <span className="inline-flex items-center gap-1 rounded-full bg-emerald-50 px-2.5 py-0.5 text-xs font-medium text-emerald-700 ring-1 ring-inset ring-emerald-200">
                <CheckIcon className="h-3.5 w-3.5" /> Citations verified
              </span>
            ) : (
              <details className="text-xs">
                <summary className="inline-flex cursor-pointer items-center gap-1 rounded-full bg-amber-50 px-2.5 py-0.5 font-medium text-amber-700 ring-1 ring-inset ring-amber-200">
                  <AlertIcon className="h-3.5 w-3.5" />
                  {answer.dropped_claims.length
                    ? `${answer.dropped_claims.length} unsupported claim(s) removed`
                    : 'Validation issues'}
                </summary>
                <ul className="mt-2 list-inside list-disc space-y-0.5 text-amber-800">
                  {issues.map((issue, i) => (
                    <li key={i}>{issue}</li>
                  ))}
                </ul>
              </details>
            )}
          </div>
        )}
      </div>

      {answer.assumptions.length > 0 && (
        <div className="rounded-xl border border-slate-200 bg-slate-50 p-3.5">
          <div className="mb-1 text-xs font-semibold uppercase tracking-wide text-slate-500">Assumptions</div>
          <ul className="list-inside list-disc space-y-0.5 text-sm text-slate-700">
            {answer.assumptions.map((assumption, i) => (
              <li key={i}>{assumption}</li>
            ))}
          </ul>
        </div>
      )}

      {grounded && answer.coverage.length > 0 && (
        <div className="rounded-xl border border-slate-200 bg-white p-3.5">
          <CoverageMatrix cells={answer.coverage} />
        </div>
      )}

      {answer.calculations.length > 0 && (
        <div className="rounded-xl border border-slate-200 bg-slate-50 p-3.5">
          <div className="mb-1 text-xs font-semibold uppercase tracking-wide text-slate-500">Calculations</div>
          <ul className="space-y-1.5 text-sm text-slate-700">
            {answer.calculations.map((calc, i) => (
              <li key={i}>
                {calc.description}:{' '}
                <code className="font-mono text-slate-900">
                  {calc.expression} = {calc.result.toLocaleString()}
                </code>
                {calc.inputs.length > 0 && (
                  <span className="text-slate-500">
                    {' '}
                    (inputs from{' '}
                    <Inline text={[...new Set(calc.inputs.map((input) => `[${input.citation_id}]`))].join(' ')} onCite={cite} />)
                  </span>
                )}
              </li>
            ))}
          </ul>
        </div>
      )}

      {answer.sources.length > 0 && (
        <div className="space-y-2">
          <h3 className="text-sm font-semibold text-slate-700">Sources ({answer.sources.length})</h3>
          {answer.sources.map((source) => (
            <SourceCard
              key={source.source_id}
              source={source}
              anchorId={`source-${cardId}-${source.source_id}`}
              isExpanded={expanded.has(source.source_id)}
              onToggle={() => toggle(source.source_id)}
            />
          ))}
        </div>
      )}
    </div>
  );
}
