'use client';

import { AlertIcon, CheckIcon, Spinner } from '@/components/icons';
import type { TimelineItem } from '@/lib/chat';

function Row({ item }: { item: TimelineItem }) {
  return (
    <li className="flex items-start gap-2.5 py-1">
      <span className="mt-0.5 flex h-4 w-4 shrink-0 items-center justify-center">
        {item.status === 'running' && <Spinner className="h-3.5 w-3.5 text-brand-600" />}
        {item.status === 'done' && <CheckIcon className="h-4 w-4 text-emerald-600" />}
        {item.status === 'error' && <AlertIcon className="h-4 w-4 text-amber-600" />}
      </span>
      <span className={`text-sm ${item.node === 'tool' ? 'font-mono text-xs text-slate-600' : 'text-slate-700'}`}>
        {item.label}
        {item.detail && <span className="text-slate-400"> · {item.detail}</span>}
      </span>
    </li>
  );
}

/** Live research steps; collapses to a one-line summary once the turn is finished. */
export function StepTimeline({
  items,
  streaming,
  latencyMs,
}: {
  items: TimelineItem[];
  streaming: boolean;
  latencyMs?: number;
}) {
  if (items.length === 0) {
    return streaming ? (
      <div className="flex items-center gap-2 text-sm text-slate-500">
        <Spinner className="h-3.5 w-3.5 text-brand-600" /> Starting…
      </div>
    ) : null;
  }
  const list = (
    <ul className="border-l border-slate-200 pl-3" aria-live="polite">
      {items.map((item) => (
        <Row key={item.key} item={item} />
      ))}
    </ul>
  );
  if (streaming) return list;
  const searches = items.filter((item) => item.node === 'tool').length;
  return (
    <details className="group text-sm">
      <summary className="cursor-pointer select-none text-slate-500 hover:text-slate-700">
        {items.length} steps
        {searches > 0 && <> · {searches} searches</>}
        {latencyMs != null && <> · {(latencyMs / 1000).toFixed(1)}s</>}
      </summary>
      <div className="mt-2">{list}</div>
    </details>
  );
}
