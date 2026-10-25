'use client';

import { useEffect, useState } from 'react';
import type { Quota } from '@/lib/api';

// The backend uses a 60 second moving window, so a reading older than that is back to the full limit.
const WINDOW_MS = 60_000;

export function QuotaBadge({ quota }: { quota: Quota | null }) {
  const [now, setNow] = useState(() => Date.now());

  useEffect(() => {
    if (!quota) return;
    const id = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(id);
  }, [quota]);

  if (!quota) return null;

  const stale = now - quota.receivedAt >= WINDOW_MS;
  const remaining = stale ? quota.limit : quota.remaining;
  const ratio = quota.limit > 0 ? remaining / quota.limit : 1;
  const tone =
    remaining === 0 ? 'bg-red-500' : ratio <= 0.2 ? 'bg-amber-500' : 'bg-emerald-500';

  return (
    <div className="flex items-center gap-2 text-xs text-slate-500" title="Questions allowed per minute">
      <div className="h-1.5 w-20 overflow-hidden rounded-full bg-slate-200">
        <div className={`h-full ${tone} transition-all`} style={{ width: `${Math.round(ratio * 100)}%` }} />
      </div>
      <span>
        {remaining} of {quota.limit} questions left this minute
      </span>
    </div>
  );
}
