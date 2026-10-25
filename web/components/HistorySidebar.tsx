'use client';

import { ClockIcon, TrashIcon } from '@/components/icons';
import type { HistoryEntry } from '@/lib/history';

interface HistorySidebarProps {
  entries: HistoryEntry[];
  activeId: string | null;
  onSelect: (entry: HistoryEntry) => void;
  onDelete: (id: string) => void;
  onClear: () => void;
}

function timeAgo(iso: string): string {
  const seconds = Math.max(0, Math.floor((Date.now() - new Date(iso).getTime()) / 1000));
  if (seconds < 60) return 'just now';
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  return `${Math.floor(hours / 24)}d ago`;
}

export function HistorySidebar({ entries, activeId, onSelect, onDelete, onClear }: HistorySidebarProps) {
  return (
    <div className="rounded-2xl border border-slate-200 bg-white shadow-sm">
      <div className="flex items-center justify-between border-b border-slate-100 px-4 py-3">
        <h2 className="flex items-center gap-2 text-sm font-semibold text-slate-900">
          <ClockIcon className="h-4 w-4 text-slate-500" /> History
        </h2>
        {entries.length > 0 && (
          <button onClick={onClear} className="text-xs font-medium text-slate-500 hover:text-red-600">
            Clear all
          </button>
        )}
      </div>

      {entries.length === 0 ? (
        <p className="px-4 py-6 text-center text-sm text-slate-500">
          Your questions will appear here. History is saved in this browser.
        </p>
      ) : (
        <ul className="max-h-[60vh] divide-y divide-slate-100 overflow-y-auto">
          {entries.map((entry) => (
            <li key={entry.id} className={`group relative ${entry.id === activeId ? 'bg-brand-50' : 'hover:bg-slate-50'}`}>
              <button onClick={() => onSelect(entry)} className="w-full px-4 py-3 pr-10 text-left">
                <p className="line-clamp-2 text-sm text-slate-800">{entry.question}</p>
                <p className="mt-1 text-xs text-slate-500">
                  {entry.doc_id.split('_')[0]} · {timeAgo(entry.createdAt)}
                </p>
              </button>
              <button
                onClick={() => onDelete(entry.id)}
                aria-label="Delete from history"
                className="absolute right-2 top-3 rounded p-1 text-slate-400 opacity-100 hover:bg-slate-200 hover:text-red-600 lg:opacity-0 lg:group-hover:opacity-100"
              >
                <TrashIcon />
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
