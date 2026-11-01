'use client';

import { ChatIcon, PlusIcon, TrashIcon } from '@/components/icons';

export interface ThreadItem {
  id: string;
  title: string;
  updatedAt: string;
}

interface ThreadSidebarProps {
  threads: ThreadItem[];
  activeId: string | null;
  /** 'server' = saved per user by the API; 'local' = this browser only (server history off). */
  source: 'server' | 'local' | 'loading';
  disabled?: boolean;
  onSelect: (id: string) => void;
  onNew: () => void;
  onDelete?: (id: string) => void;
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

export function ThreadSidebar({ threads, activeId, source, disabled, onSelect, onNew, onDelete }: ThreadSidebarProps) {
  return (
    <div className="rounded-2xl border border-slate-200 bg-white shadow-sm">
      <div className="flex items-center justify-between border-b border-slate-100 px-4 py-3">
        <h2 className="flex items-center gap-2 text-sm font-semibold text-slate-900">
          <ChatIcon className="h-4 w-4 text-slate-500" /> Conversations
        </h2>
        <button
          type="button"
          onClick={onNew}
          disabled={disabled}
          className="flex items-center gap-1 rounded-lg px-2 py-1 text-xs font-medium text-brand-700 hover:bg-brand-50 disabled:opacity-50"
        >
          <PlusIcon className="h-3.5 w-3.5" /> New
        </button>
      </div>

      {threads.length === 0 ? (
        <p className="px-4 py-6 text-center text-sm text-slate-500">
          {source === 'loading'
            ? 'Loading…'
            : source === 'server'
              ? 'Your conversations will appear here.'
              : 'Your conversations will appear here. They are saved in this browser.'}
        </p>
      ) : (
        <ul className="max-h-[60vh] divide-y divide-slate-100 overflow-y-auto">
          {threads.map((thread) => (
            <li
              key={thread.id}
              className={`group relative ${thread.id === activeId ? 'bg-brand-50' : 'hover:bg-slate-50'}`}
            >
              <button
                type="button"
                disabled={disabled}
                onClick={() => onSelect(thread.id)}
                className="w-full px-4 py-3 pr-10 text-left disabled:cursor-not-allowed"
              >
                <p className="line-clamp-2 text-sm text-slate-800">{thread.title}</p>
                <p className="mt-1 text-xs text-slate-500">{timeAgo(thread.updatedAt)}</p>
              </button>
              {onDelete && (
                <button
                  type="button"
                  onClick={() => onDelete(thread.id)}
                  aria-label="Delete conversation"
                  className="absolute right-2 top-3 rounded p-1 text-slate-400 opacity-100 hover:bg-slate-200 hover:text-red-600 lg:opacity-0 lg:group-hover:opacity-100"
                >
                  <TrashIcon />
                </button>
              )}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
