import type { QuestionResponse } from '@/types';

export interface HistoryEntry {
  id: string;
  question: string;
  doc_id: string;
  response: QuestionResponse;
  createdAt: string;
}

const MAX_ENTRIES = 20;

// Browser-local storage, kept per user. Swap these three functions to move to a server-side store.
const storageKey = (userId: string | null) => `tfc:history:${userId ?? 'local'}`;

export function loadHistory(userId: string | null): HistoryEntry[] {
  try {
    const raw = window.localStorage.getItem(storageKey(userId));
    const parsed = raw ? JSON.parse(raw) : [];
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

export function saveHistory(userId: string | null, entries: HistoryEntry[]): void {
  let toSave = entries.slice(0, MAX_ENTRIES);
  // Responses include source text; if storage is full, keep progressively fewer entries
  while (toSave.length >= 0) {
    try {
      window.localStorage.setItem(storageKey(userId), JSON.stringify(toSave));
      return;
    } catch {
      if (toSave.length === 0) return;
      toSave = toSave.slice(0, Math.floor(toSave.length / 2));
    }
  }
}

export function newHistoryEntry(question: string, response: QuestionResponse): HistoryEntry {
  return {
    id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    question,
    doc_id: response.doc_id ?? '',
    response,
    createdAt: new Date().toISOString(),
  };
}
