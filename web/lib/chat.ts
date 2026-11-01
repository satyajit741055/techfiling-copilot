import type {
  ChatAnswer,
  ChatEvent,
  ClarificationRequest,
  DoneEvent,
  Filing,
  StoredMessage,
} from '@/types';

export interface TimelineItem {
  key: string;
  node: string;
  label: string;
  detail?: string | null;
  status: 'running' | 'done' | 'error';
}

/** One question and everything streamed for it (a clarification and its resume share a turn). */
export interface ChatTurnView {
  id: string;
  turnId: string | null;
  question: string;
  timeline: TimelineItem[];
  answer?: ChatAnswer;
  clarification?: ClarificationRequest;
  clarificationChoice?: string;
  error?: string;
  done?: DoneEvent;
  streaming: boolean;
}

export function newTurn(question: string): ChatTurnView {
  return {
    id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    turnId: null,
    question,
    timeline: [],
    streaming: true,
  };
}

function settle(timeline: TimelineItem[], status: 'done' | 'error'): TimelineItem[] {
  return timeline.map((item) => (item.status === 'running' ? { ...item, status } : item));
}

function toolLabel(tool: string, ticker: string, years: number[]): string {
  const scope = years.length ? `${ticker} FY${years.join(', FY')}` : ticker;
  return `${tool.replace(/_/g, ' ')} · ${scope}`;
}

/** Fold one server event into the turn. Pure, so it is safe inside a state updater. */
export function applyEvent(turn: ChatTurnView, event: ChatEvent): ChatTurnView {
  switch (event.event) {
    case 'thread':
      return { ...turn, turnId: event.data.turn_id ?? turn.turnId };
    case 'step': {
      const { node, status, label, detail } = event.data;
      if (status === 'started') {
        const key = `${node}-${turn.timeline.length}`;
        return { ...turn, timeline: [...turn.timeline, { key, node, label, detail, status: 'running' }] };
      }
      const index = turn.timeline.findLastIndex((item) => item.node === node && item.status === 'running');
      if (index === -1) return turn;
      const timeline = [...turn.timeline];
      timeline[index] = { ...timeline[index], status: 'done', detail: timeline[index].detail ?? detail };
      return { ...turn, timeline };
    }
    case 'tool': {
      const { tool, ticker, fiscal_year, fiscal_years, n_results, error, task_id } = event.data;
      const years = fiscal_years?.length ? fiscal_years : fiscal_year != null ? [fiscal_year] : [];
      const item: TimelineItem = {
        key: `tool-${task_id}-${turn.timeline.length}`,
        node: 'tool',
        label: toolLabel(tool, ticker, years),
        detail: error ? 'failed' : `${n_results} result${n_results === 1 ? '' : 's'}`,
        status: error ? 'error' : 'done',
      };
      return { ...turn, timeline: [...turn.timeline, item] };
    }
    case 'clarification':
      return { ...turn, clarification: event.data, clarificationChoice: undefined };
    case 'answer':
      return { ...turn, answer: event.data, timeline: settle(turn.timeline, 'done') };
    case 'error':
      return { ...turn, error: event.data.detail, timeline: settle(turn.timeline, 'error') };
    case 'done':
      return { ...turn, done: event.data, streaming: false, timeline: settle(turn.timeline, 'done') };
  }
}

/** Saved rows → turn views; a clarification and the answer that resumed it are merged. */
export function turnsFromMessages(messages: StoredMessage[]): ChatTurnView[] {
  const turns: ChatTurnView[] = [];
  for (const message of messages) {
    const previous = turns[turns.length - 1];
    const sameTurn = previous && message.turn_id && previous.turnId === message.turn_id && previous.clarification;
    const turn: ChatTurnView = sameTurn
      ? { ...previous, clarificationChoice: previous.clarificationChoice ?? 'answered' }
      : { id: message.id, turnId: message.turn_id, question: message.question, timeline: [], streaming: false };
    const body = message.answer;
    if (message.status === 'clarification' && body && 'clarification' in body) {
      turn.clarification = body.clarification;
    } else if (message.status === 'error') {
      turn.error = 'This question could not be answered.';
    } else if (body && !('clarification' in body)) {
      turn.answer = body as ChatAnswer;
    }
    if (sameTurn) turns[turns.length - 1] = turn;
    else turns.push(turn);
  }
  return turns;
}

/** Example questions from the catalog (mirrors FilingCatalog.example_questions). */
export function exampleQuestions(filings: Filing[]): string[] {
  const companies = new Map<string, { company: string; years: number[] }>();
  for (const filing of filings) {
    const entry = companies.get(filing.ticker) ?? { company: filing.company, years: [] };
    entry.years.push(filing.fiscal_year);
    companies.set(filing.ticker, entry);
  }
  const tickers = [...companies.keys()].sort();
  if (!tickers.length) return [];
  const first = companies.get(tickers[0])!;
  const years = [...first.years].sort((a, b) => a - b);
  const examples = [`What was ${first.company}'s total revenue in FY${years[years.length - 1]}?`];
  if (tickers.length > 1) {
    examples.push(
      `Compare research and development spending for ${first.company} and ${companies.get(tickers[1])!.company}.`
    );
  }
  if (years.length > 1) {
    examples.push(
      `How did ${first.company}'s main risk factors change from FY${years[years.length - 2]} to FY${years[years.length - 1]}?`
    );
  }
  return examples.slice(0, 3);
}

// ---------------------------------------------------------------------------
// Browser fallback when server history is off (HISTORY_STORE=none or no owner)
// ---------------------------------------------------------------------------

export interface LocalThread {
  id: string;
  title: string;
  updatedAt: string;
  turns: ChatTurnView[];
}

const MAX_LOCAL_THREADS = 10;
const localKey = (userId: string | null) => `tfc:chat:${userId ?? 'local'}`;

export function loadLocalThreads(userId: string | null): LocalThread[] {
  try {
    const parsed = JSON.parse(window.localStorage.getItem(localKey(userId)) ?? '[]');
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

export function saveLocalThreads(userId: string | null, threads: LocalThread[]): void {
  let toSave = threads.slice(0, MAX_LOCAL_THREADS);
  // Answers include source text; if storage is full, keep progressively fewer threads
  for (;;) {
    try {
      window.localStorage.setItem(localKey(userId), JSON.stringify(toSave));
      return;
    } catch {
      if (toSave.length === 0) return;
      toSave = toSave.slice(0, Math.floor(toSave.length / 2));
    }
  }
}

export function threadTitle(question: string): string {
  const text = question.split(/\s+/).join(' ').trim();
  return text.length <= 80 ? text : `${text.slice(0, 79).trimEnd()}…`;
}
