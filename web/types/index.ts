// Types matching the FastAPI backend schemas (api/schemas/questions.py)

export interface Filing {
  doc_id: string;
  company: string;
  ticker: string;
  form: string;
  fiscal_year: number;
  period_end: string;
}

export interface FilingsResponse {
  filings: Filing[];
}

export interface Source {
  source_id: string;
  doc_id: string;
  chunk_type: string;
  section: string | null;
  section_title: string | null;
  table_id: string | null;
  score: number;
  elements: number[];
  text: string;
  breadcrumb: string | null;
  ticker?: string | null;
  fiscal_year?: number | null;
  period_end?: string | null;
}

export interface Answer {
  answer: string | null;
  explanation: string;
  citations: string[];
  calculation: string | null;
  insufficient_context: boolean;
  is_valid: boolean;
  validation_errors: string[];
}

export interface QuestionResponse {
  question: string;
  doc_id: string | null;
  answer: Answer;
  sources: Source[];
  model: string;
  context_k: number;
  prompt_tokens: number;
  completion_tokens: number;
  latency_seconds: number;
  timestamp: string;
}

export interface HealthResponse {
  status: string;
  environment: string;
  model: string;
  filings_loaded: number;
}

export interface ErrorResponse {
  detail: string;
}

// ---------------------------------------------------------------------------
// Research chat (api/schemas/chat.py, src/agent/schemas.py, src/agent/events.py)
// ---------------------------------------------------------------------------

export type CoverageStatus = 'pending' | 'found' | 'missing' | 'not_disclosed' | 'not_in_corpus';
export type ChatRoute = 'refuse' | 'onboard' | 'clarify' | 'simple_rag' | 'research';

export interface Claim {
  text: string;
  citation_ids: string[];
  excerpt: string;
  ticker: string;
  fiscal_year: number;
}

export interface CoverageCell {
  ticker: string;
  fiscal_year: number;
  status: CoverageStatus;
  detail: string | null;
  available_years: number[];
}

export interface Calculation {
  description: string;
  expression: string;
  result: number;
  inputs: { value: number; citation_id: string }[];
}

export interface ChatSource {
  source_id: string;
  chunk_id: string;
  ticker: string;
  fiscal_year: number;
  period_end: string | null;
  section: string | null;
  table_id: string | null;
  units: string | null;
  text: string;
}

export interface ChatUsage {
  prompt_tokens: number;
  completion_tokens: number;
  llm_calls: number;
  tool_calls: number;
  cost_usd: number;
}

export interface ChatAnswer {
  answer: string | null;
  claims: Claim[];
  coverage: CoverageCell[];
  assumptions: string[];
  calculations: Calculation[];
  sources: ChatSource[];
  is_valid: boolean;
  validation_errors: string[];
  dropped_claims: string[];
  route: ChatRoute;
  usage: ChatUsage;
}

export interface ClarificationOption {
  id: string;
  label: string;
}

export interface ClarificationRequest {
  question: string;
  options: ClarificationOption[];
  allow_free_text: boolean;
}

export interface ChatFilters {
  tickers: string[];
  fiscal_years: number[];
}

export interface ChatRequestBody {
  message?: string;
  thread_id?: string;
  filters?: ChatFilters;
  resume?: { option_id: string } | { text: string };
}

export interface StepEvent {
  node: string;
  status: 'started' | 'completed';
  label: string;
  detail?: string | null;
  elapsed_ms: number;
}

export interface ToolEvent {
  task_id: string;
  tool: string;
  args: Record<string, unknown>;
  n_results: number;
  ticker: string;
  fiscal_year: number | null;
  fiscal_years?: number[];
  error?: string | null;
}

export interface DoneEvent {
  latency_ms: number;
  prompt_tokens: number;
  completion_tokens: number;
  cost_usd: number;
  llm_calls: number;
  tool_calls: number;
}

export interface ChatErrorEvent {
  code: string;
  detail: string;
  correlation_id: string | null;
}

export type ChatEvent =
  | { event: 'thread'; data: { thread_id: string; turn_id: string | null } }
  | { event: 'step'; data: StepEvent }
  | { event: 'tool'; data: ToolEvent }
  | { event: 'clarification'; data: ClarificationRequest }
  | { event: 'answer'; data: ChatAnswer }
  | { event: 'error'; data: ChatErrorEvent }
  | { event: 'done'; data: DoneEvent };

export interface ThreadSummary {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
}

export interface StoredMessage {
  id: string;
  turn_id: string | null;
  question: string;
  status: 'answered' | 'clarification' | 'refused' | 'error';
  route: string | null;
  answer_markdown: string | null;
  answer: (ChatAnswer & { clarification?: never }) | { clarification: ClarificationRequest } | null;
  created_at: string;
}
