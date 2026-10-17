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
