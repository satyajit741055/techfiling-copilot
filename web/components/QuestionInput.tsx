'use client';

import { useCallback } from 'react';
import { ArrowUp, Spinner } from '@/components/icons';

interface QuestionInputProps {
  value: string;
  onChange: (value: string) => void;
  onSubmit: (question: string) => void;
  disabled?: boolean;
  loading?: boolean;
  placeholder?: string;
}

const MIN_LENGTH = 3;

export function QuestionInput({
  value,
  onChange,
  onSubmit,
  disabled = false,
  loading = false,
  placeholder = 'Ask a question about the filing...',
}: QuestionInputProps) {
  const trimmed = value.trim();
  const canSubmit = trimmed.length >= MIN_LENGTH && !disabled && !loading;

  const submit = useCallback(() => {
    if (canSubmit) onSubmit(trimmed);
  }, [canSubmit, onSubmit, trimmed]);

  return (
    <div className="space-y-2">
      <label htmlFor="question-input" className="block text-sm font-medium text-slate-700">
        Your question
      </label>
      <div className="relative">
        <textarea
          id="question-input"
          rows={2}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault();
              submit();
            }
          }}
          placeholder={placeholder}
          disabled={disabled || loading}
          className="w-full resize-none rounded-xl border border-slate-300 bg-white py-3 pl-4 pr-14 text-slate-900 placeholder-slate-400 focus:border-brand-500 focus:outline-none focus:ring-2 focus:ring-brand-500/30 disabled:bg-slate-100 disabled:text-slate-500"
        />
        <button
          type="button"
          onClick={submit}
          disabled={!canSubmit}
          aria-label="Ask"
          className="absolute bottom-2.5 right-2.5 flex h-9 w-9 items-center justify-center rounded-lg bg-brand-600 text-white shadow-sm transition-colors hover:bg-brand-700 disabled:cursor-not-allowed disabled:bg-slate-300"
        >
          {loading ? <Spinner /> : <ArrowUp />}
        </button>
      </div>
      <p className="text-xs text-slate-500">
        {value.length > 0 && trimmed.length < MIN_LENGTH
          ? `Question must be at least ${MIN_LENGTH} characters.`
          : 'Press Enter to ask, Shift+Enter for a new line.'}
      </p>
    </div>
  );
}
