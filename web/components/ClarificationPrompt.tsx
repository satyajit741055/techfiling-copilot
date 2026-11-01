'use client';

import { useState } from 'react';
import type { ClarificationRequest } from '@/types';

interface ClarificationPromptProps {
  request: ClarificationRequest;
  /** The answer already given (the prompt is then read-only). */
  answered?: string;
  disabled?: boolean;
  onOption: (optionId: string, label: string) => void;
  onText: (text: string) => void;
}

export function ClarificationPrompt({ request, answered, disabled, onOption, onText }: ClarificationPromptProps) {
  const [text, setText] = useState('');
  const locked = disabled || answered !== undefined;
  const trimmed = text.trim();

  return (
    <div className="rounded-2xl border border-brand-100 bg-brand-50 p-4">
      <p className="text-sm font-medium text-brand-800">{request.question}</p>
      <div className="mt-3 flex flex-wrap gap-2">
        {request.options.map((option) => (
          <button
            key={option.id}
            type="button"
            disabled={locked}
            onClick={() => onOption(option.id, option.label)}
            className={`rounded-full border px-3.5 py-1.5 text-sm transition-colors disabled:cursor-not-allowed ${
              answered === option.label
                ? 'border-brand-600 bg-brand-600 text-white'
                : 'border-brand-100 bg-white text-brand-700 hover:bg-brand-100 disabled:opacity-60'
            }`}
          >
            {option.label}
          </button>
        ))}
      </div>
      {request.allow_free_text && !locked && (
        <form
          className="mt-3 flex gap-2"
          onSubmit={(e) => {
            e.preventDefault();
            if (trimmed) onText(trimmed);
          }}
        >
          <input
            value={text}
            onChange={(e) => setText(e.target.value)}
            maxLength={500}
            placeholder="Or type your answer…"
            aria-label="Clarification answer"
            className="min-w-0 flex-1 rounded-lg border border-slate-300 bg-white px-3 py-1.5 text-sm focus:border-brand-500 focus:outline-none focus:ring-2 focus:ring-brand-500/30"
          />
          <button
            type="submit"
            disabled={!trimmed}
            className="rounded-lg bg-brand-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-brand-700 disabled:bg-slate-300"
          >
            Send
          </button>
        </form>
      )}
      {answered !== undefined && !request.options.some((o) => o.label === answered) && (
        <p className="mt-2 text-xs text-brand-700">Answered: {answered}</p>
      )}
    </div>
  );
}
