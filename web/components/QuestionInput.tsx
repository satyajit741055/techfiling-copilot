'use client';

import { useState, useCallback } from 'react';

interface QuestionInputProps {
  onSubmit: (question: string) => void;
  disabled?: boolean;
  placeholder?: string;
}

export function QuestionInput({
  onSubmit,
  disabled = false,
  placeholder = 'Ask a question about the filing...',
}: QuestionInputProps) {
  const [question, setQuestion] = useState('');

  const handleSubmit = useCallback(
    (e: React.FormEvent) => {
      e.preventDefault();
      const trimmed = question.trim();
      if (trimmed && !disabled) {
        onSubmit(trimmed);
        // Don't clear - let user see what they asked
      }
    },
    [question, disabled, onSubmit]
  );

  const isValid = question.trim().length >= 3;

  return (
    <form onSubmit={handleSubmit} className="space-y-2">
      <label
        htmlFor="question-input"
        className="block text-sm font-medium text-gray-700"
      >
        Your Question
      </label>
      <div className="flex gap-2">
        <input
          id="question-input"
          type="text"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder={placeholder}
          disabled={disabled}
          className={`
            flex-1 p-3 border border-gray-300 rounded-lg
            text-gray-900 text-base placeholder-gray-400
            focus:ring-2 focus:ring-blue-500 focus:border-blue-500
            disabled:bg-gray-100 disabled:text-gray-500 disabled:cursor-not-allowed
            transition-colors
          `}
        />
        <button
          type="submit"
          disabled={disabled || !isValid}
          className={`
            px-6 py-3 rounded-lg font-medium
            bg-blue-600 text-white
            hover:bg-blue-700
            focus:ring-2 focus:ring-blue-500 focus:ring-offset-2
            disabled:bg-gray-300 disabled:text-gray-500 disabled:cursor-not-allowed
            transition-colors
          `}
        >
          {disabled ? (
            <span className="flex items-center gap-2">
              <svg
                className="animate-spin h-4 w-4"
                xmlns="http://www.w3.org/2000/svg"
                fill="none"
                viewBox="0 0 24 24"
              >
                <circle
                  className="opacity-25"
                  cx="12"
                  cy="12"
                  r="10"
                  stroke="currentColor"
                  strokeWidth="4"
                />
                <path
                  className="opacity-75"
                  fill="currentColor"
                  d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                />
              </svg>
              <span>Thinking...</span>
            </span>
          ) : (
            'Ask'
          )}
        </button>
      </div>
      {question.length > 0 && !isValid && (
        <p className="text-xs text-amber-600">
          Question must be at least 3 characters
        </p>
      )}
    </form>
  );
}
