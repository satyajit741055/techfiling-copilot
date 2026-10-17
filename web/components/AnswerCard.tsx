'use client';

import type { Answer } from '@/types';

interface AnswerCardProps {
  answer: Answer;
  question: string;
  onCitationClick: (sourceId: string) => void;
}

export function AnswerCard({ answer, question, onCitationClick }: AnswerCardProps) {
  // Insufficient context - show warning
  if (answer.insufficient_context) {
    return (
      <div className="p-5 bg-amber-50 border border-amber-200 rounded-lg">
        <div className="flex items-start gap-3">
          <span className="text-2xl">⚠️</span>
          <div className="flex-1">
            <h3 className="font-semibold text-amber-800 mb-2">
              Insufficient Evidence
            </h3>
            <p className="text-amber-700">{answer.explanation}</p>
            <p className="text-sm text-amber-600 mt-3">
              The filing does not contain enough information to answer this question
              with confidence.
            </p>
          </div>
        </div>
      </div>
    );
  }

  // Validation errors - show warning but still display answer
  if (!answer.is_valid && answer.validation_errors.length > 0) {
    return (
      <div className="space-y-4">
        <div className="p-4 bg-red-50 border border-red-200 rounded-lg">
          <h3 className="font-semibold text-red-800 mb-2">Validation Issues</h3>
          <ul className="list-disc list-inside text-red-700 text-sm">
            {answer.validation_errors.map((err, i) => (
              <li key={i}>{err}</li>
            ))}
          </ul>
        </div>
        {answer.answer && (
          <div className="p-5 bg-white border border-gray-200 rounded-lg shadow-sm">
            <div className="text-xl font-bold text-gray-900 mb-3">
              {answer.answer}
            </div>
            <p className="text-gray-700">{answer.explanation}</p>
          </div>
        )}
      </div>
    );
  }

  // Normal answer
  return (
    <div className="p-5 bg-white border border-gray-200 rounded-lg shadow-sm">
      {/* Question echo */}
      <div className="text-sm text-gray-500 mb-3">
        <span className="font-medium">Q:</span> {question}
      </div>

      {/* Main answer */}
      {answer.answer && (
        <div className="text-2xl font-bold text-gray-900 mb-3">
          {answer.answer}
        </div>
      )}

      {/* Explanation */}
      <p className="text-gray-700 leading-relaxed">{answer.explanation}</p>

      {/* Calculation (if present) */}
      {answer.calculation && (
        <div className="mt-4 p-3 bg-gray-50 rounded-lg border border-gray-100">
          <div className="text-xs text-gray-500 mb-1 font-medium">Calculation</div>
          <code className="text-sm text-gray-800 font-mono whitespace-pre-wrap">
            {answer.calculation}
          </code>
        </div>
      )}

      {/* Citations */}
      {answer.citations.length > 0 && (
        <div className="mt-4 pt-4 border-t border-gray-100">
          <div className="flex items-center gap-2 flex-wrap">
            <span className="text-sm text-gray-500">Sources:</span>
            {answer.citations.map((citation) => (
              <button
                key={citation}
                onClick={() => onCitationClick(citation)}
                className={`
                  px-2.5 py-1 rounded-md text-sm font-medium
                  bg-blue-100 text-blue-700
                  hover:bg-blue-200
                  focus:ring-2 focus:ring-blue-500 focus:ring-offset-1
                  transition-colors
                `}
              >
                [{citation}]
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
