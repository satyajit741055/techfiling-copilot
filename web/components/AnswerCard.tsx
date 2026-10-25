'use client';

import { AlertIcon } from '@/components/icons';
import type { Answer } from '@/types';

interface AnswerCardProps {
  answer: Answer;
  question: string;
  onCitationClick: (sourceId: string) => void;
}

function QuestionEcho({ question }: { question: string }) {
  return (
    <p className="mb-3 text-sm text-slate-500">
      <span className="font-semibold text-slate-600">Q</span> {question}
    </p>
  );
}

export function AnswerCard({ answer, question, onCitationClick }: AnswerCardProps) {
  if (answer.insufficient_context) {
    return (
      <div className="rounded-2xl border border-amber-200 bg-amber-50 p-5">
        <QuestionEcho question={question} />
        <div className="flex items-start gap-3">
          <AlertIcon className="mt-0.5 h-5 w-5 shrink-0 text-amber-600" />
          <div>
            <h3 className="font-semibold text-amber-900">Not enough evidence in this filing</h3>
            <p className="mt-1.5 leading-relaxed text-amber-800">{answer.explanation}</p>
            <p className="mt-3 text-sm text-amber-700">
              The filing does not contain enough information to answer this with confidence. Try rephrasing, or pick a
              different filing.
            </p>
          </div>
        </div>
      </div>
    );
  }

  const hasValidationIssues = !answer.is_valid && answer.validation_errors.length > 0;

  return (
    <div className="space-y-4">
      {hasValidationIssues && (
        <div className="rounded-2xl border border-red-200 bg-red-50 p-4">
          <h3 className="font-semibold text-red-800">Validation issues</h3>
          <ul className="mt-1.5 list-inside list-disc text-sm text-red-700">
            {answer.validation_errors.map((err, i) => (
              <li key={i}>{err}</li>
            ))}
          </ul>
        </div>
      )}

      <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
        <QuestionEcho question={question} />

        {answer.answer && (
          <div className="mb-3 text-2xl font-bold leading-snug tracking-tight text-slate-900">{answer.answer}</div>
        )}

        <p className="leading-relaxed text-slate-700">{answer.explanation}</p>

        {answer.calculation && (
          <div className="mt-5 rounded-xl border border-slate-200 bg-slate-50 p-3.5">
            <div className="mb-1 text-xs font-semibold uppercase tracking-wide text-slate-500">Calculation</div>
            <code className="whitespace-pre-wrap font-mono text-sm text-slate-800">{answer.calculation}</code>
          </div>
        )}

        {answer.citations.length > 0 && (
          <div className="mt-5 flex flex-wrap items-center gap-2 border-t border-slate-100 pt-4">
            <span className="text-sm text-slate-500">Sources</span>
            {answer.citations.map((citation) => (
              <button
                key={citation}
                onClick={() => onCitationClick(citation)}
                className="rounded-md bg-brand-50 px-2.5 py-1 text-sm font-medium text-brand-700 ring-1 ring-inset ring-brand-100 transition-colors hover:bg-brand-100 focus:outline-none focus:ring-2 focus:ring-brand-500"
              >
                [{citation}]
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
