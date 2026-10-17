'use client';

import { useState, useEffect, useCallback } from 'react';
import { FilingSelector } from '@/components/FilingSelector';
import { QuestionInput } from '@/components/QuestionInput';
import { AnswerCard } from '@/components/AnswerCard';
import { SourceCard } from '@/components/SourceCard';
import { getFilings, askQuestion, APIError } from '@/lib/api';
import type { Filing, QuestionResponse } from '@/types';

export default function Home() {
  // Data state
  const [filings, setFilings] = useState<Filing[]>([]);
  const [selectedFiling, setSelectedFiling] = useState<string | null>(null);
  const [result, setResult] = useState<QuestionResponse | null>(null);
  const [lastQuestion, setLastQuestion] = useState<string>('');

  // UI state
  const [loading, setLoading] = useState(false);
  const [initialLoading, setInitialLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [expandedSources, setExpandedSources] = useState<Set<string>>(new Set());

  // Fetch filings on mount
  useEffect(() => {
    const fetchFilings = async () => {
      try {
        const response = await getFilings();
        setFilings(response.filings);
        // Auto-select first filing if available
        if (response.filings.length > 0) {
          setSelectedFiling(response.filings[0].doc_id);
        }
      } catch (e) {
        if (e instanceof APIError) {
          setError(e.message);
        } else {
          setError('Failed to load filings');
        }
      } finally {
        setInitialLoading(false);
      }
    };

    fetchFilings();
  }, []);

  // Handle question submission
  const handleSubmit = useCallback(
    async (question: string) => {
      if (!selectedFiling) {
        setError('Please select a filing first');
        return;
      }

      // Clear previous result and show loading state
      setResult(null);
      setExpandedSources(new Set());
      setLoading(true);
      setError(null);
      setLastQuestion(question);

      try {
        const response = await askQuestion(question, selectedFiling);
        setResult(response);

        // Auto-expand first source for immediate verification
        if (response.sources.length > 0) {
          setExpandedSources(new Set([response.sources[0].source_id]));
        } else {
          setExpandedSources(new Set());
        }
      } catch (e) {
        if (e instanceof APIError) {
          setError(e.message);
        } else {
          setError('An unexpected error occurred');
        }
        setResult(null);
      } finally {
        setLoading(false);
      }
    },
    [selectedFiling]
  );

  // Handle citation click - scroll to source and expand it
  const handleCitationClick = useCallback((sourceId: string) => {
    // Expand the source
    setExpandedSources((prev) => new Set([...prev, sourceId]));

    // Scroll to the source card
    setTimeout(() => {
      const element = document.getElementById(`source-${sourceId}`);
      if (element) {
        element.scrollIntoView({ behavior: 'smooth', block: 'center' });
      }
    }, 100);
  }, []);

  // Toggle source expansion
  const toggleSource = useCallback((sourceId: string) => {
    setExpandedSources((prev) => {
      const next = new Set(prev);
      if (next.has(sourceId)) {
        next.delete(sourceId);
      } else {
        next.add(sourceId);
      }
      return next;
    });
  }, []);

  // Handle filing change - clear previous results
  const handleFilingChange = useCallback((doc_id: string) => {
    setSelectedFiling(doc_id);
    setResult(null);
    setError(null);
    setExpandedSources(new Set());
  }, []);

  // Initial loading state
  if (initialLoading) {
    return (
      <div className="flex items-center justify-center py-20">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4" />
          <p className="text-gray-600">Loading filings...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Error banner */}
      {error && (
        <div className="p-4 bg-red-50 border border-red-200 rounded-lg flex items-start justify-between">
          <div className="flex items-start gap-3">
            <span className="text-red-500 text-xl">⚠️</span>
            <div>
              <p className="text-red-700 font-medium">Error</p>
              <p className="text-red-600 text-sm">{error}</p>
            </div>
          </div>
          <button
            onClick={() => setError(null)}
            className="text-red-400 hover:text-red-600 text-xl leading-none"
          >
            ×
          </button>
        </div>
      )}

      {/* Filing selector and question input */}
      <div className="grid gap-4 md:grid-cols-3">
        <div className="md:col-span-1">
          <FilingSelector
            filings={filings}
            selected={selectedFiling}
            onSelect={handleFilingChange}
            disabled={loading}
          />
        </div>
        <div className="md:col-span-2">
          <QuestionInput
            onSubmit={handleSubmit}
            disabled={loading || !selectedFiling}
            placeholder={
              selectedFiling
                ? 'Ask a question about the filing...'
                : 'Select a filing first...'
            }
          />
        </div>
      </div>

      {/* Loading state */}
      {loading && (
        <div className="p-6 bg-white border border-gray-200 rounded-lg shadow-sm">
          <div className="flex items-center gap-4">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600" />
            <div>
              <p className="font-medium text-gray-900">Analyzing filing...</p>
              <p className="text-sm text-gray-500">
                Retrieving relevant passages and generating answer for: "{lastQuestion}"
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Suggested questions (only show when no result and not loading) */}
      {!result && !loading && selectedFiling && (
        <div className="p-4 bg-blue-50 border border-blue-100 rounded-lg">
          <p className="text-sm text-blue-800 font-medium mb-2">
            Try asking:
          </p>
          <div className="flex flex-wrap gap-2">
            {[
              'What was total net sales?',
              'What were the main risk factors?',
              'How much was spent on R&D?',
            ].map((q) => (
              <button
                key={q}
                onClick={() => handleSubmit(q)}
                className="px-3 py-1.5 bg-white border border-blue-200 rounded-full text-sm text-blue-700 hover:bg-blue-100 transition-colors"
              >
                {q}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Results section */}
      {result && (
        <div className="space-y-6">
          {/* Answer card */}
          <AnswerCard
            answer={result.answer}
            question={lastQuestion}
            onCitationClick={handleCitationClick}
          />

          {/* Sources section */}
          {result.sources.length > 0 && (
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <h2 className="text-lg font-semibold text-gray-900">
                  Sources ({result.sources.length})
                </h2>
                <button
                  onClick={() => {
                    if (expandedSources.size === result.sources.length) {
                      setExpandedSources(new Set());
                    } else {
                      setExpandedSources(
                        new Set(result.sources.map((s) => s.source_id))
                      );
                    }
                  }}
                  className="text-sm text-blue-600 hover:text-blue-800"
                >
                  {expandedSources.size === result.sources.length
                    ? 'Collapse all'
                    : 'Expand all'}
                </button>
              </div>

              <div className="space-y-2">
                {result.sources.map((source) => (
                  <SourceCard
                    key={source.source_id}
                    source={source}
                    isExpanded={expandedSources.has(source.source_id)}
                    onToggle={() => toggleSource(source.source_id)}
                  />
                ))}
              </div>
            </div>
          )}

          {/* Metadata footer */}
          <div className="flex items-center justify-between text-xs text-gray-500 pt-4 border-t border-gray-200">
            <div className="flex items-center gap-4">
              <span>⏱️ {result.latency_seconds.toFixed(2)}s</span>
              <span>🤖 {result.model}</span>
              <span>
                📊 {result.prompt_tokens} + {result.completion_tokens} tokens
              </span>
            </div>
            <span>{new Date(result.timestamp).toLocaleTimeString()}</span>
          </div>
        </div>
      )}

      {/* Empty state */}
      {!result && !loading && !error && filings.length === 0 && (
        <div className="text-center py-12">
          <p className="text-gray-500">
            No filings available. Please ensure the backend is running.
          </p>
        </div>
      )}
    </div>
  );
}
