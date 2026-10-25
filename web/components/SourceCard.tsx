'use client';

import { useMemo } from 'react';
import type { Source } from '@/types';

interface SourceCardProps {
  source: Source;
  isExpanded: boolean;
  onToggle: () => void;
}

/**
 * Parse markdown table into structured data
 */
function parseMarkdownTable(text: string): { headers: string[]; rows: string[][] } | null {
  const lines = text.split('\n').filter(line => line.trim().startsWith('|'));
  
  if (lines.length < 2) return null;
  
  // Parse header row
  const headerLine = lines[0];
  const headers = headerLine
    .split('|')
    .map(cell => cell.trim())
    .filter(cell => cell.length > 0);
  
  // Skip separator row (|---|---|...)
  // Parse data rows
  const rows: string[][] = [];
  for (let i = 2; i < lines.length; i++) {
    const cells = lines[i]
      .split('|')
      .map(cell => cell.trim())
      .filter(cell => cell.length > 0);
    if (cells.length > 0) {
      rows.push(cells);
    }
  }
  
  return { headers, rows };
}

/**
 * Render source text with proper formatting for tables and paragraphs
 */
function FormattedSourceText({ text, isTable }: { text: string; isTable: boolean }) {
  const content = useMemo(() => {
    if (!isTable) {
      // For narrative text, just render paragraphs
      return (
        <div className="prose prose-sm max-w-none">
          {text.split('\n\n').map((para, i) => (
            <p key={i} className="mb-3 last:mb-0">
              {para}
            </p>
          ))}
        </div>
      );
    }
    
    // For tables, try to parse and render as HTML table
    // Split by double newline to separate title/description from table
    const parts = text.split('\n\n');
    const elements: React.ReactNode[] = [];
    
    parts.forEach((part, partIndex) => {
      const trimmed = part.trim();
      
      // Check if this part contains a markdown table
      if (trimmed.includes('|') && trimmed.includes('---')) {
        const tableData = parseMarkdownTable(trimmed);
        
        if (tableData && tableData.headers.length > 0) {
          elements.push(
            <div key={partIndex} className="overflow-x-auto my-3">
              <table className="min-w-full border-collapse border border-gray-300 text-sm">
                <thead>
                  <tr className="bg-gray-100">
                    {tableData.headers.map((header, i) => (
                      <th
                        key={i}
                        className="border border-gray-300 px-3 py-2 text-left font-semibold text-gray-700"
                      >
                        {header}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {tableData.rows.map((row, rowIndex) => (
                    <tr
                      key={rowIndex}
                      className={rowIndex % 2 === 0 ? 'bg-white' : 'bg-gray-50'}
                    >
                      {row.map((cell, cellIndex) => (
                        <td
                          key={cellIndex}
                          className={`border border-gray-300 px-3 py-2 ${
                            cellIndex === 0 ? 'font-medium text-gray-800' : 'text-gray-600'
                          } ${
                            // Right-align numbers
                            /^[\d,.\-$%()]+$/.test(cell.trim()) ? 'text-right' : ''
                          }`}
                        >
                          {cell}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          );
        } else {
          // Fallback: render as preformatted text
          elements.push(
            <pre key={partIndex} className="whitespace-pre-wrap text-sm my-2">
              {trimmed}
            </pre>
          );
        }
      } else if (trimmed.startsWith('**') && trimmed.endsWith('**')) {
        // Bold title
        elements.push(
          <h4 key={partIndex} className="font-semibold text-gray-800 my-2">
            {trimmed.replace(/\*\*/g, '')}
          </h4>
        );
      } else if (trimmed.length > 0) {
        // Regular paragraph
        elements.push(
          <p key={partIndex} className="text-gray-700 my-2">
            {trimmed}
          </p>
        );
      }
    });
    
    return <div>{elements}</div>;
  }, [text, isTable]);
  
  return <>{content}</>;
}

export function SourceCard({ source, isExpanded, onToggle }: SourceCardProps) {
  const isTable = source.chunk_type === 'table';

  return (
    <div
      id={`source-${source.source_id}`}
      className="overflow-hidden rounded-xl border border-slate-200 bg-white shadow-sm"
    >
      <button
        onClick={onToggle}
        aria-expanded={isExpanded}
        className="flex w-full items-center justify-between gap-3 p-4 text-left transition-colors hover:bg-slate-50"
      >
        <div className="flex min-w-0 flex-wrap items-center gap-x-3 gap-y-1">
          <span className="rounded-md bg-brand-600 px-2.5 py-0.5 text-sm font-semibold text-white">
            {source.source_id}
          </span>
          <span className="text-sm text-slate-700">
            {source.section ?? 'Unknown section'}
            {source.section_title && source.section_title !== source.section && <> · {source.section_title}</>}
          </span>
          <span className="rounded-full bg-slate-100 px-2 py-0.5 text-xs font-medium text-slate-600">
            {isTable ? 'Table' : 'Text'}
          </span>
        </div>
        <svg
          className={`h-4 w-4 shrink-0 text-slate-400 transition-transform ${isExpanded ? 'rotate-180' : ''}`}
          xmlns="http://www.w3.org/2000/svg"
          fill="none"
          viewBox="0 0 24 24"
          strokeWidth={2}
          stroke="currentColor"
          aria-hidden
        >
          <path strokeLinecap="round" strokeLinejoin="round" d="M6 9l6 6 6-6" />
        </svg>
      </button>

      {isExpanded && (
        <div className="border-t border-slate-200">
          {source.breadcrumb && (
            <div className="border-b border-brand-100 bg-brand-50 px-4 py-2 text-xs text-brand-700">
              <span className="font-medium">Path: </span>
              <span className="font-mono">
                {source.breadcrumb.split(' > ').map((part, i, arr) => (
                  <span key={i}>
                    <span className={i === arr.length - 1 ? 'font-semibold' : ''}>{part}</span>
                    {i < arr.length - 1 && <span className="mx-1 text-brand-500">›</span>}
                  </span>
                ))}
              </span>
            </div>
          )}

          <div className="flex flex-wrap items-center gap-4 bg-slate-50 px-4 py-2 text-xs text-slate-500">
            <span>
              <span className="font-medium">Relevance:</span> {source.score.toFixed(2)}
            </span>
            <span>
              <span className="font-medium">Document:</span> {source.doc_id}
            </span>
          </div>

          <div className="p-4">
            <div className="source-text max-h-96 overflow-auto text-sm leading-relaxed text-slate-800">
              <FormattedSourceText text={source.text} isTable={isTable} />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
