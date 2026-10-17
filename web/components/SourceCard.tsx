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
  // Format the chunk type for display
  const chunkTypeLabel = source.chunk_type === 'table' ? '📊 Table' : '📝 Text';
  const isTable = source.chunk_type === 'table';

  return (
    <div
      id={`source-${source.source_id}`}
      className="border border-gray-200 rounded-lg overflow-hidden bg-white"
    >
      {/* Header - always visible */}
      <button
        onClick={onToggle}
        className={`
          w-full p-4 flex items-center justify-between
          hover:bg-gray-50 transition-colors
          text-left
        `}
      >
        <div className="flex items-center gap-3 flex-wrap">
          {/* Source ID badge */}
          <span className="px-2.5 py-1 bg-blue-600 text-white rounded-md text-sm font-semibold">
            {source.source_id}
          </span>

          {/* Metadata */}
          <span className="text-gray-600 text-sm">
            {source.section ?? 'Unknown section'}
            {source.section_title && source.section_title !== source.section && (
              <> · {source.section_title}</>
            )}
          </span>
          <span className="text-gray-400">•</span>
          <span className="text-gray-500 text-sm">{chunkTypeLabel}</span>
        </div>

        {/* Expand/collapse indicator */}
        <span className="text-gray-400 ml-2 flex-shrink-0">
          {isExpanded ? '▼' : '▶'}
        </span>
      </button>

      {/* Expandable content */}
      {isExpanded && (
        <div className="border-t border-gray-200">
          {/* Breadcrumb - full path to the source */}
          {source.breadcrumb && (
            <div className="px-4 py-2 bg-blue-50 border-b border-blue-100">
              <div className="flex items-center gap-2 text-xs text-blue-700">
                <span className="font-medium">📍 Path:</span>
                <span className="font-mono">
                  {source.breadcrumb.split(' > ').map((part, i, arr) => (
                    <span key={i}>
                      <span className={i === arr.length - 1 ? 'font-semibold' : ''}>
                        {part}
                      </span>
                      {i < arr.length - 1 && (
                        <span className="mx-1 text-blue-400">›</span>
                      )}
                    </span>
                  ))}
                </span>
              </div>
            </div>
          )}

          {/* Metadata bar */}
          <div className="px-4 py-2 bg-gray-50 flex items-center gap-4 text-xs text-gray-500 flex-wrap">
            <span>
              <span className="font-medium">Score:</span>{' '}
              {source.score.toFixed(2)}
            </span>
            <span>
              <span className="font-medium">Doc:</span> {source.doc_id}
            </span>
          </div>

          {/* Source text */}
          <div className="p-4">
            <div className="source-text max-h-96 overflow-auto text-sm text-gray-800 leading-relaxed">
              <FormattedSourceText text={source.text} isTable={isTable} />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
