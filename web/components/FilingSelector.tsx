'use client';

import type { Filing } from '@/types';

interface FilingSelectorProps {
  filings: Filing[];
  selected: string | null;
  onSelect: (doc_id: string) => void;
  disabled?: boolean;
}

export function FilingSelector({
  filings,
  selected,
  onSelect,
  disabled = false,
}: FilingSelectorProps) {
  return (
    <div className="space-y-2">
      <label
        htmlFor="filing-select"
        className="block text-sm font-medium text-gray-700"
      >
        Select Filing
      </label>
      <select
        id="filing-select"
        value={selected || ''}
        onChange={(e) => onSelect(e.target.value)}
        disabled={disabled}
        className={`
          w-full p-3 border border-gray-300 rounded-lg bg-white
          text-gray-900 text-base
          focus:ring-2 focus:ring-blue-500 focus:border-blue-500
          disabled:bg-gray-100 disabled:text-gray-500 disabled:cursor-not-allowed
          transition-colors
        `}
      >
        <option value="" disabled>
          Choose a company filing...
        </option>
        {filings.map((filing) => (
          <option key={filing.doc_id} value={filing.doc_id}>
            {filing.company} {filing.form} FY{filing.fiscal_year}
          </option>
        ))}
      </select>
      {selected && (
        <p className="text-xs text-gray-500">
          {filings.find((f) => f.doc_id === selected)?.period_end &&
            `Period ending: ${filings.find((f) => f.doc_id === selected)?.period_end}`}
        </p>
      )}
    </div>
  );
}
