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
  const current = filings.find((f) => f.doc_id === selected);

  return (
    <div className="space-y-2">
      <label htmlFor="filing-select" className="block text-sm font-medium text-slate-700">
        Filing
      </label>
      <select
        id="filing-select"
        value={selected || ''}
        onChange={(e) => onSelect(e.target.value)}
        disabled={disabled}
        className="w-full rounded-xl border border-slate-300 bg-white px-3.5 py-3 text-slate-900 focus:border-brand-500 focus:outline-none focus:ring-2 focus:ring-brand-500/30 disabled:cursor-not-allowed disabled:bg-slate-100 disabled:text-slate-500"
      >
        <option value="" disabled>
          Choose a company filing...
        </option>
        {filings.map((filing) => (
          <option key={filing.doc_id} value={filing.doc_id}>
            {filing.company} ({filing.ticker}) {filing.form} FY{filing.fiscal_year}
          </option>
        ))}
      </select>
      <p className="text-xs text-slate-500">
        {current?.period_end ? `Period ending ${current.period_end}` : 'Select the filing to search.'}
      </p>
    </div>
  );
}
