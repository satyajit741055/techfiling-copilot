'use client';

import { AlertIcon, CloseIcon } from '@/components/icons';

export function ErrorBanner({ message, onDismiss }: { message: string; onDismiss?: () => void }) {
  return (
    <div
      role="alert"
      className="flex items-start justify-between gap-3 rounded-xl border border-red-200 bg-red-50 p-4"
    >
      <div className="flex items-start gap-3">
        <AlertIcon className="mt-0.5 h-5 w-5 shrink-0 text-red-500" />
        <p className="text-sm leading-relaxed text-red-800">{message}</p>
      </div>
      {onDismiss && (
        <button
          onClick={onDismiss}
          aria-label="Dismiss"
          className="shrink-0 rounded p-1 text-red-400 hover:bg-red-100 hover:text-red-600"
        >
          <CloseIcon />
        </button>
      )}
    </div>
  );
}
