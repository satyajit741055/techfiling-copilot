import Link from 'next/link';
import { DocumentIcon, ShieldCheck, TableIcon } from '@/components/icons';

const FEATURES = [
  {
    icon: ShieldCheck,
    title: 'Grounded answers',
    body: 'Every answer is generated only from the filing text. When the evidence is not there, the copilot says so instead of guessing.',
  },
  {
    icon: DocumentIcon,
    title: 'Citations you can verify',
    body: 'Each claim links to the exact passage it came from, with the section path, so you can check the source in one click.',
  },
  {
    icon: TableIcon,
    title: 'Tables, not just text',
    body: 'Financial statements and footnote tables are retrieved and rendered as tables, with calculations shown step by step.',
  },
];

export function Landing() {
  return (
    <div className="mx-auto max-w-4xl py-8 text-center sm:py-14">
      <span className="inline-block rounded-full bg-brand-50 px-3 py-1 text-xs font-semibold uppercase tracking-wide text-brand-700">
        Research assistant for equity analysts
      </span>
      <h1 className="mt-5 text-4xl font-bold tracking-tight text-slate-900 sm:text-5xl">
        Ask SEC filings anything.
        <br className="hidden sm:block" /> Get answers with receipts.
      </h1>
      <p className="mx-auto mt-5 max-w-2xl text-lg leading-relaxed text-slate-600">
        TechFiling Copilot reads 10-K filings with hybrid retrieval and reranking, then answers your questions with
        citations back to the source passages.
      </p>
      <div className="mt-8 flex flex-col items-center justify-center gap-3 sm:flex-row">
        <Link
          href="/signup"
          className="w-full rounded-xl bg-brand-600 px-6 py-3 text-base font-semibold text-white shadow-sm hover:bg-brand-700 sm:w-auto"
        >
          Create a free account
        </Link>
        <Link
          href="/login"
          className="w-full rounded-xl border border-slate-300 bg-white px-6 py-3 text-base font-semibold text-slate-700 hover:bg-slate-50 sm:w-auto"
        >
          Sign in
        </Link>
      </div>

      <div className="mt-16 grid gap-4 text-left sm:grid-cols-3">
        {FEATURES.map(({ icon: Icon, title, body }) => (
          <div key={title} className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
            <span className="flex h-10 w-10 items-center justify-center rounded-lg bg-brand-50 text-brand-600">
              <Icon />
            </span>
            <h3 className="mt-4 font-semibold text-slate-900">{title}</h3>
            <p className="mt-1.5 text-sm leading-relaxed text-slate-600">{body}</p>
          </div>
        ))}
      </div>
    </div>
  );
}
