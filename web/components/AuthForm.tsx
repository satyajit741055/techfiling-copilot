'use client';

import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { useEffect, useState } from 'react';
import { useAuth } from '@/components/AuthProvider';
import { Spinner } from '@/components/icons';
import { signIn, signUp } from '@/lib/supabase';

const MIN_PASSWORD = 8;

const inputClass =
  'w-full rounded-lg border border-slate-300 bg-white px-3.5 py-2.5 text-slate-900 placeholder-slate-400 ' +
  'focus:border-brand-500 focus:outline-none focus:ring-2 focus:ring-brand-500/30 disabled:bg-slate-100';

export function AuthForm({ mode }: { mode: 'login' | 'signup' }) {
  const router = useRouter();
  const { user, loading: authLoading, authEnabled } = useAuth();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirm, setConfirm] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [checkEmail, setCheckEmail] = useState(false);

  const isSignup = mode === 'signup';

  useEffect(() => {
    if (user) router.replace('/');
  }, [user, router]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);

    if (isSignup) {
      if (password.length < MIN_PASSWORD) return setError(`Password must be at least ${MIN_PASSWORD} characters.`);
      if (password !== confirm) return setError('Passwords do not match.');
    }

    setSubmitting(true);
    try {
      if (isSignup) {
        const data = await signUp(email, password);
        // No session means the project requires email confirmation first
        if (!data.session) setCheckEmail(true);
      } else {
        await signIn(email, password);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Something went wrong. Please try again.');
    } finally {
      setSubmitting(false);
    }
  };

  if (!authEnabled && !authLoading) {
    return (
      <Card title="Authentication is not configured">
        <p className="text-sm text-slate-600">
          Set <code className="rounded bg-slate-100 px-1">NEXT_PUBLIC_SUPABASE_URL</code> and{' '}
          <code className="rounded bg-slate-100 px-1">NEXT_PUBLIC_SUPABASE_ANON_KEY</code>, then rebuild the frontend.
        </p>
      </Card>
    );
  }

  if (checkEmail) {
    return (
      <Card title="Check your email">
        <p className="text-sm leading-relaxed text-slate-600">
          We sent a confirmation link to <span className="font-medium text-slate-900">{email}</span>. Open it to
          activate your account, then sign in.
        </p>
        <Link
          href="/login"
          className="mt-5 inline-block rounded-lg bg-brand-600 px-4 py-2.5 text-sm font-medium text-white hover:bg-brand-700"
        >
          Go to sign in
        </Link>
      </Card>
    );
  }

  return (
    <Card
      title={isSignup ? 'Create your account' : 'Welcome back'}
      subtitle={isSignup ? 'Start asking questions about SEC filings.' : 'Sign in to continue to TechFiling Copilot.'}
    >
      {error && (
        <div role="alert" className="mb-4 rounded-lg border border-red-200 bg-red-50 px-3.5 py-2.5 text-sm text-red-700">
          {error}
        </div>
      )}
      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label htmlFor="email" className="mb-1.5 block text-sm font-medium text-slate-700">
            Email
          </label>
          <input
            id="email"
            type="email"
            autoComplete="email"
            required
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            disabled={submitting}
            placeholder="you@company.com"
            className={inputClass}
          />
        </div>
        <div>
          <label htmlFor="password" className="mb-1.5 block text-sm font-medium text-slate-700">
            Password
          </label>
          <input
            id="password"
            type="password"
            autoComplete={isSignup ? 'new-password' : 'current-password'}
            required
            minLength={isSignup ? MIN_PASSWORD : undefined}
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            disabled={submitting}
            className={inputClass}
          />
          {isSignup && <p className="mt-1 text-xs text-slate-500">At least {MIN_PASSWORD} characters.</p>}
        </div>
        {isSignup && (
          <div>
            <label htmlFor="confirm" className="mb-1.5 block text-sm font-medium text-slate-700">
              Confirm password
            </label>
            <input
              id="confirm"
              type="password"
              autoComplete="new-password"
              required
              value={confirm}
              onChange={(e) => setConfirm(e.target.value)}
              disabled={submitting}
              className={inputClass}
            />
          </div>
        )}
        <button
          type="submit"
          disabled={submitting}
          className="flex w-full items-center justify-center gap-2 rounded-lg bg-brand-600 px-4 py-2.5 font-medium text-white shadow-sm hover:bg-brand-700 disabled:opacity-60"
        >
          {submitting && <Spinner />}
          {isSignup ? 'Create account' : 'Sign in'}
        </button>
      </form>
      <p className="mt-6 text-center text-sm text-slate-600">
        {isSignup ? 'Already have an account?' : "Don't have an account?"}{' '}
        <Link href={isSignup ? '/login' : '/signup'} className="font-medium text-brand-600 hover:text-brand-700">
          {isSignup ? 'Sign in' : 'Sign up'}
        </Link>
      </p>
    </Card>
  );
}

function Card({ title, subtitle, children }: { title: string; subtitle?: string; children: React.ReactNode }) {
  return (
    <div className="mx-auto w-full max-w-md py-6">
      <div className="rounded-2xl border border-slate-200 bg-white p-7 shadow-sm">
        <h1 className="text-2xl font-bold tracking-tight text-slate-900">{title}</h1>
        {subtitle && <p className="mt-1.5 mb-6 text-sm text-slate-600">{subtitle}</p>}
        {!subtitle && <div className="mt-4" />}
        {children}
      </div>
    </div>
  );
}
