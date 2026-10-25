'use client';

import { useAuth } from '@/components/AuthProvider';
import { Landing } from '@/components/Landing';
import { PageSkeleton } from '@/components/Skeleton';
import { Workspace } from '@/components/Workspace';

export default function Home() {
  const { user, loading, authEnabled } = useAuth();

  if (loading) return <PageSkeleton />;
  // With Supabase configured, the workspace requires a signed-in user. Without it (local dev), it stays open.
  if (authEnabled && !user) return <Landing />;

  return <Workspace key={user?.id ?? 'local'} />;
}
