'use client';

import { useEffect, useState } from 'react';
import { ModeSwitch, type AppMode } from '@/components/AppShell';
import { useAuth } from '@/components/AuthProvider';
import { ChatWorkspace } from '@/components/ChatWorkspace';
import { Landing } from '@/components/Landing';
import { PageSkeleton } from '@/components/Skeleton';
import { Workspace } from '@/components/Workspace';

const MODE_KEY = 'tfc:mode';

export default function Home() {
  const { user, loading, authEnabled } = useAuth();
  const [mode, setMode] = useState<AppMode>('chat');

  useEffect(() => {
    // Restored after mount so the server render and the first client render match.
    const saved = window.localStorage.getItem(MODE_KEY);
    // eslint-disable-next-line react-hooks/set-state-in-effect
    if (saved === 'chat' || saved === 'filing') setMode(saved);
  }, []);

  const changeMode = (next: AppMode) => {
    setMode(next);
    window.localStorage.setItem(MODE_KEY, next);
  };

  if (loading) return <PageSkeleton />;
  // With Supabase configured, the workspace requires a signed-in user. Without it (local dev), it stays open.
  if (authEnabled && !user) return <Landing />;

  const key = user?.id ?? 'local';
  return (
    <div className="space-y-6">
      <ModeSwitch mode={mode} onChange={changeMode} />
      {mode === 'chat' ? <ChatWorkspace key={key} /> : <Workspace key={key} />}
    </div>
  );
}
