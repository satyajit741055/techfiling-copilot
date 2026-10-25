import type { Metadata } from 'next';
import './globals.css';
import { AppShell } from '@/components/AppShell';
import { AuthProvider } from '@/components/AuthProvider';

export const metadata: Metadata = {
  title: 'TechFiling Copilot',
  description: 'SEC 10-K Document Q&A with RAG - Ask questions about Apple, Microsoft, and Tesla filings',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-slate-50 text-slate-900 antialiased">
        <AuthProvider>
          <AppShell>{children}</AppShell>
        </AuthProvider>
      </body>
    </html>
  );
}
