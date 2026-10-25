import type { Metadata } from 'next';
import './globals.css';
import AuthHeader from '@/components/AuthHeader';

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
      <body className="bg-gray-50 min-h-screen">
        <header className="bg-white border-b border-gray-200 sticky top-0 z-10">
          <div className="max-w-5xl mx-auto px-4 py-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <span className="text-2xl">📊</span>
                <h1 className="text-xl font-semibold text-gray-900">TechFiling Copilot</h1>
                <span className="text-sm text-gray-500 hidden sm:inline">
                  SEC 10-K Document Q&A
                </span>
              </div>
              <AuthHeader />
            </div>
          </div>
        </header>
        <main className="max-w-5xl mx-auto px-4 py-6">
          {children}
        </main>
        <footer className="border-t border-gray-200 mt-auto">
          <div className="max-w-5xl mx-auto px-4 py-4 text-center text-sm text-gray-500">
            Answers are grounded in SEC filings. Always verify critical information.
          </div>
        </footer>
      </body>
    </html>
  );
}
