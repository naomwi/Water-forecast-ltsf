import Link from 'next/link';
import { BarChart3, MessageSquareText, MapPin, Database } from 'lucide-react';
import { Button } from '@/components/ui/button';

export function Topbar() {
  return (
    <header className="sticky top-0 z-50 w-full border-b bg-white text-slate-900 shadow-sm">
      <div className="flex h-16 items-center px-6">
        <div className="flex items-center gap-2 font-bold text-xl tracking-tight text-[#164e63]">
          <MapPin className="h-6 w-6" />
          <span>HydroPred</span>
        </div>

        <nav className="ml-auto flex items-center space-x-2">
          <Link href="/">
            <Button variant="ghost" className="text-slate-600 hover:text-blue-600 hover:bg-blue-50">
              <MessageSquareText className="mr-2 h-4 w-4" />
              Chat Assistant
            </Button>
          </Link>
          <Link href="/results">
            <Button variant="ghost" className="text-slate-600 hover:text-blue-600 hover:bg-blue-50">
              <BarChart3 className="mr-2 h-4 w-4" />
              Model Results
            </Button>
          </Link>
          <Link href="/data">
            <Button variant="ghost" className="text-slate-600 hover:text-blue-600 hover:bg-blue-50">
              <Database className="mr-2 h-4 w-4" />
              Data Explorer
            </Button>
          </Link>
        </nav>
      </div>
    </header>
  );
}
