import React from 'react';
import { ShieldCheck } from 'lucide-react';

interface HeaderProps {
  isOnline: boolean;
  modelLoaded: boolean;
}

export const Header: React.FC<HeaderProps> = ({ isOnline }) => {
  return (
    <header className="sticky top-0 z-20 flex h-14 w-full items-center justify-between border-b border-slate-200 bg-white px-6">
      <div className="flex items-center gap-2">
        <ShieldCheck className="h-4 w-4 text-blue-600" />
        <span className="text-sm font-semibold text-slate-800">Loan Risk Assessment Platform</span>
      </div>

      <div className="flex items-center gap-3">
        {!isOnline && (
          <span className="text-xs text-amber-700 bg-amber-50 border border-amber-200 px-2.5 py-1 rounded-md">
            Service unavailable — predictions disabled
          </span>
        )}
      </div>
    </header>
  );
};
