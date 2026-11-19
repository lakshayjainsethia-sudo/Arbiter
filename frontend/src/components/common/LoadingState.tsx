import React from 'react';
import { Loader2 } from 'lucide-react';

interface LoadingStateProps {
  message?: string;
  subMessage?: string;
}

export const LoadingState: React.FC<LoadingStateProps> = ({
  message = 'Loading data...',
  subMessage = 'Querying ARBITER research services',
}) => {
  return (
    <div className="flex flex-col items-center justify-center rounded-xl border border-slate-200 bg-white p-12 text-center shadow-xs">
      <div className="flex h-12 w-12 items-center justify-center rounded-full bg-blue-50 text-blue-600">
        <Loader2 className="h-6 w-6 animate-spin" />
      </div>
      <h3 className="mt-4 text-sm font-semibold text-slate-800">{message}</h3>
      <p className="mt-1 text-xs text-slate-500">{subMessage}</p>
    </div>
  );
};
