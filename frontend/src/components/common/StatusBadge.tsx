import React from 'react';

interface StatusBadgeProps {
  status: 'success' | 'warning' | 'danger' | 'info' | 'neutral';
  label: string;
  size?: 'sm' | 'md';
  pulse?: boolean;
}

export const StatusBadge: React.FC<StatusBadgeProps> = ({
  status,
  label,
  size = 'md',
  pulse = false,
}) => {
  const styles = {
    success: 'bg-emerald-50 text-emerald-700 border-emerald-200 dot-emerald-500',
    warning: 'bg-amber-50 text-amber-800 border-amber-200 dot-amber-500',
    danger: 'bg-rose-50 text-rose-700 border-rose-200 dot-rose-500',
    info: 'bg-blue-50 text-blue-700 border-blue-200 dot-blue-500',
    neutral: 'bg-slate-100 text-slate-700 border-slate-200 dot-slate-400',
  }[status];

  const dotColor = {
    success: 'bg-emerald-500',
    warning: 'bg-amber-500',
    danger: 'bg-rose-500',
    info: 'bg-blue-500',
    neutral: 'bg-slate-400',
  }[status];

  const sizeClass = size === 'sm' ? 'px-2 py-0.5 text-xs' : 'px-2.5 py-1 text-xs';

  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full border font-medium ${sizeClass} ${styles}`}
    >
      <span className="relative flex h-2 w-2">
        {pulse && (
          <span
            className={`absolute inline-flex h-full w-full animate-ping rounded-full opacity-75 ${dotColor}`}
          />
        )}
        <span className={`relative inline-flex h-2 w-2 rounded-full ${dotColor}`} />
      </span>
      {label}
    </span>
  );
};
