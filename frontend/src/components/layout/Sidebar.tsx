import React from 'react';
import { NavLink } from 'react-router-dom';
import { ShieldCheck, LayoutDashboard, FileText } from 'lucide-react';

interface SidebarProps {
  isOnline: boolean;
}

const navItems = [
  { to: '/', label: 'Overview', icon: LayoutDashboard },
  { to: '/assessment', label: 'Loan Risk Assessment', icon: FileText },
];

export const Sidebar: React.FC<SidebarProps> = ({ isOnline }) => {
  return (
    <aside className="w-56 flex-shrink-0 bg-[#0f172a] text-slate-300 flex flex-col min-h-screen">
      {/* Brand */}
      <div className="px-5 py-6 border-b border-slate-800/60 flex items-center gap-3">
        <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-blue-600 text-white">
          <ShieldCheck className="h-5 w-5" />
        </div>
        <div>
          <h1 className="text-sm font-bold text-white tracking-wide">ARBITER</h1>
          <p className="text-[11px] text-slate-500">Loan Risk Assessment</p>
        </div>
      </div>

      {/* Nav */}
      <nav className="flex-1 px-3 py-4 space-y-1">
        {navItems.map((item) => {
          const Icon = item.icon;
          return (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.to === '/'}
              className={({ isActive }) =>
                `flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors ${
                  isActive
                    ? 'bg-blue-600 text-white'
                    : 'text-slate-400 hover:text-slate-100 hover:bg-slate-800/50'
                }`
              }
            >
              <Icon className="h-4 w-4 shrink-0" />
              <span>{item.label}</span>
            </NavLink>
          );
        })}
      </nav>

      {/* Connectivity dot */}
      <div className="px-5 py-4 border-t border-slate-800/60">
        <div className="flex items-center gap-2 text-xs text-slate-500">
          <span
            className={`h-1.5 w-1.5 rounded-full ${isOnline ? 'bg-emerald-400' : 'bg-rose-400'}`}
          />
          <span>{isOnline ? 'Connected' : 'Offline'}</span>
        </div>
      </div>
    </aside>
  );
};
