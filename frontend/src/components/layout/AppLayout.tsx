import React from 'react';
import { Outlet } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { arbiterApi } from '../../api/arbiterApi';
import { Sidebar } from './Sidebar';
import { Header } from './Header';
import { CurrencyProvider } from '../../context/CurrencyContext';

export const AppLayout: React.FC = () => {
  const { data: health, isError } = useQuery({
    queryKey: ['health'],
    queryFn: arbiterApi.getHealth,
    refetchInterval: 30000,
    retry: 1,
  });

  const isOnline = !isError && health?.status === 'healthy';
  const modelLoaded = Boolean(health?.model_loaded);

  return (
    <CurrencyProvider>
      <div className="flex min-h-screen bg-slate-50 font-sans text-slate-900">
        <Sidebar isOnline={isOnline} />

        <div className="flex flex-1 flex-col min-w-0">
          <Header isOnline={isOnline} modelLoaded={modelLoaded} />

          <main className="flex-1 p-6 md:p-8 max-w-6xl w-full mx-auto">
            <Outlet />
          </main>

          <footer className="border-t border-slate-200 bg-white px-6 py-3">
            <p className="text-[11px] text-slate-400 text-center">
              Risk assessments are decision-support outputs only and do not constitute loan approval, rejection or financial advice.
            </p>
          </footer>
        </div>
      </div>
    </CurrencyProvider>
  );
};
