import React from 'react';
import { Link } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { arbiterApi } from '../api/arbiterApi';
import { ArrowRight, ShieldCheck, Globe, Flag, Info } from 'lucide-react';

export const OverviewPage: React.FC = () => {
  const { data: health } = useQuery({
    queryKey: ['health'],
    queryFn: arbiterApi.getHealth,
    retry: 1,
  });

  const isOnline = health?.status === 'healthy';

  return (
    <div className="max-w-3xl mx-auto space-y-10 py-6">
      {/* Hero */}
      <div className="space-y-3">
        <div className="flex items-center gap-2 text-blue-600">
          <ShieldCheck className="h-5 w-5" />
          <span className="text-sm font-semibold tracking-wide uppercase">ARBITER</span>
        </div>
        <h1 className="text-3xl font-bold text-slate-900 tracking-tight">
          Loan Risk Assessment
        </h1>
        <p className="text-base text-slate-500 leading-relaxed max-w-xl">
          Assess borrower risk using applicant and financial information.
          Select an assessment type to begin.
        </p>
      </div>

      {/* Assessment type cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-5">
        <Link
          to="/assessment?type=india"
          className="group relative rounded-xl border border-slate-200 bg-white p-6 shadow-sm hover:border-blue-300 hover:shadow-md transition-all"
        >
          <div className="flex items-start justify-between mb-4">
            <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-blue-50 text-blue-600">
              <Flag className="h-5 w-5" />
            </div>
            <ArrowRight className="h-4 w-4 text-slate-300 group-hover:text-blue-500 transition-colors" />
          </div>
          <h2 className="text-base font-semibold text-slate-900 mb-1">Indian Loan Assessment</h2>
          <p className="text-sm text-slate-500 leading-relaxed">
            Evaluate Indian applicants using CIBIL score, INR income and India-specific loan criteria.
          </p>
        </Link>

        <Link
          to="/assessment?type=us"
          className="group relative rounded-xl border border-slate-200 bg-white p-6 shadow-sm hover:border-blue-300 hover:shadow-md transition-all"
        >
          <div className="flex items-start justify-between mb-4">
            <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-indigo-50 text-indigo-600">
              <Globe className="h-5 w-5" />
            </div>
            <ArrowRight className="h-4 w-4 text-slate-300 group-hover:text-indigo-500 transition-colors" />
          </div>
          <h2 className="text-base font-semibold text-slate-900 mb-1">US Loan Assessment</h2>
          <p className="text-sm text-slate-500 leading-relaxed">
            Evaluate US applicants using FICO score, USD income and US lending criteria.
          </p>
        </Link>
      </div>

      {/* Service status — minimal */}
      <div className="flex items-center gap-2 text-xs text-slate-400">
        <span
          className={`h-1.5 w-1.5 rounded-full ${isOnline ? 'bg-emerald-400' : 'bg-amber-400'}`}
        />
        <span>{isOnline ? 'Assessment service is available.' : 'Assessment service is currently unavailable. Please try again later.'}</span>
      </div>

      {/* Disclaimer */}
      <div className="rounded-lg border border-slate-200 bg-white p-4 flex gap-3">
        <Info className="h-4 w-4 text-slate-400 shrink-0 mt-0.5" />
        <p className="text-xs text-slate-500 leading-relaxed">
          Risk assessments are decision-support outputs produced by statistical models.
          They do not constitute a lending decision, credit approval, or financial advice.
          Results should be reviewed by a qualified professional before any credit decision is made.
        </p>
      </div>
    </div>
  );
};
