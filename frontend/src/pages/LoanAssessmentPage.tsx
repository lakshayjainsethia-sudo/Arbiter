import React, { useState, useEffect } from 'react';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import { useMutation, useQuery } from '@tanstack/react-query';
import { useSearchParams } from 'react-router-dom';
import { arbiterApi } from '../api/arbiterApi';
import type {
  LoanApplicationInput,
  LoanPredictionResponse,
  DynamicLocalExplainResponse,
  IndianLoanApplicationInput,
  IndianLoanPredictionResponse,
} from '../types/api';
import { getDecisionMeta, formatIndianLakhsCrores } from '../utils/formatters';
import {
  Globe,
  Flag,
  RotateCcw,
  AlertCircle,
  ChevronRight,
  Info,
  Loader2,
} from 'lucide-react';

// ─── Shared style constants ──────────────────────────────────────────────────
const inputCls =
  'w-full rounded-lg border border-slate-200 bg-white px-3.5 py-2.5 text-sm text-slate-900 placeholder-slate-400 focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20 outline-none transition';
const selectCls =
  'w-full rounded-lg border border-slate-200 bg-white px-3.5 py-2.5 text-sm text-slate-900 focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20 outline-none transition appearance-none';
const labelCls = 'block text-xs font-medium text-slate-600 mb-1.5';
const sectionTitleCls = 'text-xs font-semibold uppercase tracking-widest text-slate-400 mb-4';
const errorCls = 'mt-1 text-xs text-rose-600';

// ─── US form schema ──────────────────────────────────────────────────────────
const usSchema = z.object({
  revenue: z.number().min(0),
  dti_n: z.number().min(0),
  loan_amnt: z.number().gt(0),
  fico_n: z.number().min(300).max(850),
  experience_c: z.number().min(0),
  emp_length: z.string().min(1),
  purpose: z.string().min(1),
  home_ownership_n: z.string().min(1),
  addr_state: z.string().length(2),
});
type USForm = z.infer<typeof usSchema>;

// ─── Indian form schema ──────────────────────────────────────────────────────
const indiaSchema = z.object({
  age: z.number().min(18).max(70),
  gender: z.string().min(1),
  marital_status: z.string().min(1),
  city_tier: z.string().min(1),
  state: z.string().min(1),
  employment_type: z.string().min(1),
  bank_account_vintage_years: z.number().min(0).max(40),
  monthly_income_inr: z.number().min(5000).max(5000000),
  existing_loans_count: z.number().min(0).max(20),
  existing_emi_monthly_inr: z.number().min(0).max(500000),
  credit_utilization_ratio_pct: z.number().min(0).max(100),
  num_credit_inquiries_last_6m: z.number().min(0).max(20),
  late_payments_last_12m: z.number().min(0).max(24),
  loan_amount_requested_inr: z.number().min(10000).max(10000000),
  loan_tenure_months: z.number().min(6).max(360),
  loan_purpose: z.string().min(1),
  collateral_provided: z.string().min(1),
  cibil_score: z.number().min(300).max(900),
  interest_rate_offered_pct: z.number().min(1).max(40),
});
type IndiaForm = z.infer<typeof indiaSchema>;

// ─── Fallback categories ─────────────────────────────────────────────────────
const FALLBACK_INDIAN = {
  gender: ['Male', 'Female'],
  marital_status: ['Single', 'Married', 'Divorced', 'Widowed'],
  city_tier: ['Tier 1', 'Tier 2', 'Tier 3'],
  state: [
    'Andhra Pradesh', 'Assam', 'Bihar', 'Delhi', 'Gujarat', 'Haryana',
    'Jharkhand', 'Karnataka', 'Kerala', 'Madhya Pradesh', 'Maharashtra',
    'Odisha', 'Punjab', 'Rajasthan', 'Tamil Nadu', 'Telangana',
    'Uttar Pradesh', 'West Bengal',
  ],
  employment_type: ['Salaried', 'Self-Employed', 'Business Owner', 'Freelancer'],
  loan_purpose: [
    'Home Improvement', 'Medical', 'Education', 'Debt Consolidation',
    'Vehicle', 'Business', 'Wedding', 'Travel', 'Other',
  ],
  collateral_provided: ['Yes', 'No'],
};

const DEFAULT_INDIA: IndiaForm = {
  age: 32, gender: 'Male', marital_status: 'Married',
  city_tier: 'Tier 1', state: 'Maharashtra', employment_type: 'Salaried',
  bank_account_vintage_years: 5, monthly_income_inr: 75000,
  existing_loans_count: 1, existing_emi_monthly_inr: 8000,
  credit_utilization_ratio_pct: 35, num_credit_inquiries_last_6m: 2,
  late_payments_last_12m: 0, loan_amount_requested_inr: 500000,
  loan_tenure_months: 36, loan_purpose: 'Home Improvement',
  collateral_provided: 'No', cibil_score: 740, interest_rate_offered_pct: 11.5,
};

const DEFAULT_US: USForm = {
  revenue: 65000, dti_n: 16.5, loan_amnt: 12000,
  fico_n: 710, experience_c: 8, emp_length: '5 years',
  purpose: 'debt_consolidation', home_ownership_n: 'MORTGAGE', addr_state: 'CA',
};

// ─── Sub-components ──────────────────────────────────────────────────────────
const FormSection: React.FC<{ title: string; children: React.ReactNode }> = ({ title, children }) => (
  <div className="space-y-4">
    <p className={sectionTitleCls}>{title}</p>
    {children}
  </div>
);

const Field: React.FC<{ label: string; hint?: string; children: React.ReactNode; error?: string }> = ({
  label, hint, children, error,
}) => (
  <div>
    <label className={labelCls}>{label}</label>
    {children}
    {error && <p className={errorCls}>{error}</p>}
    {hint && !error && <p className="mt-1 text-[11px] text-slate-400">{hint}</p>}
  </div>
);

const TwoCol: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">{children}</div>
);

// ─── Risk result card ────────────────────────────────────────────────────────
const USResultCard: React.FC<{
  result: LoanPredictionResponse;
  explain: DynamicLocalExplainResponse | null;
  onClear: () => void;
}> = ({ result, explain, onClear }) => {
  const meta = getDecisionMeta(result.decision);
  const prob = result.default_probability;

  const riskColors = {
    lower_predicted_default_risk: { bar: 'bg-emerald-500', text: 'text-emerald-700', bg: 'bg-emerald-50', border: 'border-emerald-200' },
    manual_review: { bar: 'bg-amber-500', text: 'text-amber-700', bg: 'bg-amber-50', border: 'border-amber-200' },
    higher_predicted_default_risk: { bar: 'bg-rose-500', text: 'text-rose-700', bg: 'bg-rose-50', border: 'border-rose-200' },
  } as const;
  const colors = riskColors[result.decision as keyof typeof riskColors] ?? {
    bar: 'bg-slate-400', text: 'text-slate-700', bg: 'bg-slate-50', border: 'border-slate-200',
  };

  return (
    <div className="rounded-xl border border-slate-200 bg-white shadow-sm overflow-hidden">
      {/* Header band */}
      <div className={`px-6 py-4 border-b ${colors.bg} ${colors.border}`}>
        <div className="flex items-center justify-between">
          <div>
            <p className="text-xs font-medium text-slate-500 mb-0.5">Risk Assessment Result</p>
            <h3 className={`text-lg font-bold ${colors.text}`}>{meta.label}</h3>
          </div>
          <button onClick={onClear} className="text-xs text-slate-400 hover:text-slate-600 transition-colors">
            New assessment
          </button>
        </div>
      </div>

      <div className="p-6 space-y-6">
        {/* Probability bar */}
        <div className="space-y-2">
          <div className="flex justify-between items-baseline">
            <span className="text-xs font-medium text-slate-500">Estimated Default Probability</span>
            <span className={`text-2xl font-bold tabular-nums ${colors.text}`}>
              {(prob * 100).toFixed(1)}%
            </span>
          </div>
          <div className="relative h-2 w-full rounded-full bg-slate-100">
            {/* Zone bands */}
            <div className="absolute inset-0 rounded-full overflow-hidden flex">
              <div className="bg-emerald-100" style={{ width: '45%' }} />
              <div className="bg-amber-100" style={{ width: '10%' }} />
              <div className="bg-rose-100" style={{ width: '45%' }} />
            </div>
            {/* Filled bar */}
            <div
              className={`absolute top-0 left-0 h-full rounded-full transition-all duration-700 ${colors.bar}`}
              style={{ width: `${Math.min(100, prob * 100)}%` }}
            />
          </div>
          <div className="flex justify-between text-[10px] text-slate-400 font-mono">
            <span>0%</span>
            <span>45% · 55%</span>
            <span>100%</span>
          </div>
        </div>

        {/* Description */}
        <p className="text-sm text-slate-600 leading-relaxed">{meta.description}</p>

        {/* Key factors */}
        {explain && explain.top_contributions.length > 0 && (
          <div className="space-y-3">
            <p className="text-xs font-semibold uppercase tracking-widest text-slate-400">Key Contributing Factors</p>
            <div className="space-y-2">
              {explain.top_contributions.slice(0, 5).map((f) => {
                const isRisk = f.shap_value > 0;
                const featureName = f.feature
                  .replace('categorical__', '')
                  .replace('numeric__', '')
                  .replace(/_/g, ' ');
                const absVal = Math.abs(f.shap_value);
                const maxVal = Math.abs(explain.top_contributions[0]?.shap_value ?? 1);
                const barWidth = Math.min(100, (absVal / maxVal) * 100);
                return (
                  <div key={f.feature} className="flex items-center gap-3">
                    <span className="text-xs text-slate-600 w-36 truncate capitalize shrink-0">{featureName}</span>
                    <div className="flex-1 h-1.5 rounded-full bg-slate-100">
                      <div
                        className={`h-full rounded-full ${isRisk ? 'bg-rose-400' : 'bg-emerald-400'}`}
                        style={{ width: `${barWidth}%` }}
                      />
                    </div>
                    <span className={`text-xs font-mono w-14 text-right shrink-0 ${isRisk ? 'text-rose-600' : 'text-emerald-600'}`}>
                      {isRisk ? '↑ Risk' : '↓ Risk'}
                    </span>
                  </div>
                );
              })}
            </div>
            <p className="text-[11px] text-slate-400">
              Factors shown by relative influence on this result. Association only — not causal.
            </p>
          </div>
        )}

        {/* Disclaimer */}
        <div className="flex gap-2 rounded-lg bg-slate-50 border border-slate-200 p-3">
          <Info className="h-3.5 w-3.5 text-slate-400 shrink-0 mt-0.5" />
          <p className="text-[11px] text-slate-500 leading-relaxed">
            This output is a decision-support estimate, not a lending decision or credit determination.
          </p>
        </div>
      </div>
    </div>
  );
};

const IndianResultCard: React.FC<{
  result: IndianLoanPredictionResponse;
  shapTop: Array<{ rank: number; feature: string; mean_abs_shap: number; direction: string }> | null;
  threshold: number;
  onClear: () => void;
}> = ({ result, shapTop, threshold, onClear }) => {
  const prob = result.prediction.default_probability_raw;
  const isHigher = result.prediction.classification.includes('higher');

  const colors = isHigher
    ? { bar: 'bg-rose-500', text: 'text-rose-700', bg: 'bg-rose-50', border: 'border-rose-200' }
    : { bar: 'bg-emerald-500', text: 'text-emerald-700', bg: 'bg-emerald-50', border: 'border-emerald-200' };

  const label = isHigher ? 'Higher Predicted Risk' : 'Lower Predicted Risk';
  const description = isHigher
    ? 'The model estimates an elevated probability of default for this applicant profile.'
    : 'The model estimates a lower probability of default for this applicant profile.';

  return (
    <div className="rounded-xl border border-slate-200 bg-white shadow-sm overflow-hidden">
      <div className={`px-6 py-4 border-b ${colors.bg} ${colors.border}`}>
        <div className="flex items-center justify-between">
          <div>
            <p className="text-xs font-medium text-slate-500 mb-0.5">Risk Assessment Result</p>
            <h3 className={`text-lg font-bold ${colors.text}`}>{label}</h3>
          </div>
          <button onClick={onClear} className="text-xs text-slate-400 hover:text-slate-600 transition-colors">
            New assessment
          </button>
        </div>
      </div>

      <div className="p-6 space-y-6">
        {/* Probability bar */}
        <div className="space-y-2">
          <div className="flex justify-between items-baseline">
            <span className="text-xs font-medium text-slate-500">Estimated Default Probability</span>
            <span className={`text-2xl font-bold tabular-nums ${colors.text}`}>
              {(prob * 100).toFixed(1)}%
            </span>
          </div>
          <div className="relative h-2 w-full rounded-full bg-slate-100 overflow-visible">
            <div
              className={`h-full rounded-full transition-all duration-700 ${colors.bar}`}
              style={{ width: `${Math.min(100, prob * 100)}%` }}
            />
            {/* Threshold marker */}
            <div
              className="absolute top-1/2 -translate-y-1/2 w-px h-4 bg-slate-500"
              style={{ left: `${threshold * 100}%` }}
              title={`Classification threshold: ${(threshold * 100).toFixed(1)}%`}
            />
          </div>
          <div className="flex justify-between text-[10px] text-slate-400 font-mono">
            <span>0%</span>
            <span>Threshold: {(threshold * 100).toFixed(0)}%</span>
            <span>100%</span>
          </div>
          <p className="text-[10px] text-slate-400 italic">
            Probability estimate — not a validated real-world default rate.
          </p>
        </div>

        {/* Description */}
        <p className="text-sm text-slate-600 leading-relaxed">{description}</p>

        {/* Key factors from pre-computed SHAP */}
        {shapTop && shapTop.length > 0 && (
          <div className="space-y-3">
            <p className="text-xs font-semibold uppercase tracking-widest text-slate-400">Key Contributing Factors</p>
            <div className="space-y-2">
              {shapTop.slice(0, 5).map((f) => {
                const featureName = f.feature.replace(/_/g, ' ');
                const maxVal = shapTop[0]?.mean_abs_shap ?? 1;
                const barWidth = Math.min(100, (f.mean_abs_shap / maxVal) * 100);
                const isIncreasing = f.direction?.toLowerCase().includes('increase') ||
                  f.direction?.toLowerCase().includes('higher') ||
                  f.direction?.toLowerCase().includes('positive');
                return (
                  <div key={f.feature} className="flex items-center gap-3">
                    <span className="text-xs text-slate-600 w-36 truncate capitalize shrink-0">{featureName}</span>
                    <div className="flex-1 h-1.5 rounded-full bg-slate-100">
                      <div
                        className={`h-full rounded-full ${isIncreasing ? 'bg-rose-400' : 'bg-emerald-400'}`}
                        style={{ width: `${barWidth}%` }}
                      />
                    </div>
                    <span className={`text-xs font-mono w-14 text-right shrink-0 ${isIncreasing ? 'text-rose-600' : 'text-emerald-600'}`}>
                      {isIncreasing ? '↑ Risk' : '↓ Risk'}
                    </span>
                  </div>
                );
              })}
            </div>
            <p className="text-[11px] text-slate-400">
              Factors shown by relative importance from model training. Association only — not causal.
            </p>
          </div>
        )}

        {/* Disclaimer */}
        <div className="flex gap-2 rounded-lg bg-slate-50 border border-slate-200 p-3">
          <Info className="h-3.5 w-3.5 text-slate-400 shrink-0 mt-0.5" />
          <p className="text-[11px] text-slate-500 leading-relaxed">
            This output is a decision-support estimate, not a lending decision or credit determination.
            Assessment based on a research dataset of unverified provenance.
          </p>
        </div>
      </div>
    </div>
  );
};

// ─── Main page ───────────────────────────────────────────────────────────────
export const LoanAssessmentPage: React.FC = () => {
  const [searchParams] = useSearchParams();
  const initialType = searchParams.get('type') === 'us' ? 'us' : 'india';
  const [mode, setMode] = useState<'india' | 'us'>(initialType);

  // US state
  const [usPrediction, setUsPrediction] = useState<LoanPredictionResponse | null>(null);
  const [usExplain, setUsExplain] = useState<DynamicLocalExplainResponse | null>(null);

  // Indian state
  const [indianPrediction, setIndianPrediction] = useState<IndianLoanPredictionResponse | null>(null);

  // US metadata
  const { data: usMetadata } = useQuery({ queryKey: ['metadata'], queryFn: arbiterApi.getMetadata });
  // Indian metadata
  const { data: indianMetadata } = useQuery({ queryKey: ['indianMetadata'], queryFn: arbiterApi.getIndianLoanMetadata });

  const indianCategories = indianMetadata?.valid_categories ?? FALLBACK_INDIAN;
  const indianThreshold = indianMetadata?.model.cv_threshold ?? 0.7623;
  const shapTop = indianMetadata?.shap_top_features ?? null;

  const usCategories = usMetadata?.supported_categories || {
    emp_length: ['1 year', '10+ years', '2 years', '3 years', '4 years', '5 years', '6 years', '7 years', '8 years', '9 years', '< 1 year', 'NI'],
    purpose: ['car', 'credit_card', 'debt_consolidation', 'home_improvement', 'house', 'major_purchase', 'medical', 'moving', 'other', 'renewable_energy', 'small_business', 'vacation'],
    home_ownership_n: ['MORTGAGE', 'OTHER', 'OWN', 'RENT'],
    addr_state: ['AK','AL','AR','AZ','CA','CO','CT','DC','DE','FL','GA','HI','IL','IN','KS','KY','LA','MA','MD','ME','MI','MN','MO','MS','MT','NC','ND','NE','NH','NJ','NM','NV','NY','OH','OK','OR','PA','RI','SC','SD','TN','TX','UT','VA','VT','WA','WI','WV','WY'],
  };

  // ── US form
  const usForm = useForm<USForm>({ resolver: zodResolver(usSchema), defaultValues: DEFAULT_US });
  const { register: ru, handleSubmit: hsu, reset: resetUS, watch: wu, formState: { errors: eu } } = usForm;

  // ── Indian form
  const indiaForm = useForm<IndiaForm>({ resolver: zodResolver(indiaSchema), defaultValues: DEFAULT_INDIA });
  const { register: ri, handleSubmit: hsi, reset: resetIN, watch: wi, formState: { errors: ei } } = indiaForm;

  // ── Switch mode — clear results
  const switchMode = (m: 'india' | 'us') => {
    setMode(m);
    setUsPrediction(null); setUsExplain(null); setIndianPrediction(null);
  };

  // Sync from URL param on mount
  useEffect(() => {
    const t = searchParams.get('type');
    if (t === 'us') switchMode('us');
    else if (t === 'india') switchMode('india');
  }, []);

  // ── Mutations
  const usMutation = useMutation({
    mutationFn: (d: LoanApplicationInput) => arbiterApi.predictLoan(d),
    onSuccess: (data, vars) => { setUsPrediction(data); explainMutation.mutate(vars); },
  });
  const explainMutation = useMutation({
    mutationFn: (d: LoanApplicationInput) => arbiterApi.explainLocal(d),
    onSuccess: (data) => setUsExplain(data),
  });
  const indianMutation = useMutation({
    mutationFn: (d: IndianLoanApplicationInput) => arbiterApi.predictIndianLoan(d),
    onSuccess: (data) => setIndianPrediction(data),
  });

  const currentFico = wu('fico_n');
  const currentCibil = wi('cibil_score');
  const currentRevenue = wu('revenue');
  const currentLoan = wu('loan_amnt');
  const currentMonthlyIncome = wi('monthly_income_inr');
  const currentIndianLoan = wi('loan_amount_requested_inr');

  const hasUSResult = !!usPrediction;
  const hasIndianResult = !!indianPrediction;

  return (
    <div className="space-y-6">
      {/* Page heading */}
      <div>
        <h1 className="text-xl font-bold text-slate-900">Loan Risk Assessment</h1>
        <p className="text-sm text-slate-500 mt-0.5">Assess borrower risk using applicant and financial information.</p>
      </div>

      {/* Mode selector */}
      <div className="flex items-center gap-1 p-1 rounded-xl bg-slate-100 w-fit">
        <button
          type="button"
          onClick={() => switchMode('india')}
          className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all ${
            mode === 'india' ? 'bg-white text-slate-900 shadow-sm' : 'text-slate-500 hover:text-slate-700'
          }`}
        >
          <Flag className="h-3.5 w-3.5" />
          Indian Loan Assessment
        </button>
        <button
          type="button"
          onClick={() => switchMode('us')}
          className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all ${
            mode === 'us' ? 'bg-white text-slate-900 shadow-sm' : 'text-slate-500 hover:text-slate-700'
          }`}
        >
          <Globe className="h-3.5 w-3.5" />
          US Loan Assessment
        </button>
      </div>

      {/* Two-column layout */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">

        {/* ── Left: Form ── */}
        <div className="lg:col-span-7">

          {/* =================== INDIAN FORM =================== */}
          {mode === 'india' && (
            <form
              onSubmit={hsi((values) => indianMutation.mutate(values as IndianLoanApplicationInput))}
              className="bg-white rounded-xl border border-slate-200 shadow-sm divide-y divide-slate-100"
            >
              {/* Form header */}
              <div className="px-6 py-4 flex items-center justify-between">
                <div>
                  <h2 className="text-sm font-semibold text-slate-900">Indian Loan Assessment</h2>
                  <p className="text-xs text-slate-500 mt-0.5">CIBIL score · INR · Indian lending criteria</p>
                </div>
                <button type="button" onClick={() => resetIN(DEFAULT_INDIA)}
                  className="text-xs text-slate-400 hover:text-slate-600 flex items-center gap-1 transition-colors">
                  <RotateCcw className="h-3 w-3" /> Reset
                </button>
              </div>

              <div className="px-6 py-5 space-y-6">
                {/* Personal Information */}
                <FormSection title="Personal Information">
                  <TwoCol>
                    <Field label="Age" error={ei.age?.message}>
                      <input type="number" min="18" max="70" placeholder="e.g. 32"
                        {...ri('age', { valueAsNumber: true })} className={inputCls} />
                    </Field>
                    <Field label="Gender" error={ei.gender?.message}>
                      <select {...ri('gender')} className={selectCls}>
                        {indianCategories.gender.map((v) => <option key={v} value={v}>{v}</option>)}
                      </select>
                    </Field>
                    <Field label="Marital Status" error={ei.marital_status?.message}>
                      <select {...ri('marital_status')} className={selectCls}>
                        {indianCategories.marital_status.map((v) => <option key={v} value={v}>{v}</option>)}
                      </select>
                    </Field>
                    <Field label="City Tier" error={ei.city_tier?.message}>
                      <select {...ri('city_tier')} className={selectCls}>
                        {indianCategories.city_tier.map((v) => <option key={v} value={v}>{v}</option>)}
                      </select>
                    </Field>
                    <Field label="State" error={ei.state?.message}>
                      <select {...ri('state')} className={selectCls}>
                        {indianCategories.state.map((v) => <option key={v} value={v}>{v}</option>)}
                      </select>
                    </Field>
                  </TwoCol>
                </FormSection>

                {/* Employment & Financial */}
                <FormSection title="Employment & Financial Information">
                  <TwoCol>
                    <Field label="Employment Type" error={ei.employment_type?.message}>
                      <select {...ri('employment_type')} className={selectCls}>
                        {indianCategories.employment_type.map((v) => <option key={v} value={v}>{v}</option>)}
                      </select>
                    </Field>
                    <Field label="Monthly Income (₹)"
                      hint={currentMonthlyIncome ? formatIndianLakhsCrores(currentMonthlyIncome * 12) + ' p.a.' : ''}
                      error={ei.monthly_income_inr?.message}>
                      <input type="number" step="1000" placeholder="e.g. 75000"
                        {...ri('monthly_income_inr', { valueAsNumber: true })} className={inputCls} />
                    </Field>
                    <Field label="Bank Account Vintage (Years)" error={ei.bank_account_vintage_years?.message}>
                      <input type="number" step="1" min="0" max="40" placeholder="e.g. 5"
                        {...ri('bank_account_vintage_years', { valueAsNumber: true })} className={inputCls} />
                    </Field>
                    <Field label="Number of Existing Loans" error={ei.existing_loans_count?.message}>
                      <input type="number" step="1" min="0" max="20" placeholder="e.g. 1"
                        {...ri('existing_loans_count', { valueAsNumber: true })} className={inputCls} />
                    </Field>
                    <Field label="Existing Monthly EMI (₹)" hint="Total current obligations"
                      error={ei.existing_emi_monthly_inr?.message}>
                      <input type="number" step="500" placeholder="e.g. 8000"
                        {...ri('existing_emi_monthly_inr', { valueAsNumber: true })} className={inputCls} />
                    </Field>
                  </TwoCol>
                </FormSection>

                {/* Credit Information */}
                <FormSection title="Credit Information">
                  <TwoCol>
                    <Field label={`CIBIL Score (300–900)`}
                      hint={currentCibil ? `Score: ${currentCibil}` : ''}
                      error={ei.cibil_score?.message}>
                      <input type="number" min="300" max="900" placeholder="e.g. 740"
                        {...ri('cibil_score', { valueAsNumber: true })} className={inputCls} />
                    </Field>
                    <Field label="Credit Utilization (%)" hint="Outstanding / total credit limit"
                      error={ei.credit_utilization_ratio_pct?.message}>
                      <input type="number" step="0.1" min="0" max="100" placeholder="e.g. 35"
                        {...ri('credit_utilization_ratio_pct', { valueAsNumber: true })} className={inputCls} />
                    </Field>
                    <Field label="Credit Inquiries (past 6 months)" error={ei.num_credit_inquiries_last_6m?.message}>
                      <input type="number" step="1" min="0" max="20" placeholder="e.g. 2"
                        {...ri('num_credit_inquiries_last_6m', { valueAsNumber: true })} className={inputCls} />
                    </Field>
                    <Field label="Late Payments (past 12 months)" error={ei.late_payments_last_12m?.message}>
                      <input type="number" step="1" min="0" max="24" placeholder="e.g. 0"
                        {...ri('late_payments_last_12m', { valueAsNumber: true })} className={inputCls} />
                    </Field>
                  </TwoCol>
                </FormSection>

                {/* Loan Details */}
                <FormSection title="Loan Details">
                  <TwoCol>
                    <Field label="Requested Loan Amount (₹)"
                      hint={currentIndianLoan ? formatIndianLakhsCrores(currentIndianLoan) : ''}
                      error={ei.loan_amount_requested_inr?.message}>
                      <input type="number" step="10000" min="10000" placeholder="e.g. 500000"
                        {...ri('loan_amount_requested_inr', { valueAsNumber: true })} className={inputCls} />
                    </Field>
                    <Field label="Loan Tenure (Months)" error={ei.loan_tenure_months?.message}>
                      <input type="number" step="6" min="6" max="360" placeholder="e.g. 36"
                        {...ri('loan_tenure_months', { valueAsNumber: true })} className={inputCls} />
                    </Field>
                    <Field label="Interest Rate Offered (%)" error={ei.interest_rate_offered_pct?.message}>
                      <input type="number" step="0.1" min="1" max="40" placeholder="e.g. 11.5"
                        {...ri('interest_rate_offered_pct', { valueAsNumber: true })} className={inputCls} />
                    </Field>
                    <Field label="Collateral Provided" error={ei.collateral_provided?.message}>
                      <select {...ri('collateral_provided')} className={selectCls}>
                        {indianCategories.collateral_provided.map((v) => <option key={v} value={v}>{v}</option>)}
                      </select>
                    </Field>
                    <Field label="Loan Purpose" error={ei.loan_purpose?.message}>
                      <select {...ri('loan_purpose')} className={selectCls}>
                        {indianCategories.loan_purpose.map((v) => <option key={v} value={v}>{v}</option>)}
                      </select>
                    </Field>
                  </TwoCol>
                </FormSection>
              </div>

              {/* Error */}
              {indianMutation.isError && (
                <div className="mx-6 mb-4 flex items-start gap-2 rounded-lg border border-rose-200 bg-rose-50 p-3 text-xs text-rose-700">
                  <AlertCircle className="h-4 w-4 shrink-0 mt-0.5" />
                  <span>{(indianMutation.error as Error)?.message || 'Unable to reach assessment service. Please try again.'}</span>
                </div>
              )}

              {/* Submit */}
              <div className="px-6 py-4">
                <button type="submit" disabled={indianMutation.isPending}
                  className="w-full flex items-center justify-center gap-2 rounded-lg bg-blue-600 px-4 py-3 text-sm font-semibold text-white hover:bg-blue-700 disabled:opacity-50 transition-colors">
                  {indianMutation.isPending
                    ? <><Loader2 className="h-4 w-4 animate-spin" /><span>Assessing…</span></>
                    : <><span>Run Assessment</span><ChevronRight className="h-4 w-4" /></>}
                </button>
              </div>
            </form>
          )}

          {/* =================== US FORM =================== */}
          {mode === 'us' && (
            <form
              onSubmit={hsu((values) => usMutation.mutate(values))}
              className="bg-white rounded-xl border border-slate-200 shadow-sm divide-y divide-slate-100"
            >
              {/* Form header */}
              <div className="px-6 py-4 flex items-center justify-between">
                <div>
                  <h2 className="text-sm font-semibold text-slate-900">US Loan Assessment</h2>
                  <p className="text-xs text-slate-500 mt-0.5">FICO score · USD · US lending criteria</p>
                </div>
                <button type="button" onClick={() => { resetUS(DEFAULT_US); }}
                  className="text-xs text-slate-400 hover:text-slate-600 flex items-center gap-1 transition-colors">
                  <RotateCcw className="h-3 w-3" /> Reset
                </button>
              </div>

              <div className="px-6 py-5 space-y-6">
                {/* Financial Information */}
                <FormSection title="Financial Information">
                  <TwoCol>
                    <Field label="Annual Income (USD)" hint={currentRevenue ? `${formatIndianLakhsCrores(currentRevenue * 85)} approx.` : ''}
                      error={eu.revenue?.message}>
                      <input type="number" step="500" placeholder="e.g. 65000"
                        {...ru('revenue', { valueAsNumber: true })} className={inputCls} />
                    </Field>
                    <Field label="Requested Loan Amount (USD)"
                      hint={currentLoan ? `${formatIndianLakhsCrores(currentLoan * 85)} approx.` : ''}
                      error={eu.loan_amnt?.message}>
                      <input type="number" step="500" placeholder="e.g. 12000"
                        {...ru('loan_amnt', { valueAsNumber: true })} className={inputCls} />
                    </Field>
                    <Field label="Debt-to-Income Ratio (%)" hint="Monthly obligations / gross income"
                      error={eu.dti_n?.message}>
                      <input type="number" step="0.1" placeholder="e.g. 16.5"
                        {...ru('dti_n', { valueAsNumber: true })} className={inputCls} />
                    </Field>
                    <Field label={`FICO Credit Score (300–850)`}
                      hint={currentFico ? `Score: ${currentFico}` : ''}
                      error={eu.fico_n?.message}>
                      <input type="number" min="300" max="850" placeholder="e.g. 710"
                        {...ru('fico_n', { valueAsNumber: true })} className={inputCls} />
                    </Field>
                  </TwoCol>
                </FormSection>

                {/* Employment & Credit */}
                <FormSection title="Employment & Credit History">
                  <TwoCol>
                    <Field label="Employment Length" error={eu.emp_length?.message}>
                      <select {...ru('emp_length')} className={selectCls}>
                        {usCategories.emp_length.map((v) => (
                          <option key={v} value={v}>{v === 'NI' ? 'Not Indicated' : v}</option>
                        ))}
                      </select>
                    </Field>
                    <Field label="Years of Credit History" error={eu.experience_c?.message}>
                      <input type="number" step="1" placeholder="e.g. 8"
                        {...ru('experience_c', { valueAsNumber: true })} className={inputCls} />
                    </Field>
                  </TwoCol>
                </FormSection>

                {/* Loan Details */}
                <FormSection title="Loan Details">
                  <TwoCol>
                    <Field label="Loan Purpose" error={eu.purpose?.message}>
                      <select {...ru('purpose')} className={selectCls}>
                        {usCategories.purpose.map((v) => (
                          <option key={v} value={v}>{v.replace(/_/g, ' ')}</option>
                        ))}
                      </select>
                    </Field>
                    <Field label="Home Ownership" error={eu.home_ownership_n?.message}>
                      <select {...ru('home_ownership_n')} className={selectCls}>
                        {usCategories.home_ownership_n.map((v) => (
                          <option key={v} value={v}>{v}</option>
                        ))}
                      </select>
                    </Field>
                    <Field label="State" error={eu.addr_state?.message}>
                      <select {...ru('addr_state')} className={selectCls}>
                        {usCategories.addr_state.map((v) => (
                          <option key={v} value={v}>{v}</option>
                        ))}
                      </select>
                    </Field>
                  </TwoCol>
                </FormSection>

                {/* Presets */}
                {usMetadata?.preset_profiles && usMetadata.preset_profiles.length > 0 && (
                  <FormSection title="Quick Presets">
                    <div className="grid grid-cols-2 gap-2">
                      {usMetadata.preset_profiles.map((p) => (
                        <button key={p.id} type="button"
                          onClick={() => resetUS(p.data)}
                          className="text-left p-3 rounded-lg border border-slate-200 hover:border-blue-300 hover:bg-blue-50/50 text-xs transition-all">
                          <div className="font-medium text-slate-800 truncate">{p.label}</div>
                          <div className="text-slate-500 truncate mt-0.5">{p.description}</div>
                        </button>
                      ))}
                    </div>
                  </FormSection>
                )}
              </div>

              {/* Error */}
              {usMutation.isError && (
                <div className="mx-6 mb-4 flex items-start gap-2 rounded-lg border border-rose-200 bg-rose-50 p-3 text-xs text-rose-700">
                  <AlertCircle className="h-4 w-4 shrink-0 mt-0.5" />
                  <span>{(usMutation.error as Error)?.message || 'Unable to reach assessment service. Please try again.'}</span>
                </div>
              )}

              {/* Submit */}
              <div className="px-6 py-4">
                <button type="submit" disabled={usMutation.isPending}
                  className="w-full flex items-center justify-center gap-2 rounded-lg bg-blue-600 px-4 py-3 text-sm font-semibold text-white hover:bg-blue-700 disabled:opacity-50 transition-colors">
                  {usMutation.isPending
                    ? <><Loader2 className="h-4 w-4 animate-spin" /><span>Assessing…</span></>
                    : <><span>Run Assessment</span><ChevronRight className="h-4 w-4" /></>}
                </button>
              </div>
            </form>
          )}
        </div>

        {/* ── Right: Results ── */}
        <div className="lg:col-span-5">
          {mode === 'us' && hasUSResult && usPrediction && (
            <USResultCard
              result={usPrediction}
              explain={usExplain}
              onClear={() => { setUsPrediction(null); setUsExplain(null); }}
            />
          )}

          {mode === 'india' && hasIndianResult && indianPrediction && (
            <IndianResultCard
              result={indianPrediction}
              shapTop={shapTop}
              threshold={indianThreshold}
              onClear={() => setIndianPrediction(null)}
            />
          )}

          {((mode === 'us' && !hasUSResult) || (mode === 'india' && !hasIndianResult)) && (
            <div className="rounded-xl border border-dashed border-slate-300 bg-white p-10 text-center space-y-3">
              <div className="mx-auto h-10 w-10 rounded-full bg-slate-100 flex items-center justify-center">
                <ChevronRight className="h-5 w-5 text-slate-400" />
              </div>
              <p className="text-sm font-medium text-slate-600">Assessment result will appear here</p>
              <p className="text-xs text-slate-400">Complete the form and submit to see the risk evaluation.</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
