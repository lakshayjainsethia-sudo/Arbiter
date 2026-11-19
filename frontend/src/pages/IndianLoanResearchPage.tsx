import React, { useState } from 'react';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import { useMutation, useQuery } from '@tanstack/react-query';
import { arbiterApi } from '../api/arbiterApi';
import type {
  IndianLoanApplicationInput,
  IndianLoanPredictionResponse,
} from '../types/api';
import {
  AlertTriangle,
  FlaskConical,
  ShieldAlert,
  Info,
  RotateCcw,
  BarChart3,
  Flag,
  CheckCircle,
  XCircle,
  Loader2,
} from 'lucide-react';

// ─────────────────────────────────────────────────────────────
// Validation schema
// ─────────────────────────────────────────────────────────────
const VALID_GENDERS       = ['Male', 'Female', 'Other'] as const;
const VALID_MARITAL       = ['Single', 'Married', 'Divorced', 'Widowed'] as const;
const VALID_CITY_TIER     = ['Tier 1', 'Tier 2', 'Tier 3'] as const;
const VALID_STATES        = [
  'Andhra Pradesh', 'Bihar', 'Delhi', 'Gujarat', 'Karnataka',
  'Kerala', 'Madhya Pradesh', 'Maharashtra', 'Odisha', 'Punjab',
  'Rajasthan', 'Tamil Nadu', 'Telangana', 'Uttar Pradesh', 'West Bengal',
] as const;
const VALID_EMPLOYMENT    = ['Salaried', 'Self-Employed', 'Business Owner', 'Unemployed'] as const;
const VALID_PURPOSE       = [
  'Home Improvement', 'Education', 'Medical', 'Debt Consolidation',
  'Business', 'Vehicle', 'Personal', 'Travel', 'Other',
] as const;
const VALID_COLLATERAL    = ['Yes', 'No'] as const;

const indianLoanSchema = z.object({
  age:                          z.number().int().min(18).max(70),
  gender:                       z.enum(VALID_GENDERS),
  marital_status:               z.enum(VALID_MARITAL),
  city_tier:                    z.enum(VALID_CITY_TIER),
  state:                        z.enum(VALID_STATES),
  employment_type:              z.enum(VALID_EMPLOYMENT),
  bank_account_vintage_years:   z.number().min(0).max(40),
  monthly_income_inr:           z.number().positive('Must be positive'),
  existing_loans_count:         z.number().int().min(0).max(20),
  existing_emi_monthly_inr:     z.number().min(0),
  credit_utilization_ratio_pct: z.number().min(0).max(100),
  num_credit_inquiries_last_6m: z.number().int().min(0).max(50),
  late_payments_last_12m:       z.number().int().min(0).max(12),
  loan_amount_requested_inr:    z.number().positive('Must be positive'),
  loan_tenure_months:           z.number().int().min(6).max(360),
  loan_purpose:                 z.enum(VALID_PURPOSE),
  collateral_provided:          z.enum(VALID_COLLATERAL),
  cibil_score:                  z.number().int().min(300).max(900),
  interest_rate_offered_pct:    z.number().min(0).max(100),
});

type IndianLoanFormValues = z.infer<typeof indianLoanSchema>;

const DEFAULT_VALUES: IndianLoanFormValues = {
  age:                          35,
  gender:                       'Male',
  marital_status:               'Married',
  city_tier:                    'Tier 1',
  state:                        'Maharashtra',
  employment_type:              'Salaried',
  bank_account_vintage_years:   5,
  monthly_income_inr:           75000,
  existing_loans_count:         1,
  existing_emi_monthly_inr:     8000,
  credit_utilization_ratio_pct: 35,
  num_credit_inquiries_last_6m: 2,
  late_payments_last_12m:       0,
  loan_amount_requested_inr:    500000,
  loan_tenure_months:           36,
  loan_purpose:                 'Home Improvement',
  collateral_provided:          'No',
  cibil_score:                  740,
  interest_rate_offered_pct:    11.5,
};

// ─────────────────────────────────────────────────────────────
// Helper components
// ─────────────────────────────────────────────────────────────
const FieldError: React.FC<{ msg?: string }> = ({ msg }) =>
  msg ? <p className="mt-1 text-xs text-rose-400">{msg}</p> : null;

const Label: React.FC<{ children: React.ReactNode; htmlFor: string }> = ({ children, htmlFor }) => (
  <label htmlFor={htmlFor} className="block text-xs font-medium text-slate-400 mb-1">
    {children}
  </label>
);

const inputCls = "w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-100 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent placeholder-slate-500";
const selectCls = "w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-100 focus:outline-none focus:ring-2 focus:ring-blue-500";

// ─────────────────────────────────────────────────────────────
// Main page
// ─────────────────────────────────────────────────────────────
export const IndianLoanResearchPage: React.FC = () => {
  const [result, setResult] = useState<IndianLoanPredictionResponse | null>(null);

  const { data: meta } = useQuery({
    queryKey: ['indian-loan-metadata'],
    queryFn: arbiterApi.getIndianLoanMetadata,
    retry: 1,
  });

  const {
    register,
    handleSubmit,
    reset,
    formState: { errors },
  } = useForm<IndianLoanFormValues>({
    resolver: zodResolver(indianLoanSchema),
    defaultValues: DEFAULT_VALUES,
  });

  const predictMutation = useMutation({
    mutationFn: (data: IndianLoanApplicationInput) => arbiterApi.predictIndianLoan(data),
    onSuccess: (data) => setResult(data),
  });

  const onSubmit = (values: IndianLoanFormValues) => {
    predictMutation.mutate(values);
  };

  const prob = result?.prediction.default_probability_raw ?? null;
  const isHighRisk = result
    ? result.prediction.predicted_class === 1
    : false;

  const classificationLabel = result
    ? result.prediction.classification === 'higher_predicted_default_risk'
      ? 'Higher predicted default risk'
      : 'Lower predicted default risk'
    : null;

  const shap = meta?.shap_top_features ?? [];

  return (
    <div className="p-6 max-w-6xl mx-auto space-y-6">
      {/* ── Page header ─────────────────────────────────────── */}
      <div className="flex items-start justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-amber-500/10 border border-amber-500/30">
            <FlaskConical className="h-5 w-5 text-amber-400" />
          </div>
          <div>
            <h1 className="text-xl font-bold text-slate-100">
              Indian Loan Default Research
            </h1>
            <p className="text-xs text-slate-400">
              Experimental Logistic Regression · Stage 3B · v{meta?.model.version ?? '3B-experimental'}
            </p>
          </div>
        </div>
        <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-amber-500/10 border border-amber-500/30 text-amber-400 text-xs font-bold tracking-wider">
          <AlertTriangle className="h-3.5 w-3.5" />
          EXPERIMENTAL
        </span>
      </div>

      {/* ── Global disclaimer ────────────────────────────────── */}
      <div className="rounded-xl border border-amber-500/30 bg-amber-500/5 p-4">
        <div className="flex gap-3">
          <ShieldAlert className="h-5 w-5 text-amber-400 shrink-0 mt-0.5" />
          <div className="space-y-1">
            <p className="text-sm font-semibold text-amber-300">
              Research Prototype — Not for Real Lending Decisions
            </p>
            <p className="text-xs text-slate-400 leading-relaxed">
              This model was trained on a dataset of <strong className="text-slate-300">unverified provenance</strong>.
              The state distribution (~6.5–7% per state) is consistent with synthetic or stratified generation.
              Results are experimental research outputs only. They must not be used to approve, reject, price,
              or determine eligibility for any loan.
            </p>
            {meta?.model.dataset_caveat && (
              <p className="text-xs text-slate-500 italic">{meta.model.dataset_caveat}</p>
            )}
          </div>
        </div>
      </div>

      {/* ── Gender fairness warning ──────────────────────────── */}
      <div className="rounded-xl border border-orange-500/30 bg-orange-500/5 p-4 flex gap-3">
        <Flag className="h-5 w-5 text-orange-400 shrink-0 mt-0.5" />
        <div>
          <p className="text-sm font-semibold text-orange-300">Fairness Review Required</p>
          <p className="text-xs text-slate-400 mt-0.5">
            <strong className="text-slate-300">Gender</strong> is an active model feature (ranked #9 by SHAP importance).
            Its impact on fairness across groups has not been assessed.
            This model must not be deployed without a fairness audit.
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-5 gap-6">
        {/* ── Input form (left 3 cols) ─────────────────────── */}
        <div className="xl:col-span-3 space-y-5">
          <form onSubmit={handleSubmit(onSubmit)} className="space-y-5">
            {/* Numeric fields */}
            <section className="rounded-xl border border-slate-700/60 bg-slate-900 p-5 space-y-4">
              <h2 className="text-sm font-semibold text-slate-200 flex items-center gap-2">
                <BarChart3 className="h-4 w-4 text-blue-400" />
                Applicant &amp; Loan Attributes
              </h2>
              <div className="grid grid-cols-2 gap-4">
                {/* Age */}
                <div>
                  <Label htmlFor="age">Age (18–70)</Label>
                  <input id="age" type="number" {...register('age', { valueAsNumber: true })} className={inputCls} />
                  <FieldError msg={errors.age?.message} />
                </div>
                {/* CIBIL */}
                <div>
                  <Label htmlFor="cibil_score">CIBIL Score (300–900)</Label>
                  <input id="cibil_score" type="number" {...register('cibil_score', { valueAsNumber: true })} className={inputCls} />
                  <FieldError msg={errors.cibil_score?.message} />
                </div>
                {/* Monthly income */}
                <div>
                  <Label htmlFor="monthly_income_inr">Monthly Income (₹)</Label>
                  <input id="monthly_income_inr" type="number" {...register('monthly_income_inr', { valueAsNumber: true })} className={inputCls} />
                  <FieldError msg={errors.monthly_income_inr?.message} />
                </div>
                {/* Loan amount */}
                <div>
                  <Label htmlFor="loan_amount_requested_inr">Loan Amount (₹)</Label>
                  <input id="loan_amount_requested_inr" type="number" {...register('loan_amount_requested_inr', { valueAsNumber: true })} className={inputCls} />
                  <FieldError msg={errors.loan_amount_requested_inr?.message} />
                </div>
                {/* Tenure */}
                <div>
                  <Label htmlFor="loan_tenure_months">Loan Tenure (months)</Label>
                  <input id="loan_tenure_months" type="number" {...register('loan_tenure_months', { valueAsNumber: true })} className={inputCls} />
                  <FieldError msg={errors.loan_tenure_months?.message} />
                </div>
                {/* Interest rate */}
                <div>
                  <Label htmlFor="interest_rate_offered_pct">Interest Rate (%)</Label>
                  <input id="interest_rate_offered_pct" type="number" step="0.01" {...register('interest_rate_offered_pct', { valueAsNumber: true })} className={inputCls} />
                  <FieldError msg={errors.interest_rate_offered_pct?.message} />
                </div>
                {/* Existing loans */}
                <div>
                  <Label htmlFor="existing_loans_count">Existing Loans Count</Label>
                  <input id="existing_loans_count" type="number" {...register('existing_loans_count', { valueAsNumber: true })} className={inputCls} />
                  <FieldError msg={errors.existing_loans_count?.message} />
                </div>
                {/* Existing EMI */}
                <div>
                  <Label htmlFor="existing_emi_monthly_inr">Existing EMI / Month (₹)</Label>
                  <input id="existing_emi_monthly_inr" type="number" {...register('existing_emi_monthly_inr', { valueAsNumber: true })} className={inputCls} />
                  <FieldError msg={errors.existing_emi_monthly_inr?.message} />
                </div>
                {/* Credit utilisation */}
                <div>
                  <Label htmlFor="credit_utilization_ratio_pct">Credit Utilization (%)</Label>
                  <input id="credit_utilization_ratio_pct" type="number" step="0.1" {...register('credit_utilization_ratio_pct', { valueAsNumber: true })} className={inputCls} />
                  <FieldError msg={errors.credit_utilization_ratio_pct?.message} />
                </div>
                {/* Credit inquiries */}
                <div>
                  <Label htmlFor="num_credit_inquiries_last_6m">Credit Inquiries (6m)</Label>
                  <input id="num_credit_inquiries_last_6m" type="number" {...register('num_credit_inquiries_last_6m', { valueAsNumber: true })} className={inputCls} />
                  <FieldError msg={errors.num_credit_inquiries_last_6m?.message} />
                </div>
                {/* Late payments */}
                <div>
                  <Label htmlFor="late_payments_last_12m">Late Payments (12m)</Label>
                  <input id="late_payments_last_12m" type="number" {...register('late_payments_last_12m', { valueAsNumber: true })} className={inputCls} />
                  <FieldError msg={errors.late_payments_last_12m?.message} />
                </div>
                {/* Bank vintage */}
                <div>
                  <Label htmlFor="bank_account_vintage_years">Bank Account Vintage (yrs)</Label>
                  <input id="bank_account_vintage_years" type="number" step="0.5" {...register('bank_account_vintage_years', { valueAsNumber: true })} className={inputCls} />
                  <FieldError msg={errors.bank_account_vintage_years?.message} />
                </div>
              </div>
            </section>

            {/* Categorical fields */}
            <section className="rounded-xl border border-slate-700/60 bg-slate-900 p-5 space-y-4">
              <h2 className="text-sm font-semibold text-slate-200">Demographic &amp; Categorical Fields</h2>
              <div className="grid grid-cols-2 gap-4">
                {/* Gender — show fairness note */}
                <div>
                  <Label htmlFor="gender">
                    Gender{' '}
                    <span className="text-orange-400 text-[10px] font-semibold ml-1">[FAIRNESS FLAG]</span>
                  </Label>
                  <select id="gender" {...register('gender')} className={selectCls}>
                    {VALID_GENDERS.map(g => <option key={g} value={g}>{g}</option>)}
                  </select>
                  <p className="mt-1 text-[10px] text-orange-400">Active model feature — fairness unverified</p>
                  <FieldError msg={errors.gender?.message} />
                </div>
                {/* Marital status */}
                <div>
                  <Label htmlFor="marital_status">Marital Status</Label>
                  <select id="marital_status" {...register('marital_status')} className={selectCls}>
                    {VALID_MARITAL.map(v => <option key={v} value={v}>{v}</option>)}
                  </select>
                  <FieldError msg={errors.marital_status?.message} />
                </div>
                {/* City tier */}
                <div>
                  <Label htmlFor="city_tier">City Tier</Label>
                  <select id="city_tier" {...register('city_tier')} className={selectCls}>
                    {VALID_CITY_TIER.map(v => <option key={v} value={v}>{v}</option>)}
                  </select>
                  <FieldError msg={errors.city_tier?.message} />
                </div>
                {/* State */}
                <div>
                  <Label htmlFor="state">State</Label>
                  <select id="state" {...register('state')} className={selectCls}>
                    {VALID_STATES.map(v => <option key={v} value={v}>{v}</option>)}
                  </select>
                  <FieldError msg={errors.state?.message} />
                </div>
                {/* Employment */}
                <div>
                  <Label htmlFor="employment_type">Employment Type</Label>
                  <select id="employment_type" {...register('employment_type')} className={selectCls}>
                    {VALID_EMPLOYMENT.map(v => <option key={v} value={v}>{v}</option>)}
                  </select>
                  <FieldError msg={errors.employment_type?.message} />
                </div>
                {/* Loan purpose */}
                <div>
                  <Label htmlFor="loan_purpose">Loan Purpose</Label>
                  <select id="loan_purpose" {...register('loan_purpose')} className={selectCls}>
                    {VALID_PURPOSE.map(v => <option key={v} value={v}>{v}</option>)}
                  </select>
                  <FieldError msg={errors.loan_purpose?.message} />
                </div>
                {/* Collateral */}
                <div>
                  <Label htmlFor="collateral_provided">Collateral Provided</Label>
                  <select id="collateral_provided" {...register('collateral_provided')} className={selectCls}>
                    {VALID_COLLATERAL.map(v => <option key={v} value={v}>{v}</option>)}
                  </select>
                  <FieldError msg={errors.collateral_provided?.message} />
                </div>
              </div>
            </section>

            <div className="flex gap-3">
              <button
                type="submit"
                disabled={predictMutation.isPending}
                className="flex-1 flex items-center justify-center gap-2 rounded-lg bg-blue-600 hover:bg-blue-700 disabled:opacity-60 disabled:cursor-not-allowed text-white font-semibold text-sm py-2.5 px-4 transition-colors"
              >
                {predictMutation.isPending ? (
                  <><Loader2 className="h-4 w-4 animate-spin" /> Computing…</>
                ) : (
                  'Run Research Prediction'
                )}
              </button>
              <button
                type="button"
                onClick={() => { reset(DEFAULT_VALUES); setResult(null); }}
                className="flex items-center gap-1.5 rounded-lg border border-slate-700 hover:border-slate-600 text-slate-400 hover:text-slate-200 text-sm py-2.5 px-4 transition-colors"
              >
                <RotateCcw className="h-4 w-4" /> Reset
              </button>
            </div>

            {predictMutation.isError && (
              <div className="rounded-lg bg-rose-900/20 border border-rose-700/40 p-3 text-sm text-rose-300">
                <strong>Error:</strong>{' '}
                {predictMutation.error instanceof Error
                  ? predictMutation.error.message
                  : 'Request failed. Check network or server.'}
              </div>
            )}
          </form>
        </div>

        {/* ── Results panel (right 2 cols) ─────────────────── */}
        <div className="xl:col-span-2 space-y-4">
          {/* Result card */}
          {result ? (
            <div className="rounded-xl border border-slate-700/60 bg-slate-900 p-5 space-y-5">
              {/* Classification headline */}
              <div className={`rounded-lg p-4 border ${
                isHighRisk
                  ? 'bg-rose-900/20 border-rose-700/40'
                  : 'bg-emerald-900/20 border-emerald-700/40'
              }`}>
                <div className="flex items-center gap-2 mb-1">
                  {isHighRisk
                    ? <XCircle className="h-5 w-5 text-rose-400" />
                    : <CheckCircle className="h-5 w-5 text-emerald-400" />}
                  <span className={`text-sm font-semibold ${isHighRisk ? 'text-rose-300' : 'text-emerald-300'}`}>
                    {classificationLabel}
                  </span>
                </div>
                <p className="text-xs text-slate-400">
                  At CV threshold <strong className="text-slate-300">{result.prediction.cv_threshold}</strong> (OOF cross-validation selected)
                </p>
              </div>

              {/* Probability */}
              <div>
                <div className="flex justify-between text-xs text-slate-400 mb-1.5">
                  <span>Raw default probability</span>
                  <span className="font-mono text-slate-200">
                    {((prob ?? 0) * 100).toFixed(2)}%
                  </span>
                </div>
                <div className="w-full bg-slate-800 rounded-full h-2.5 overflow-hidden">
                  <div
                    className={`h-2.5 rounded-full transition-all ${isHighRisk ? 'bg-rose-500' : 'bg-emerald-500'}`}
                    style={{ width: `${Math.min((prob ?? 0) * 100, 100)}%` }}
                  />
                </div>
                {/* Threshold marker */}
                <div className="relative h-1 mt-0.5">
                  <div
                    className="absolute top-0 w-0.5 h-3 bg-amber-400 rounded"
                    style={{ left: `${result.prediction.cv_threshold * 100}%` }}
                    title={`CV threshold: ${result.prediction.cv_threshold}`}
                  />
                </div>
                <p className="text-[10px] text-amber-400 mt-2">
                  ↑ Amber marker = CV threshold ({result.prediction.cv_threshold})
                </p>
              </div>

              {/* Probability calibration warning */}
              <div className="rounded-lg bg-slate-800/60 border border-rose-700/20 p-3">
                <p className="text-[11px] text-rose-300 font-semibold mb-1">Probability Label</p>
                <p className="text-[10px] text-slate-400 leading-relaxed">
                  {result.prediction.probability_label}
                </p>
              </div>

              {/* Governance warnings */}
              <div className="space-y-2">
                <p className="text-xs font-semibold text-slate-300 flex items-center gap-1.5">
                  <AlertTriangle className="h-3.5 w-3.5 text-amber-400" />
                  Governance Warnings
                </p>
                <ul className="space-y-1">
                  {result.governance.warnings.map((w, i) => (
                    <li key={i} className="flex gap-2 text-[11px] text-slate-400">
                      <span className="text-amber-400 shrink-0">•</span>
                      <span>{w}</span>
                    </li>
                  ))}
                </ul>
              </div>

              {/* Dataset caveat */}
              <div className="rounded-lg bg-slate-800/40 border border-slate-700/40 p-3">
                <p className="text-[10px] text-slate-500 leading-relaxed">
                  <strong className="text-slate-400">Dataset caveat:</strong>{' '}
                  {result.model.dataset_caveat}
                </p>
              </div>

              {/* Note */}
              <p className="text-[10px] text-slate-500 italic text-center">
                {result.note}
              </p>
            </div>
          ) : (
            <div className="rounded-xl border border-slate-700/40 bg-slate-900/50 p-8 flex flex-col items-center gap-3 text-center">
              <FlaskConical className="h-10 w-10 text-slate-600" />
              <p className="text-sm text-slate-500">
                Fill in the form and click <strong className="text-slate-400">Run Research Prediction</strong> to see an experimental result.
              </p>
              <p className="text-xs text-slate-600">Results are experimental research only.</p>
            </div>
          )}

          {/* SHAP global importance */}
          {shap.length > 0 && (
            <div className="rounded-xl border border-slate-700/60 bg-slate-900 p-5 space-y-3">
              <h3 className="text-sm font-semibold text-slate-200 flex items-center gap-2">
                <Info className="h-4 w-4 text-blue-400" />
                Global Feature Importance
                <span className="text-[10px] text-slate-500 font-normal">(mean |SHAP| · association only)</span>
              </h3>
              <div className="space-y-2">
                {shap.slice(0, 10).map((f) => {
                  const maxShap = shap[0].mean_abs_shap;
                  const pct = (f.mean_abs_shap / maxShap) * 100;
                  const isFairness = f.feature === 'gender';
                  return (
                    <div key={f.feature} className="space-y-0.5">
                      <div className="flex justify-between items-center">
                        <span className={`text-xs ${isFairness ? 'text-orange-300 font-semibold' : 'text-slate-300'}`}>
                          {isFairness && <Flag className="inline h-3 w-3 mr-1 text-orange-400" />}
                          {f.feature}
                          {isFairness && <span className="text-[9px] text-orange-400 ml-1">[fairness flag]</span>}
                        </span>
                        <span className="text-[10px] text-slate-500 font-mono">{f.mean_abs_shap.toFixed(3)}</span>
                      </div>
                      <div className="w-full bg-slate-800 rounded-full h-1.5">
                        <div className={`h-1.5 rounded-full ${isFairness ? 'bg-orange-400' : 'bg-blue-500'}`}
                             style={{ width: `${pct}%` }} />
                      </div>
                      <p className="text-[9px] text-slate-600">{f.direction}</p>
                    </div>
                  );
                })}
              </div>
              <p className="text-[10px] text-slate-600 italic">
                SHAP values reflect model associations, not causal relationships.
                Computed on 2,000 test samples in Stage 3C.
              </p>
            </div>
          )}

          {/* Model identity card */}
          <div className="rounded-xl border border-slate-700/60 bg-slate-900 p-4 space-y-2">
            <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Model Identity</h3>
            {[
              ['Name',      meta?.model.name ?? 'indian_loan_logistic_regression'],
              ['Version',   meta?.model.version ?? '3B-experimental'],
              ['Type',      'Logistic Regression (C=0.1, class_weight=balanced)'],
              ['Threshold', `${meta?.model.cv_threshold ?? 0.7623} (OOF CV)`],
              ['Features',  `${meta?.model.features_used ?? 19} baseline features`],
              ['Excluded',  (meta?.model.excluded ?? ['applicant_id', 'default_risk_score']).join(', ')],
              ['Status',    meta?.model.experiment_label ?? 'EXPERIMENTAL'],
            ].map(([k, v]) => (
              <div key={k} className="flex justify-between items-start gap-2">
                <span className="text-[11px] text-slate-500 shrink-0">{k}</span>
                <span className="text-[11px] text-slate-300 text-right font-mono">{v}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};
