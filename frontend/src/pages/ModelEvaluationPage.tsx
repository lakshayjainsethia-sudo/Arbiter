import React from 'react';
import { useQuery } from '@tanstack/react-query';
import { arbiterApi } from '../api/arbiterApi';
import { LoadingState } from '../components/common/LoadingState';
import { ErrorState } from '../components/common/ErrorState';
import { MetricCard } from '../components/common/MetricCard';
import {
  FileCheck2,
  AlertTriangle,
  Scale,
  Info,
} from 'lucide-react';

export const ModelEvaluationPage: React.FC = () => {
  const {
    data: metrics,
    isLoading,
    error,
    refetch,
  } = useQuery({
    queryKey: ['metrics'],
    queryFn: arbiterApi.getMetrics,
  });

  if (isLoading) {
    return <LoadingState message="Loading Model Evaluation Benchmarks..." />;
  }

  if (error && !metrics) {
    return (
      <ErrorState
        title="Could Not Load Evaluation Metrics"
        message="Failed to retrieve metrics artifact from the backend."
        onRetry={refetch}
      />
    );
  }

  const clean2k = metrics?.robustness_sample_clean || {
    roc_auc: 0.6679,
    pr_auc: 0.3318,
    precision: 0.2948,
    recall: 0.591,
    f1: 0.3934,
    review_count: 486,
    review_rate: 0.243,
    reviewed_defaults: 98,
    automatic_false_negatives: 117,
    original_false_negatives: 164,
  };

  const clean10k = metrics?.full_test_baseline;
  const ablation = metrics?.zip_ablation_comparison || [];

  return (
    <div className="space-y-8">
      {/* Overview Context Banner */}
      <div className="rounded-xl border border-slate-200 bg-white p-6 shadow-xs">
        <div className="flex items-center gap-2 mb-1">
          <span className="text-xs font-bold uppercase tracking-wider text-blue-700">
            Model Benchmarks & Governance
          </span>
          <span className="rounded-md bg-blue-100 px-2 py-0.5 text-[11px] font-semibold text-blue-800">
            Research Evaluation
          </span>
        </div>
        <h2 className="text-xl font-bold text-slate-900">
          Model Evaluation & Statistical Metrics
        </h2>
        <p className="mt-1 text-xs text-slate-600 max-w-3xl leading-relaxed">
          Comprehensive review of the XGBoost loan default prediction system across evaluation test partitions. Accurately reports discrimination metrics, confusion distributions, ablation differences, and research limitations.
        </p>
      </div>

      {/* Clean 2,000 Sample Metrics Grid */}
      <div>
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-sm font-semibold text-slate-800 uppercase tracking-wider">
            Primary Robustness Sample Metrics (N = 2,000)
          </h3>
          <span className="text-xs font-mono text-slate-500">
            Sample with 401 Defaults (20.05% Prevalence)
          </span>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-4">
          <MetricCard
            title="ROC-AUC"
            value={clean2k.roc_auc.toFixed(4)}
            subtitle="Area under ROC Curve"
            badge={{ text: 'Discriminative', variant: 'neutral' }}
          />
          <MetricCard
            title="PR-AUC"
            value={clean2k.pr_auc.toFixed(4)}
            subtitle="Average Precision"
            badge={{ text: 'Imbalanced', variant: 'neutral' }}
          />
          <MetricCard
            title="Default Precision"
            value={(clean2k.precision * 100).toFixed(2) + '%'}
            subtitle="TP / (TP + FP) at 0.50 threshold"
            badge={{ text: '0.50 cutoff', variant: 'neutral' }}
          />
          <MetricCard
            title="Default Recall"
            value={(clean2k.recall * 100).toFixed(2) + '%'}
            subtitle="TP / Actual Defaults"
            badge={{ text: '59.10%', variant: 'positive' }}
          />
          <MetricCard
            title="Default F1 Score"
            value={clean2k.f1.toFixed(4)}
            subtitle="Harmonic mean of precision & recall"
            badge={{ text: 'Balanced F1', variant: 'neutral' }}
          />
        </div>
      </div>

      {/* Sample Size Clarification Callout */}
      <div className="rounded-xl border border-blue-200 bg-blue-50/60 p-5 text-xs text-blue-950 space-y-2">
        <div className="flex items-center gap-2 font-bold text-blue-900">
          <Info className="h-4 w-4 text-blue-700" />
          Critical Distinction: 2,000-Row Sample vs 10,000-Row Evaluation Partition
        </div>
        <p className="leading-relaxed text-blue-900/90">
          The project evaluated the XGBoost model across two distinct sample sizes:
        </p>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 pt-1">
          <div className="rounded-lg bg-white p-3 border border-blue-200 text-slate-800">
            <span className="font-bold block text-blue-900">Final Robustness Test Sample (2,000 rows)</span>
            <div className="mt-1 font-mono text-[11px] text-slate-600 space-y-0.5">
              <div>ROC-AUC: <strong>0.6679</strong> &bull; PR-AUC: <strong>0.3318</strong></div>
              <div>Defaults: 401 &bull; Review Count: 486 (24.3%)</div>
              <div>Used for multi-scenario perturbation stress testing.</div>
            </div>
          </div>
          <div className="rounded-lg bg-white p-3 border border-blue-200 text-slate-800">
            <span className="font-bold text-blue-900 block">Full Test Baseline Partition (10,000 rows)</span>
            <div className="mt-1 font-mono text-[11px] text-slate-600 space-y-0.5">
              <div>
                ROC-AUC: <strong>{clean10k?.roc_auc.toFixed(4) || '0.6744'}</strong> &bull; PR-AUC:{' '}
                <strong>{clean10k?.pr_auc.toFixed(4) || '0.3403'}</strong>
              </div>
              <div>
                Default Precision: {((clean10k?.default_precision ?? 0.2991) * 100).toFixed(2)}% &bull; Recall:{' '}
                {((clean10k?.default_recall ?? 0.6121) * 100).toFixed(2)}%
              </div>
              <div>Standard 20% test partition of the 50,000 training dataset ({clean10k?.positive_predictions ?? 4179} positive predictions).</div>
            </div>
          </div>
        </div>
        <p className="text-[11px] text-blue-800">
          These represent distinct evaluation subsets and must not be conflated as conflicting runs of identical size.
        </p>
      </div>

      {/* Decision Routing Matrix (Confusion Breakdown under Review Policy) */}
      <div className="rounded-xl border border-slate-200 bg-white p-6 shadow-xs space-y-4">
        <div className="flex items-center gap-2 border-b border-slate-100 pb-3">
          <Scale className="h-4 w-4 text-blue-600" />
          <h3 className="text-base font-bold text-slate-900">
            Review-Policy Decision Breakdown (2,000-Row Benchmark)
          </h3>
        </div>

        <p className="text-xs text-slate-600">
          How the 2,000 applicant cases are distributed across decisions when the [0.45, 0.55] uncertainty band is enforced:
        </p>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 text-xs font-mono">
          <div className="rounded-lg bg-emerald-50 p-4 border border-emerald-200">
            <span className="font-sans font-bold text-emerald-800 block mb-1">
              Automatic Low Risk (&lt; 0.45)
            </span>
            <div className="text-xl font-bold text-emerald-900">1,159 Loans</div>
            <div className="text-[11px] text-emerald-700 mt-1 font-sans">
              1,042 True Negatives + 117 Auto False Negatives
            </div>
          </div>

          <div className="rounded-lg bg-amber-50 p-4 border border-amber-200">
            <span className="font-sans font-bold text-amber-800 block mb-1">
              Manual Review Band [0.45 – 0.55]
            </span>
            <div className="text-xl font-bold text-amber-900">486 Loans (24.3%)</div>
            <div className="text-[11px] text-amber-700 mt-1 font-sans">
              <strong>98 Defaults</strong> + 388 Non-defaults routed to human underwriter
            </div>
          </div>

          <div className="rounded-lg bg-rose-50 p-4 border border-rose-200">
            <span className="font-sans font-bold text-rose-800 block mb-1">
              Automatic High Risk (&gt; 0.55)
            </span>
            <div className="text-xl font-bold text-rose-900">355 Loans</div>
            <div className="text-[11px] text-rose-700 mt-1 font-sans">
              186 True Positives + 169 False Positives
            </div>
          </div>
        </div>

        <div className="rounded-lg bg-slate-50 p-3 text-xs text-slate-600">
          <strong>Impact of Review Policy:</strong> Routing 486 loans (24.3%) to human underwriters caught <strong>98 defaults</strong> that were previously vulnerable to borderline misclassification, reducing automatic false negatives from 164 to 117.
        </div>
      </div>

      {/* ZIP Code Ablation Comparison Table */}
      <div className="rounded-xl border border-slate-200 bg-white p-6 shadow-xs space-y-4">
        <div className="flex items-center gap-2 border-b border-slate-100 pb-3">
          <FileCheck2 className="h-4 w-4 text-indigo-600" />
          <h3 className="text-base font-bold text-slate-900">
            ZIP-Code Exclusion Ablation Benchmark
          </h3>
        </div>

        <div className="overflow-x-auto rounded-lg border border-slate-200">
          <table className="w-full text-left text-xs">
            <thead className="bg-slate-50 text-slate-700 border-b border-slate-200">
              <tr>
                <th className="px-4 py-3 font-semibold">Model Experiment</th>
                <th className="px-4 py-3 font-semibold">ROC-AUC</th>
                <th className="px-4 py-3 font-semibold">PR-AUC</th>
                <th className="px-4 py-3 font-semibold">Default Precision</th>
                <th className="px-4 py-3 font-semibold">Default Recall</th>
                <th className="px-4 py-3 font-semibold">Default F1</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 font-mono text-slate-800">
              {ablation.map((row, idx) => (
                <tr key={row.model} className={idx === 1 ? 'bg-blue-50/50 font-bold' : ''}>
                  <td className="px-4 py-3 font-sans text-slate-900">{row.model}</td>
                  <td className="px-4 py-3">{row.roc_auc.toFixed(6)}</td>
                  <td className="px-4 py-3">{row.pr_auc.toFixed(6)}</td>
                  <td className="px-4 py-3">{row.default_precision.toFixed(6)}</td>
                  <td className="px-4 py-3">{row.default_recall.toFixed(6)}</td>
                  <td className="px-4 py-3">{row.default_f1.toFixed(6)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Documented Research Limitations */}
      <div className="rounded-xl border border-slate-200 bg-white p-6 shadow-xs space-y-4">
        <div className="flex items-center gap-2 border-b border-slate-100 pb-3">
          <AlertTriangle className="h-4 w-4 text-amber-600" />
          <h3 className="text-base font-bold text-slate-900">
            Documented Research Limitations & Non-Claims
          </h3>
        </div>

        <div className="space-y-2.5 text-xs text-slate-700">
          <div className="flex items-start gap-2">
            <span className="flex h-4 w-4 shrink-0 items-center justify-center rounded-full bg-amber-100 text-[10px] font-bold text-amber-800">
              1
            </span>
            <p>
              <strong>Sample Subset Limitation:</strong> Evaluations use a stratified subset (50,000 rows training, 2,000 rows robustness sample) from the historical LendingClub dataset. Production deployment requires out-of-time validation across full macroeconomic cycles.
            </p>
          </div>

          <div className="flex items-start gap-2">
            <span className="flex h-4 w-4 shrink-0 items-center justify-center rounded-full bg-amber-100 text-[10px] font-bold text-amber-800">
              2
            </span>
            <p>
              <strong>Random Splitting vs Temporal Dynamics:</strong> Random stratified splitting does not measure temporal distribution shift or macroeconomic shocks (such as interest rate cycles or inflation recessions).
            </p>
          </div>

          <div className="flex items-start gap-2">
            <span className="flex h-4 w-4 shrink-0 items-center justify-center rounded-full bg-amber-100 text-[10px] font-bold text-amber-800">
              3
            </span>
            <p>
              <strong>Local Sensitivity vs Mathematical Robustness:</strong> The evaluated input perturbations represent controlled sensitivity testing (e.g. ±5 FICO points, ±5% income). They do not constitute certified mathematical adversarial robustness guarantees.
            </p>
          </div>

          <div className="flex items-start gap-2">
            <span className="flex h-4 w-4 shrink-0 items-center justify-center rounded-full bg-amber-100 text-[10px] font-bold text-amber-800">
              4
            </span>
            <p>
              <strong>Geographic Fairness Auditing Required:</strong> While removing 3-digit ZIP codes mitigates localized proxy redlining, state-level features still register prominent SHAP weights. Dedicated demographic fairness auditing is essential.
            </p>
          </div>

          <div className="flex items-start gap-2">
            <span className="flex h-4 w-4 shrink-0 items-center justify-center rounded-full bg-amber-100 text-[10px] font-bold text-amber-800">
              5
            </span>
            <p>
              <strong>Decision Support Prototype:</strong> ARBITER is designed exclusively as an underwriting decision-support and interpretability demonstrator. It must never be used as an autonomous, unmonitored credit rejection engine.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};
