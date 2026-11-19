import React, { useState } from 'react';
import { useQuery, useMutation } from '@tanstack/react-query';
import { arbiterApi } from '../api/arbiterApi';
import type { LoanApplicationInput, LoanPredictionResponse } from '../types/api';
import { LoadingState } from '../components/common/LoadingState';
import { ErrorState } from '../components/common/ErrorState';
import { formatPercent, getDecisionMeta } from '../utils/formatters';
import {
  FlaskConical,
  Filter,
  Sliders,
  RefreshCw,
  Info,
  Lightbulb,
} from 'lucide-react';
import {
  BarChart,
  Bar,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';

export const RobustnessPage: React.FC = () => {
  const [featureFilter, setFeatureFilter] = useState<string>('all');
  const [directionFilter, setDirectionFilter] = useState<string>('all');

  // Benchmark applicant #0 baseline
  const baseSandboxInput: LoanApplicationInput = {
    revenue: 52000,
    dti_n: 5.7,
    loan_amnt: 11000,
    fico_n: 662,
    experience_c: 1,
    emp_length: '5 years',
    purpose: 'debt_consolidation',
    home_ownership_n: 'MORTGAGE',
    addr_state: 'CA',
  };

  const [sandboxFeature, setSandboxFeature] = useState<'fico_n' | 'dti_n' | 'revenue' | 'loan_amnt'>('fico_n');
  const [sandboxDelta, setSandboxDelta] = useState<number>(-5);

  const [cleanPrediction, setCleanPrediction] = useState<LoanPredictionResponse | null>(null);
  const [perturbedPrediction, setPerturbedPrediction] = useState<LoanPredictionResponse | null>(null);

  const {
    data: robustness,
    isLoading,
    error,
    refetch,
  } = useQuery({
    queryKey: ['robustness'],
    queryFn: arbiterApi.getRobustness,
  });

  // Mutation to test perturbed applicant in the sandbox
  const sandboxMutation = useMutation({
    mutationFn: async () => {
      // 1. Predict clean
      const clean = await arbiterApi.predictLoan(baseSandboxInput);

      // 2. Predict perturbed
      const perturbedInput: LoanApplicationInput = {
        ...baseSandboxInput,
        [sandboxFeature]: Math.max(0, Number(baseSandboxInput[sandboxFeature]) + sandboxDelta),
      };
      const perturbed = await arbiterApi.predictLoan(perturbedInput);

      return { clean, perturbed };
    },
    onSuccess: (data) => {
      setCleanPrediction(data.clean);
      setPerturbedPrediction(data.perturbed);
    },
  });

  const handleApplySandboxPreset = (feature: 'fico_n' | 'dti_n' | 'revenue' | 'loan_amnt', delta: number) => {
    setSandboxFeature(feature);
    setSandboxDelta(delta);
    setTimeout(() => {
      sandboxMutation.mutate();
    }, 50);
  };

  if (isLoading) {
    return <LoadingState message="Loading Robustness Stress Test Artifacts..." />;
  }

  if (error && !robustness) {
    return (
      <ErrorState
        title="Could Not Load Robustness Data"
        message="Failed to retrieve final_robustness_comparison.csv from ARBITER backend."
        onRetry={refetch}
      />
    );
  }

  const rawScenarios = robustness?.comparison || [];

  // Filter table rows
  const filteredScenarios = rawScenarios.filter((row) => {
    if (featureFilter !== 'all') {
      if (featureFilter === 'clean' && row.scenario !== 'clean') return false;
      if (featureFilter !== 'clean' && !row.scenario.startsWith(featureFilter)) return false;
    }
    if (directionFilter !== 'all') {
      if (directionFilter === 'decrease' && !row.scenario.includes('decrease')) return false;
      if (directionFilter === 'increase' && !row.scenario.includes('increase')) return false;
    }
    return true;
  });

  // Chart dataset
  const chartData = rawScenarios.map((row) => ({
    scenario: row.scenario
      .replace('clean', 'Clean')
      .replace('_decrease', ' (-)')
      .replace('_increase', ' (+)'),
    roc_auc: Number(row.roc_auc.toFixed(4)),
    pr_auc: Number(row.pr_auc.toFixed(4)),
    recall: Number((row.recall * 100).toFixed(2)),
    review_rate: Number((row.review_rate * 100).toFixed(2)),
    flip_rate: Number((row.class_flip_rate_vs_clean * 100).toFixed(2)),
    reviewed_defaults: row.reviewed_defaults,
    auto_fn: row.automatic_false_negatives,
  }));

  const cleanMeta = cleanPrediction ? getDecisionMeta(cleanPrediction.decision) : null;
  const perturbedMeta = perturbedPrediction ? getDecisionMeta(perturbedPrediction.decision) : null;

  // Unrounded probability difference for functional precision
  const rawProbDiff =
    cleanPrediction && perturbedPrediction
      ? perturbedPrediction.default_probability - cleanPrediction.default_probability
      : 0;

  return (
    <div className="space-y-8">
      {/* Overview Context Banner */}
      <div className="rounded-xl border border-blue-100 bg-gradient-to-r from-blue-50/70 via-slate-50 to-white p-6 shadow-xs">
        <div className="flex items-center gap-2 mb-1">
          <span className="text-xs font-bold uppercase tracking-wider text-blue-700">
            Adversarial & Empirical Stress Testing
          </span>
          <span className="rounded-md bg-blue-100 px-2 py-0.5 text-[11px] font-semibold text-blue-800">
            17 Perturbation Scenarios
          </span>
        </div>
        <h2 className="text-xl font-bold text-slate-900">
          Robustness Lab & Sensitivity Benchmarks
        </h2>
        <p className="mt-1 text-xs text-slate-600 max-w-3xl leading-relaxed">
          Evaluate how XGBoost decision boundaries and the [0.45, 0.55] manual-review policy buffer against systematic feature drift, noise perturbations, and adverse borrower shifts across 2,000 evaluation loans.
        </p>
      </div>

      {/* Two Comparative Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Chart 1: Discrimination Across Perturbations */}
        <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-xs">
          <div className="mb-4">
            <h3 className="text-sm font-semibold text-slate-900">
              Discrimination Metrics (ROC-AUC vs PR-AUC)
            </h3>
            <p className="text-xs text-slate-500">
              Sensitivity across 17 perturbation scenarios vs clean baseline
            </p>
          </div>
          <div className="h-72 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={chartData}
                margin={{ top: 10, right: 10, left: -20, bottom: 40 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                <XAxis
                  dataKey="scenario"
                  angle={-30}
                  textAnchor="end"
                  interval={0}
                  tick={{ fontSize: 10, fill: '#64748b' }}
                />
                <YAxis domain={[0.25, 0.75]} tick={{ fontSize: 10, fill: '#64748b' }} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#ffffff',
                    borderColor: '#e2e8f0',
                    borderRadius: '8px',
                    fontSize: '12px',
                  }}
                />
                <Legend verticalAlign="top" height={36} iconSize={10} />
                <Bar dataKey="roc_auc" name="ROC-AUC" fill="#2563eb" radius={[4, 4, 0, 0]} />
                <Bar dataKey="pr_auc" name="PR-AUC" fill="#059669" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Chart 2: Flip Rate vs Review Defaults */}
        <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-xs">
          <div className="mb-4">
            <h3 className="text-sm font-semibold text-slate-900">
              Class Flip Rate vs Review Policy Interception
            </h3>
            <p className="text-xs text-slate-500">
              Rate of decision flips from clean baseline vs reviewed default volume
            </p>
          </div>
          <div className="h-72 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart
                data={chartData}
                margin={{ top: 10, right: 10, left: -20, bottom: 40 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                <XAxis
                  dataKey="scenario"
                  angle={-30}
                  textAnchor="end"
                  interval={0}
                  tick={{ fontSize: 10, fill: '#64748b' }}
                />
                <YAxis domain={[0, 5]} unit="%" tick={{ fontSize: 10, fill: '#64748b' }} />
                <Tooltip
                  formatter={(val: any) => [`${val}%`, 'Class-Flip Rate']}
                  contentStyle={{
                    backgroundColor: '#ffffff',
                    borderColor: '#e2e8f0',
                    borderRadius: '8px',
                    fontSize: '12px',
                  }}
                />
                <Legend verticalAlign="top" height={36} iconSize={10} />
                <Line
                  type="monotone"
                  dataKey="flip_rate"
                  name="Class-Flip Rate (%)"
                  stroke="#ef4444"
                  strokeWidth={2}
                  dot={{ r: 3 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Scenario Comparison Table with Filters */}
      <div className="rounded-xl border border-slate-200 bg-white p-6 shadow-xs space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-100 pb-3">
          <div>
            <h3 className="text-base font-bold text-slate-900 flex items-center gap-2">
              <Filter className="h-4 w-4 text-blue-600" />
              Detailed Perturbation Scenario Benchmarks
            </h3>
            <p className="text-xs text-slate-500">
              Exact verified values loaded directly from final_robustness_comparison.csv
            </p>
          </div>

          {/* Filters */}
          <div className="flex flex-wrap items-center gap-2">
            <select
              value={featureFilter}
              onChange={(e) => setFeatureFilter(e.target.value)}
              className="rounded-lg border border-slate-200 bg-white px-2.5 py-1 text-xs text-slate-700"
            >
              <option value="all">All Features</option>
              <option value="clean">Clean Only</option>
              <option value="fico_n">FICO</option>
              <option value="dti_n">DTI</option>
              <option value="revenue">Revenue</option>
              <option value="loan_amnt">Loan Amount</option>
            </select>

            <select
              value={directionFilter}
              onChange={(e) => setDirectionFilter(e.target.value)}
              className="rounded-lg border border-slate-200 bg-white px-2.5 py-1 text-xs text-slate-700"
            >
              <option value="all">All Directions</option>
              <option value="decrease">Decrease (-)</option>
              <option value="increase">Increase (+)</option>
            </select>
          </div>
        </div>

        {/* Table */}
        <div className="overflow-x-auto rounded-lg border border-slate-200">
          <table className="w-full text-left text-xs">
            <thead className="bg-slate-50 text-slate-700 border-b border-slate-200 sticky top-0">
              <tr>
                <th className="px-4 py-3 font-semibold">Scenario</th>
                <th className="px-4 py-3 font-semibold">ROC-AUC</th>
                <th className="px-4 py-3 font-semibold">PR-AUC</th>
                <th className="px-4 py-3 font-semibold">Recall</th>
                <th className="px-4 py-3 font-semibold">Review Rate</th>
                <th className="px-4 py-3 font-semibold">Reviewed Defaults</th>
                <th className="px-4 py-3 font-semibold">Auto False Negatives</th>
                <th className="px-4 py-3 font-semibold">Class-Flip Rate</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 text-slate-800 font-mono">
              {filteredScenarios.map((row) => {
                const isClean = row.scenario === 'clean';
                return (
                  <tr
                    key={row.scenario}
                    className={isClean ? 'bg-blue-50/50 font-bold' : 'hover:bg-slate-50/50'}
                  >
                    <td className="px-4 py-3 font-sans capitalize text-slate-900">
                      {row.scenario.replace(/_/g, ' ')}
                      {isClean && (
                        <span className="ml-2 rounded bg-blue-100 text-blue-800 px-1.5 py-0.5 text-[10px] font-sans">
                          Baseline
                        </span>
                      )}
                    </td>
                    <td className="px-4 py-3">{row.roc_auc.toFixed(4)}</td>
                    <td className="px-4 py-3">{row.pr_auc.toFixed(4)}</td>
                    <td className="px-4 py-3">{(row.recall * 100).toFixed(2)}%</td>
                    <td className="px-4 py-3">{(row.review_rate * 100).toFixed(2)}%</td>
                    <td className="px-4 py-3 text-emerald-700">{row.reviewed_defaults}</td>
                    <td className="px-4 py-3 text-rose-600">{row.automatic_false_negatives}</td>
                    <td className="px-4 py-3">
                      {isClean ? '0.00%' : `${(row.class_flip_rate_vs_clean * 100).toFixed(2)}%`}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Interactive Sensitivity Sandbox */}
      <div className="rounded-xl border border-slate-200 bg-white p-6 shadow-xs space-y-4">
        <div className="flex items-center justify-between border-b border-slate-100 pb-3">
          <div className="flex items-center gap-2">
            <Sliders className="h-4 w-4 text-blue-600" />
            <h3 className="text-base font-bold text-slate-900">
              Interactive Single-Input Perturbation Sandbox
            </h3>
          </div>
          <span className="text-[11px] font-mono text-slate-400">
            Baseline: Applicant #0 (FICO 662, DTI 5.7%)
          </span>
        </div>

        <p className="text-xs text-slate-600 leading-relaxed">
          Apply a live feature perturbation to benchmark applicant #0 to observe decision tree split mechanics, probability sensitivity, and potential decision category flips.
        </p>

        {/* Quick Empirical Test Presets */}
        <div className="rounded-lg border border-slate-100 bg-slate-50 p-3 space-y-2">
          <div className="text-[11px] font-bold text-slate-700 uppercase tracking-wider flex items-center gap-1.5">
            <Lightbulb className="h-3.5 w-3.5 text-amber-500" />
            <span>Empirical Step-Function Demonstrator Presets</span>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-2 text-xs">
            <button
              type="button"
              onClick={() => handleApplySandboxPreset('fico_n', -5)}
              className="p-2 rounded-md bg-white border border-slate-200 hover:border-blue-400 text-left transition-colors cursor-pointer"
            >
              <div className="font-semibold text-slate-900">FICO -5 (Within Leaf)</div>
              <div className="text-[10px] text-slate-500">662 → 657 &bull; Same partition (&le; 672.0)</div>
            </button>

            <button
              type="button"
              onClick={() => handleApplySandboxPreset('fico_n', 20)}
              className="p-2 rounded-md bg-white border border-slate-200 hover:border-blue-400 text-left transition-colors cursor-pointer"
            >
              <div className="font-semibold text-slate-900 text-emerald-700">FICO +20 (Crosses Split)</div>
              <div className="text-[10px] text-slate-500">662 → 682 &bull; Crosses 672.0 &bull; Risk &darr;</div>
            </button>

            <button
              type="button"
              onClick={() => handleApplySandboxPreset('dti_n', 5.0)}
              className="p-2 rounded-md bg-white border border-slate-200 hover:border-blue-400 text-left transition-colors cursor-pointer"
            >
              <div className="font-semibold text-slate-900 text-amber-700">DTI +5.0 (Boundary Flip)</div>
              <div className="text-[10px] text-slate-500">5.7 &rarr; 10.7 &bull; Flips to Manual Review</div>
            </button>

            <button
              type="button"
              onClick={() => handleApplySandboxPreset('loan_amnt', 2000)}
              className="p-2 rounded-md bg-white border border-slate-200 hover:border-blue-400 text-left transition-colors cursor-pointer"
            >
              <div className="font-semibold text-slate-900 text-rose-700">Loan +$2,000 (Stress)</div>
              <div className="text-[10px] text-slate-500">$11k &rarr; $13k &bull; Jumps to 54.00%</div>
            </button>
          </div>
        </div>

        {/* Custom Sandbox Controls */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 pt-1">
          <div>
            <label className="block text-xs font-semibold text-slate-700 mb-1">
              Select Feature to Perturb
            </label>
            <select
              value={sandboxFeature}
              onChange={(e) => {
                const feat = e.target.value as any;
                setSandboxFeature(feat);
                if (feat === 'fico_n') setSandboxDelta(-5);
                else if (feat === 'dti_n') setSandboxDelta(0.5);
                else if (feat === 'revenue') setSandboxDelta(-2600);
                else if (feat === 'loan_amnt') setSandboxDelta(600);
              }}
              className="w-full rounded-lg border border-slate-300 px-3 py-2 text-xs text-slate-900 bg-white"
            >
              <option value="fico_n">FICO Score</option>
              <option value="dti_n">Debt-to-Income (DTI %)</option>
              <option value="revenue">Annual Revenue ($)</option>
              <option value="loan_amnt">Loan Amount ($)</option>
            </select>
          </div>

          <div>
            <label className="block text-xs font-semibold text-slate-700 mb-1">
              Perturbation Delta (+ / -)
            </label>
            <input
              type="number"
              value={sandboxDelta}
              onChange={(e) => setSandboxDelta(Number(e.target.value))}
              className="w-full rounded-lg border border-slate-300 px-3 py-2 text-xs text-slate-900 font-mono"
            />
          </div>

          <div className="flex items-end">
            <button
              type="button"
              onClick={() => sandboxMutation.mutate()}
              disabled={sandboxMutation.isPending}
              className="w-full rounded-lg bg-blue-600 py-2.5 text-xs font-bold text-white shadow-xs hover:bg-blue-700 disabled:opacity-50 transition-colors flex items-center justify-center gap-2 cursor-pointer"
            >
              {sandboxMutation.isPending ? (
                <>
                  <RefreshCw className="h-3.5 w-3.5 animate-spin" />
                  <span>Evaluating Live...</span>
                </>
              ) : (
                <>
                  <FlaskConical className="h-3.5 w-3.5" />
                  <span>Test Perturbation</span>
                </>
              )}
            </button>
          </div>
        </div>

        {/* Live Sandbox Comparison Results */}
        {cleanPrediction && perturbedPrediction && cleanMeta && perturbedMeta && (
          <div className="mt-4 rounded-lg border border-slate-200 bg-slate-50/70 p-4 space-y-3">
            <div className="text-xs font-bold text-slate-800">
              Live Prediction Comparison (Benchmark Loan #0)
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 text-xs">
              {/* Clean */}
              <div className="rounded-lg border border-slate-200 bg-white p-3 space-y-2">
                <div className="text-[11px] font-semibold text-slate-500 uppercase">
                  Clean Baseline Input
                </div>
                <div className="flex items-baseline justify-between">
                  <span className="text-slate-600">Default Probability:</span>
                  <span className="text-base font-bold font-mono text-slate-900">
                    {formatPercent(cleanPrediction.default_probability, 4)}
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-slate-600">Routing Decision:</span>
                  <span className={`px-2 py-0.5 rounded text-[11px] font-semibold ${cleanMeta.badgeBg} ${cleanMeta.badgeText}`}>
                    {cleanMeta.label}
                  </span>
                </div>
              </div>

              {/* Perturbed */}
              <div className="rounded-lg border border-slate-200 bg-white p-3 space-y-2">
                <div className="text-[11px] font-semibold text-slate-500 uppercase">
                  Perturbed Input ({sandboxFeature} {sandboxDelta > 0 ? `+${sandboxDelta}` : sandboxDelta})
                </div>
                <div className="flex items-baseline justify-between">
                  <span className="text-slate-600">Default Probability:</span>
                  <span className="text-base font-bold font-mono text-slate-900">
                    {formatPercent(perturbedPrediction.default_probability, 4)}
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-slate-600">Routing Decision:</span>
                  <span className={`px-2 py-0.5 rounded text-[11px] font-semibold ${perturbedMeta.badgeBg} ${perturbedMeta.badgeText}`}>
                    {perturbedMeta.label}
                  </span>
                </div>
              </div>
            </div>

            {/* Functional Precision & Educational ML Invariance Callout */}
            <div className="rounded-lg border border-blue-200 bg-blue-50/70 p-3 text-xs text-blue-900 space-y-1.5">
              <div className="flex items-center justify-between">
                <span className="font-semibold flex items-center gap-1.5">
                  <Info className="h-4 w-4 text-blue-700" />
                  <span>Exact Unrounded Delta:</span>
                </span>
                <span className="font-mono font-bold text-sm text-blue-950">
                  Δ = {(rawProbDiff * 100).toFixed(4)}% ({rawProbDiff >= 0 ? '+' : ''}{rawProbDiff.toFixed(6)})
                </span>
              </div>

              {rawProbDiff === 0 ? (
                <p className="text-[11px] text-blue-800 leading-relaxed pt-1 border-t border-blue-200/60">
                  <strong>Decision Tree Partition Invariance:</strong> Gradient-boosted decision trees partition continuous feature space into orthogonal step functions. Because applicant #0&apos;s clean FICO is 662 and the nearest tree split threshold above it is <strong>672.00</strong> (as verified in LIME rule <code className="font-mono">fico_n &le; 672.00</code>), subtracting 5 points leaves all tree leaf assignments completely unchanged. Test <strong>FICO +20</strong> or <strong>DTI +5.0</strong> to cross partition thresholds!
                </p>
              ) : (
                <p className="text-[11px] text-blue-800 leading-relaxed pt-1 border-t border-blue-200/60">
                  <strong>Threshold Boundary Crossed:</strong> This perturbation altered the leaf path across one or more decision trees in the ensemble, shifting the default probability by <strong>{(rawProbDiff * 100).toFixed(4)}%</strong>.
                  {cleanPrediction.decision !== perturbedPrediction.decision && (
                    <span className="ml-1 font-bold text-amber-900">
                      (Decision policy flipped from &apos;{cleanPrediction.decision}&apos; to &apos;{perturbedPrediction.decision}&apos;!)
                    </span>
                  )}
                </p>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
