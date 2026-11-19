import React, { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { arbiterApi } from '../api/arbiterApi';
import { LoadingState } from '../components/common/LoadingState';
import { ErrorState } from '../components/common/ErrorState';
import {
  BarChart2,
  AlertTriangle,
  Info,
  Layers,
  Scale,
} from 'lucide-react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from 'recharts';

export const ExplainabilityPage: React.FC = () => {
  // Default to Top 10 features as requested in the design spec
  const [featureLimit, setFeatureLimit] = useState<number>(10);
  const [activeTab, setActiveTab] = useState<'shap' | 'lime'>('shap');

  const {
    data: globalShap,
    isLoading: globalLoading,
    error: globalError,
  } = useQuery({
    queryKey: ['explainability', 'global'],
    queryFn: arbiterApi.getGlobalExplainability,
  });

  const {
    data: localExample,
    isLoading: localLoading,
  } = useQuery({
    queryKey: ['explainability', 'local', 'example'],
    queryFn: arbiterApi.getLocalExampleExplainability,
  });

  if (globalLoading || localLoading) {
    return <LoadingState message="Loading Model Explainability Artifacts..." />;
  }

  if (globalError && !globalShap) {
    return (
      <ErrorState
        title="Could Not Load Explainability Data"
        message="The global SHAP feature importance artifacts could not be retrieved from the server."
      />
    );
  }

  // Process global features for horizontal bar chart (Top N sorted descending)
  const processedGlobal = (globalShap?.features || [])
    .slice(0, featureLimit)
    .map((item) => {
      const cleanName = item.feature
        .replace('categorical__', '')
        .replace('numeric__', '')
        .replace(/_/g, ' ');

      const isGeo = item.feature.includes('addr_state');
      return {
        raw: item.feature,
        name: cleanName,
        importance: Number(item.mean_absolute_shap.toFixed(4)),
        isGeo,
      };
    })
    .reverse(); // Reverse so highest importance appears at top of vertical layout

  // Local benchmark SHAP contributions (top 15)
  const localShapData = (localExample?.shap_contributions || [])
    .slice(0, 15)
    .map((item) => ({
      name: item.feature
        .replace('categorical__', '')
        .replace('numeric__', '')
        .replace(/_/g, ' '),
      shap_value: Number(item.shap_value.toFixed(4)),
      absolute_shap: Number(item.absolute_shap.toFixed(4)),
      feature_value: item.feature_value,
    }))
    .reverse();

  // Local benchmark LIME conditions
  const localLimeData = localExample?.lime_weights || [];

  return (
    <div className="space-y-8">
      {/* Header Context Banner */}
      <div className="rounded-xl border border-indigo-100 bg-gradient-to-r from-indigo-50/70 via-slate-50 to-white p-6 shadow-xs">
        <div className="flex items-center gap-2 mb-1">
          <span className="text-xs font-bold uppercase tracking-wider text-indigo-700">
            Model Interpretability & Auditability
          </span>
          <span className="rounded-md bg-indigo-100 px-2 py-0.5 text-[11px] font-semibold text-indigo-800">
            TreeSHAP & LIME Tabular
          </span>
        </div>
        <h2 className="text-xl font-bold text-slate-900">
          Model Explainability Center
        </h2>
        <p className="mt-1 text-xs text-slate-600 max-w-3xl leading-relaxed">
          Examine feature attribution at both the population level (Global TreeSHAP) and individual loan applicant level (Local SHAP & LIME). Interpretability metrics quantify model dependency across fitted decision trees; they do not establish causal relationships.
        </p>
      </div>

      {/* Section 1: Global Feature Importance */}
      <div className="rounded-xl border border-slate-200 bg-white p-6 shadow-xs space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-100 pb-4">
          <div>
            <h3 className="text-base font-bold text-slate-900 flex items-center gap-2">
              <BarChart2 className="h-4 w-4 text-blue-600" />
              Global SHAP Feature Importance
            </h3>
            <p className="text-xs text-slate-500">
              Mean absolute SHAP value across representative LendingClub evaluation partition
            </p>
          </div>

          {/* Top 10 / 15 / 25 Selector */}
          <div className="flex items-center gap-2">
            <span className="text-xs font-medium text-slate-500">Display Top:</span>
            {[10, 15, 25].map((limit) => (
              <button
                key={limit}
                type="button"
                onClick={() => setFeatureLimit(limit)}
                className={`px-3 py-1 text-xs font-bold rounded-lg border transition-colors cursor-pointer ${
                  featureLimit === limit
                    ? 'border-blue-600 bg-blue-50 text-blue-700'
                    : 'border-slate-200 bg-white text-slate-600 hover:bg-slate-50'
                }`}
              >
                {limit}
              </button>
            ))}
          </div>
        </div>

        {/* Global SHAP Horizontal Bar Chart with generous left margin */}
        <div className="h-[420px] w-full pt-2">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              layout="vertical"
              data={processedGlobal}
              margin={{ top: 10, right: 30, left: 160, bottom: 10 }}
            >
              <CartesianGrid strokeDasharray="3 3" horizontal stroke="#f1f5f9" vertical={false} />
              <XAxis
                type="number"
                tick={{ fontSize: 11, fill: '#64748b' }}
                domain={[0, 'dataMax + 0.05']}
              />
              <YAxis
                type="category"
                dataKey="name"
                tick={{ fontSize: 11, fill: '#1e293b' }}
                interval={0}
                width={150}
              />
              <Tooltip
                formatter={(val: any) => [val, 'Mean |SHAP| Value']}
                contentStyle={{
                  backgroundColor: '#ffffff',
                  borderColor: '#e2e8f0',
                  borderRadius: '8px',
                  fontSize: '12px',
                }}
              />
              <Bar dataKey="importance" radius={[0, 4, 4, 0]}>
                {processedGlobal.map((entry, index) => (
                  <Cell
                    key={`cell-${index}`}
                    fill={entry.isGeo ? '#6366f1' : '#2563eb'}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Legend & Interpretation Callout */}
        <div className="flex items-center gap-4 text-xs pt-1 border-t border-slate-100">
          <div className="flex items-center gap-1.5">
            <span className="h-3 w-3 rounded-xs bg-blue-600 inline-block" />
            <span className="text-slate-600 font-medium">Financial & Credit Features (FICO, DTI, Loan Amount)</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="h-3 w-3 rounded-xs bg-indigo-500 inline-block" />
            <span className="text-slate-600 font-medium">Geographic Variables (US State One-Hot)</span>
          </div>
        </div>

        <div className="rounded-lg border border-slate-200 bg-slate-50 p-4 text-xs text-slate-600 flex items-start gap-3">
          <Info className="h-5 w-5 text-blue-600 shrink-0 mt-0.5" />
          <div className="space-y-1">
            <strong className="text-slate-800">Attribution vs Causation Advisory:</strong>
            <p>
              High global SHAP values indicate features that the tree ensemble relies on most heavily across split nodes to partition loan default outcomes. Prominent state variables (<span className="text-indigo-700 font-semibold">purple bars</span>) demonstrate that geographic proxies persist in tree branches even when ZIP codes are removed. High SHAP attribution does not indicate that applicant location causes loan default.
            </p>
          </div>
        </div>
      </div>

      {/* Section 2: Local Benchmark Explanations (Clearly separated and labeled) */}
      <div className="rounded-xl border border-slate-200 bg-white p-6 shadow-xs space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-100 pb-4">
          <div>
            <div className="flex items-center gap-2">
              <span className="text-xs font-bold uppercase tracking-wider text-slate-400">
                Benchmark Example: Applicant #0
              </span>
              <span className="rounded bg-slate-100 px-1.5 py-0.5 text-[10px] font-mono text-slate-600">
                LendingClub Reference Profile
              </span>
            </div>
            <h3 className="text-base font-bold text-slate-900 flex items-center gap-2 mt-0.5">
              <Layers className="h-4 w-4 text-indigo-600" />
              Local Loan Attribution (TreeSHAP vs LIME)
            </h3>
          </div>

          {/* Tab Selector */}
          <div className="flex items-center gap-1 rounded-lg border border-slate-200 bg-slate-50 p-1">
            <button
              type="button"
              onClick={() => setActiveTab('shap')}
              className={`rounded-md px-3 py-1 text-xs font-semibold transition-colors cursor-pointer ${
                activeTab === 'shap'
                  ? 'bg-white text-blue-700 shadow-xs'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              TreeSHAP Attributions
            </button>
            <button
              type="button"
              onClick={() => setActiveTab('lime')}
              className={`rounded-md px-3 py-1 text-xs font-semibold transition-colors cursor-pointer ${
                activeTab === 'lime'
                  ? 'bg-white text-blue-700 shadow-xs'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              LIME Rule Weights
            </button>
          </div>
        </div>

        {/* Tab 1: Local SHAP */}
        {activeTab === 'shap' && (
          <div className="space-y-4">
            <p className="text-xs text-slate-600">
              Additive SHAP feature values for benchmark applicant #0 (Baseline log-odds to predicted output). Features pushing probability higher appear in red; features lowering default probability appear in green.
            </p>

            <div className="h-80 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  layout="vertical"
                  data={localShapData}
                  margin={{ top: 5, right: 30, left: 150, bottom: 5 }}
                >
                  <CartesianGrid strokeDasharray="3 3" horizontal stroke="#f1f5f9" vertical={false} />
                  <XAxis type="number" tick={{ fontSize: 11, fill: '#64748b' }} />
                  <YAxis
                    type="category"
                    dataKey="name"
                    tick={{ fontSize: 11, fill: '#1e293b' }}
                    interval={0}
                    width={140}
                  />
                  <Tooltip
                    formatter={(val: any) => [val, 'SHAP Value (Log-odds contribution)']}
                    contentStyle={{
                      backgroundColor: '#ffffff',
                      borderColor: '#e2e8f0',
                      borderRadius: '8px',
                      fontSize: '12px',
                    }}
                  />
                  <Bar dataKey="shap_value" radius={4}>
                    {localShapData.map((entry, index) => (
                      <Cell
                        key={`cell-${index}`}
                        fill={entry.shap_value > 0 ? '#ef4444' : '#10b981'}
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}

        {/* Tab 2: LIME Local surrogate */}
        {activeTab === 'lime' && (
          <div className="space-y-4">
            <p className="text-xs text-slate-600">
              LIME linear surrogate decision rules fitted in the local perturbation neighborhood of benchmark applicant #0.
            </p>

            <div className="overflow-x-auto rounded-lg border border-slate-200">
              <table className="w-full text-left text-xs">
                <thead className="bg-slate-50 text-slate-700 border-b border-slate-200">
                  <tr>
                    <th className="px-4 py-2.5 font-semibold">Local Decision Rule</th>
                    <th className="px-4 py-2.5 font-semibold">LIME Weight</th>
                    <th className="px-4 py-2.5 font-semibold">Risk Impact</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100 font-mono text-slate-800">
                  {localLimeData.map((rule, idx) => {
                    const isRiskIncrease = rule.lime_weight > 0;
                    return (
                      <tr key={idx} className="hover:bg-slate-50/50">
                        <td className="px-4 py-2.5 font-sans text-slate-900">
                          {rule.feature_condition}
                        </td>
                        <td className="px-4 py-2.5 font-bold">
                          {rule.lime_weight.toFixed(4)}
                        </td>
                        <td className="px-4 py-2.5 font-sans">
                          <span
                            className={`px-2 py-0.5 rounded text-[11px] font-semibold ${
                              isRiskIncrease
                                ? 'bg-rose-50 text-rose-700'
                                : 'bg-emerald-50 text-emerald-700'
                            }`}
                          >
                            {isRiskIncrease ? 'Increases Default Risk' : 'Reduces Default Risk'}
                          </span>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>

      {/* Section 3: ZIP Code Ablation Study (Fairness & Proxy Analysis) */}
      <div className="rounded-xl border border-slate-200 bg-white p-6 shadow-xs space-y-4">
        <div className="border-b border-slate-100 pb-3">
          <div className="flex items-center gap-2">
            <Scale className="h-4 w-4 text-blue-600" />
            <h3 className="text-base font-bold text-slate-900">
              ZIP-Code Ablation & Spatial Proxy Investigation
            </h3>
          </div>
          <p className="text-xs text-slate-500 mt-0.5">
            Ablation analysis comparing model performance with vs without 3-digit ZIP code features
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="rounded-lg border border-slate-200 bg-slate-50 p-4 space-y-1">
            <span className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider">
              With 3-Digit ZIP Codes
            </span>
            <div className="text-lg font-bold font-mono text-slate-900">ROC-AUC: 0.6710</div>
            <p className="text-xs text-slate-600">PR-AUC: 0.3340 &bull; High-dimensional spatial encoding</p>
          </div>

          <div className="rounded-lg border border-blue-200 bg-blue-50/60 p-4 space-y-1">
            <span className="text-[11px] font-semibold text-blue-700 uppercase tracking-wider">
              Without ZIP Codes (Selected)
            </span>
            <div className="text-lg font-bold font-mono text-blue-950">ROC-AUC: 0.6679</div>
            <p className="text-xs text-blue-800">PR-AUC: 0.3318 &bull; Marginal 0.0031 AUC reduction</p>
          </div>

          <div className="rounded-lg border border-slate-200 bg-slate-50 p-4 space-y-1">
            <span className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider">
              Statistical Difference
            </span>
            <div className="text-lg font-bold font-mono text-slate-900">&Delta; ROC-AUC: -0.0031</div>
            <p className="text-xs text-slate-600">Preserves predictive utility while removing granular spatial identifiers</p>
          </div>
        </div>

        <div className="rounded-lg border border-amber-200 bg-amber-50/70 p-4 text-xs text-amber-900 flex items-start gap-3">
          <AlertTriangle className="h-5 w-5 text-amber-600 shrink-0 mt-0.5" />
          <div className="space-y-1">
            <strong className="text-amber-950">
              Fairness Research Conclusion — Blind Exclusion Does Not Remove Proxies:
            </strong>
            <p>
              Excluding granular ZIP codes is a common anti-redlining engineering step, but our global TreeSHAP audit reveals that higher-level state categorical features (<code className="font-mono text-amber-950">addr_state</code>) continue to carry significant attribution. Simply dropping a sensitive feature does not guarantee fairness when correlated demographic proxies remain in the data pipeline.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};
