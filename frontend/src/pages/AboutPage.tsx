import React from 'react';
import {
  ShieldAlert,
  Database,
  Layers,
  Sparkles,
  GitBranch,
  MessageSquare,
  AlertTriangle,
  Scale,
  Globe2,
  FileCheck,
} from 'lucide-react';

export const AboutPage: React.FC = () => {
  return (
    <div className="space-y-8 max-w-5xl">
      {/* Title & Acronym Banner */}
      <div className="rounded-xl border border-blue-100 bg-gradient-to-br from-slate-900 via-slate-800 to-blue-950 p-8 text-white shadow-md">
        <div className="flex items-center gap-3 mb-2">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-blue-600 text-white shadow-md">
            <ShieldAlert className="h-6 w-6" />
          </div>
          <div>
            <h1 className="text-2xl font-extrabold tracking-wider text-white">
              ARBITER
            </h1>
            <p className="text-xs font-mono text-blue-300">
              Adversarially Robust and Bias-Informed Transaction Evaluation and Risk Scoring System
            </p>
          </div>
        </div>

        <p className="mt-4 text-xs text-slate-300 max-w-3xl leading-relaxed">
          An academic and research-driven risk intelligence architecture for consumer loan underwriting. ARBITER bridges modern gradient boosting with empirical perturbation stress testing, explainable AI (XAI), and operational human-in-the-loop review boundaries.
        </p>

        <div className="mt-6 flex flex-wrap gap-2 text-[11px]">
          <span className="rounded-full bg-blue-900/60 border border-blue-700/60 px-3 py-1 font-mono text-blue-200">
            Stage 2.5: Full-Stack Risk Platform
          </span>
          <span className="rounded-full bg-slate-800 border border-slate-700 px-3 py-1 font-mono text-slate-300">
            XGBoost ML Pipeline
          </span>
          <span className="rounded-full bg-slate-800 border border-slate-700 px-3 py-1 font-mono text-slate-300">
            TreeSHAP & LIME Interpretability
          </span>
          <span className="rounded-full bg-slate-800 border border-slate-700 px-3 py-1 font-mono text-slate-300">
            Operational Review Policy [0.45, 0.55]
          </span>
        </div>
      </div>

      {/* End-to-End System Architecture SVG Diagram */}
      <div className="rounded-xl border border-slate-200 bg-white p-6 shadow-xs space-y-4">
        <div className="flex items-center justify-between border-b border-slate-100 pb-3">
          <h3 className="text-sm font-bold text-slate-900 flex items-center gap-2">
            <Layers className="h-4 w-4 text-blue-600" />
            System Architecture & Dataflow Diagram
          </h3>
          <span className="text-[11px] font-mono text-slate-400">
            End-to-End Operational Pipeline
          </span>
        </div>

        {/* High-Contrast Technical SVG Architecture Flow */}
        <div className="w-full overflow-x-auto py-2">
          <svg
            viewBox="0 0 920 180"
            className="w-full min-w-[780px] h-auto select-none"
            xmlns="http://www.w3.org/2000/svg"
          >
            <defs>
              <marker
                id="arrowhead"
                markerWidth="8"
                markerHeight="6"
                refX="7"
                refY="3"
                orient="auto"
              >
                <polygon points="0 0, 8 3, 0 6" fill="#3b82f6" />
              </marker>
              <linearGradient id="blueGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stopColor="#1e40af" />
                <stop offset="100%" stopColor="#2563eb" />
              </linearGradient>
              <linearGradient id="slateGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stopColor="#0f172a" />
                <stop offset="100%" stopColor="#1e293b" />
              </linearGradient>
            </defs>

            {/* Step 1: Loan Application */}
            <g transform="translate(10, 30)">
              <rect width="110" height="90" rx="8" fill="#f8fafc" stroke="#cbd5e1" strokeWidth="1.5" />
              <text x="55" y="32" textAnchor="middle" fill="#0f172a" fontSize="11" fontWeight="bold">Loan Applicant</text>
              <text x="55" y="50" textAnchor="middle" fill="#64748b" fontSize="9">Borrower Profile</text>
              <text x="55" y="65" textAnchor="middle" fill="#64748b" fontSize="9">Financial Inputs</text>
              <rect x="25" y="75" width="60" height="15" rx="3" fill="#e2e8f0" />
              <text x="55" y="86" textAnchor="middle" fill="#475569" fontSize="8" fontWeight="bold">JSON Payload</text>
            </g>

            {/* Arrow 1 -> 2 */}
            <path d="M 120 75 L 145 75" stroke="#3b82f6" strokeWidth="2" markerEnd="url(#arrowhead)" />

            {/* Step 2: React UI Frontend */}
            <g transform="translate(150, 25)">
              <rect width="120" height="100" rx="8" fill="url(#blueGrad)" stroke="#1d4ed8" strokeWidth="1.5" />
              <text x="60" y="30" textAnchor="middle" fill="#ffffff" fontSize="12" fontWeight="bold">React 19 UI</text>
              <text x="60" y="46" textAnchor="middle" fill="#bfdbfe" fontSize="9">Vite & Tailwind CSS</text>
              <text x="60" y="60" textAnchor="middle" fill="#bfdbfe" fontSize="9">TanStack Query</text>
              <text x="60" y="74" textAnchor="middle" fill="#bfdbfe" fontSize="9">Zod Validation</text>
              <text x="60" y="90" textAnchor="middle" fill="#ffffff" fontSize="8" fontWeight="bold">Active Interface</text>
            </g>

            {/* Arrow 2 -> 3 */}
            <path d="M 270 75 L 295 75" stroke="#3b82f6" strokeWidth="2" markerEnd="url(#arrowhead)" />

            {/* Step 3: FastAPI Backend */}
            <g transform="translate(300, 25)">
              <rect width="120" height="100" rx="8" fill="url(#slateGrad)" stroke="#334155" strokeWidth="1.5" />
              <text x="60" y="30" textAnchor="middle" fill="#ffffff" fontSize="12" fontWeight="bold">FastAPI Service</text>
              <text x="60" y="46" textAnchor="middle" fill="#94a3b8" fontSize="9">Uvicorn Async API</text>
              <text x="60" y="60" textAnchor="middle" fill="#94a3b8" fontSize="9">Pydantic Schemas</text>
              <text x="60" y="74" textAnchor="middle" fill="#94a3b8" fontSize="9">CORS & Artifacts</text>
              <text x="60" y="90" textAnchor="middle" fill="#38bdf8" fontSize="8" fontWeight="bold">Port 8000</text>
            </g>

            {/* Arrow 3 -> 4 */}
            <path d="M 420 75 L 445 75" stroke="#3b82f6" strokeWidth="2" markerEnd="url(#arrowhead)" />

            {/* Step 4: Preprocessing & Ablation */}
            <g transform="translate(450, 30)">
              <rect width="120" height="90" rx="8" fill="#f8fafc" stroke="#cbd5e1" strokeWidth="1.5" />
              <text x="60" y="30" textAnchor="middle" fill="#0f172a" fontSize="11" fontWeight="bold">Preprocessing</text>
              <text x="60" y="46" textAnchor="middle" fill="#64748b" fontSize="9">ColumnTransformer</text>
              <text x="60" y="60" textAnchor="middle" fill="#64748b" fontSize="9">Median / OneHot</text>
              <text x="60" y="76" textAnchor="middle" fill="#b45309" fontSize="8" fontWeight="bold">ZIP Code Stripped</text>
            </g>

            {/* Arrow 4 -> 5 */}
            <path d="M 570 75 L 595 75" stroke="#3b82f6" strokeWidth="2" markerEnd="url(#arrowhead)" />

            {/* Step 5: XGBoost Classifier */}
            <g transform="translate(600, 25)">
              <rect width="130" height="100" rx="8" fill="#0f766e" stroke="#115e59" strokeWidth="1.5" />
              <text x="65" y="30" textAnchor="middle" fill="#ffffff" fontSize="12" fontWeight="bold">XGBoost Model</text>
              <text x="65" y="46" textAnchor="middle" fill="#ccfbf1" fontSize="9">200 Estimators</text>
              <text x="65" y="60" textAnchor="middle" fill="#ccfbf1" fontSize="9">Binary Logistic Loss</text>
              <text x="65" y="74" textAnchor="middle" fill="#ccfbf1" fontSize="9">Default Probability</text>
              <text x="65" y="90" textAnchor="middle" fill="#fef08a" fontSize="8" fontWeight="bold">Probability Score</text>
            </g>

            {/* Arrow 5 -> 6 */}
            <path d="M 730 75 L 755 75" stroke="#3b82f6" strokeWidth="2" markerEnd="url(#arrowhead)" />

            {/* Step 6: Decision Routing & Review Policy */}
            <g transform="translate(760, 20)">
              <rect width="145" height="110" rx="8" fill="#ffffff" stroke="#94a3b8" strokeWidth="1.5" />
              <text x="72" y="26" textAnchor="middle" fill="#0f172a" fontSize="11" fontWeight="bold">Decision Routing</text>
              <rect x="15" y="36" width="115" height="18" rx="4" fill="#ecfdf5" stroke="#a7f3d0" />
              <text x="72" y="48" textAnchor="middle" fill="#047857" fontSize="8" fontWeight="bold">Low Risk (&lt; 0.45)</text>
              <rect x="15" y="58" width="115" height="18" rx="4" fill="#fffbeb" stroke="#fde68a" />
              <text x="72" y="70" textAnchor="middle" fill="#b45309" fontSize="8" fontWeight="bold">Review Band [0.45–0.55]</text>
              <rect x="15" y="80" width="115" height="18" rx="4" fill="#fff1f2" stroke="#fecdd3" />
              <text x="72" y="92" textAnchor="middle" fill="#be123c" fontSize="8" fontWeight="bold">Elevated Risk (&gt; 0.55)</text>
              <text x="72" y="112" textAnchor="middle" fill="#64748b" fontSize="8">TreeSHAP Attribution</text>
            </g>
          </svg>
        </div>

        <div className="text-xs text-slate-500 text-center">
          Component architecture currently operational in the repository.
        </div>
      </div>

      {/* Cross-Jurisdictional Regulatory Framework Comparison */}
      <div className="rounded-xl border border-slate-200 bg-white p-6 shadow-xs space-y-4">
        <div className="flex items-center gap-2 border-b border-slate-100 pb-3">
          <Scale className="h-4 w-4 text-blue-600" />
          <h3 className="text-base font-bold text-slate-900">
            Credit Regulatory Governance: US vs Indian Jurisdictions
          </h3>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 text-xs">
          {/* US Framework */}
          <div className="rounded-lg border border-slate-200 bg-slate-50/70 p-4 space-y-2.5">
            <div className="flex items-center gap-2 font-bold text-slate-900">
              <Globe2 className="h-4 w-4 text-blue-600" />
              <span>United States Credit & AI Regulatory Framework</span>
            </div>
            <ul className="space-y-1.5 text-slate-600 list-disc pl-4 leading-relaxed">
              <li>
                <strong>Equal Credit Opportunity Act (ECOA / Reg B):</strong> Prohibits discrimination based on race, sex, age, marital status, or national origin. Requires adverse action explanations (model explainability via SHAP/reason codes).
              </li>
              <li>
                <strong>Fair Housing Act (FHA):</strong> Outlaws redlining and geographic bias in residential lending (addressed by our ZIP code ablation study).
              </li>
              <li>
                <strong>Federal Reserve SR 11-7:</strong> Supervisory Guidance on Model Risk Management, mandating conceptual soundness, stress testing, and ongoing sensitivity auditing.
              </li>
            </ul>
          </div>

          {/* Indian Framework */}
          <div className="rounded-lg border border-slate-200 bg-slate-50/70 p-4 space-y-2.5">
            <div className="flex items-center gap-2 font-bold text-slate-900">
              <FileCheck className="h-4 w-4 text-emerald-700" />
              <span>Indian Credit & Digital Lending Framework (RBI)</span>
            </div>
            <ul className="space-y-1.5 text-slate-600 list-disc pl-4 leading-relaxed">
              <li>
                <strong>RBI Guidelines on Digital Lending (2022):</strong> Regulated Entities (REs) must maintain algorithmic accountability, prevent algorithmic bias, and maintain underwriting control over automated models.
              </li>
              <li>
                <strong>RBI Fair Practices Code:</strong> Mandates transparent assessment of borrower repayment capacity without predatory FOIR (Fixed Obligation to Income Ratio) exploitation.
              </li>
              <li>
                <strong>Digital Personal Data Protection (DPDP) Act 2023:</strong> Restricts unauthorized cross-border demographic data transfers and mandates purpose limitation for credit evaluations.
              </li>
            </ul>
          </div>
        </div>
      </div>

      {/* Problem Statement & Objectives */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="rounded-xl border border-slate-200 bg-white p-6 shadow-xs space-y-2">
          <h3 className="text-sm font-bold text-slate-900 flex items-center gap-2">
            <AlertTriangle className="h-4 w-4 text-amber-600" />
            The Underwriting Problem
          </h3>
          <p className="text-xs text-slate-600 leading-relaxed">
            Machine learning models in credit underwriting face severe challenges: opaque black-box predictions, vulnerability to proxy variables (such as postal ZIP codes recreating historic redlining), and fragility under small data perturbations.
          </p>
          <p className="text-xs text-slate-600 leading-relaxed">
            Standard binary thresholding (e.g., hard rejection at 0.50) forces high numbers of automatic false negatives and discards borderline applicants who warrant nuanced manual review.
          </p>
        </div>

        <div className="rounded-xl border border-slate-200 bg-white p-6 shadow-xs space-y-2">
          <h3 className="text-sm font-bold text-slate-900 flex items-center gap-2">
            <ShieldAlert className="h-4 w-4 text-blue-600" />
            ARBITER Research Objectives
          </h3>
          <ul className="text-xs text-slate-600 space-y-1.5 list-disc pl-4 leading-relaxed">
            <li>
              <strong>Ablate Discriminatory Proxies:</strong> Empirically evaluate whether removing 3-digit ZIP codes damages default discrimination.
            </li>
            <li>
              <strong>Operationalize Review Band:</strong> Route predictions between 0.45 and 0.55 to senior credit underwriters to prevent unreviewed false negatives.
            </li>
            <li>
              <strong>Audit Local Sensitivity:</strong> Quantify model stability under directional perturbations of FICO, revenue, debt-to-income, and loan size.
            </li>
            <li>
              <strong>Surface XAI Transparency:</strong> Provide immediate TreeSHAP attributions and LIME local surrogate rules.
            </li>
          </ul>
        </div>
      </div>

      {/* Dataset & Model Specification */}
      <div className="rounded-xl border border-slate-200 bg-white p-6 shadow-xs space-y-4">
        <h3 className="text-sm font-bold text-slate-900 flex items-center gap-2">
          <Database className="h-4 w-4 text-blue-600" />
          Dataset & Model Specifications
        </h3>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 text-xs">
          <div className="rounded-lg bg-slate-50 p-3 border border-slate-100">
            <span className="text-slate-400 block mb-0.5">Source Dataset</span>
            <span className="font-bold text-slate-900">LendingClub Consumer Loans</span>
            <span className="block text-[11px] text-slate-500 mt-1">
              50,000 stratified training records with confirmed repayment outcomes.
            </span>
          </div>

          <div className="rounded-lg bg-slate-50 p-3 border border-slate-100">
            <span className="text-slate-400 block mb-0.5">Model Artifact</span>
            <span className="font-mono font-bold text-slate-900 text-[11px]">
              loan_xgboost_without_zip.joblib
            </span>
            <span className="block text-[11px] text-slate-500 mt-1">
              Exported scikit-learn pipeline with serialized XGBoost 3.4.1 classifier.
            </span>
          </div>

          <div className="rounded-lg bg-slate-50 p-3 border border-slate-100">
            <span className="text-slate-400 block mb-0.5">Explainability Suite</span>
            <span className="font-bold text-slate-900">TreeSHAP 0.53 + LIME 0.2</span>
            <span className="block text-[11px] text-slate-500 mt-1">
              Calculates exact tree path feature attributions and local linear approximations.
            </span>
          </div>
        </div>
      </div>

      {/* Future Roadmap & Stage 3 LLM Integration Points */}
      <div className="rounded-xl border border-slate-200 bg-white p-6 shadow-xs space-y-4">
        <div className="flex items-center justify-between border-b border-slate-100 pb-3">
          <div className="flex items-center gap-2">
            <GitBranch className="h-4 w-4 text-indigo-600" />
            <h3 className="text-sm font-bold text-slate-900">
              Phased Roadmap & Stage 3 LLM Integration Architecture
            </h3>
          </div>
          <span className="rounded-md bg-indigo-50 text-indigo-700 px-2 py-0.5 text-xs font-mono font-semibold">
            Extensibility Ready
          </span>
        </div>

        <p className="text-xs text-slate-600 leading-relaxed">
          The application follows a modular, decoupled architecture where future stages can be integrated without modifying the core prediction engine:
        </p>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 text-xs">
          <div className="rounded-lg border border-slate-200 p-3 bg-slate-50/50">
            <div className="flex items-center gap-1.5 text-slate-900 font-bold mb-1">
              <span className="h-2 w-2 rounded-full bg-emerald-500" />
              Stage 2.5 (Current)
            </div>
            <p className="text-slate-600 text-[11px] leading-relaxed">
              Full-stack risk intelligence dashboard, live FastAPI inference, verified robustness visualizations, Indian localization audit, and dynamic TreeSHAP attribution.
            </p>
          </div>

          <div className="rounded-lg border border-indigo-200 p-3 bg-indigo-50/30">
            <div className="flex items-center gap-1.5 text-indigo-900 font-bold mb-1">
              <MessageSquare className="h-3.5 w-3.5 text-indigo-600" />
              Stage 3: LLM Copilot (Planned)
            </div>
            <p className="text-slate-600 text-[11px] leading-relaxed">
              Synthesizes applicant attributes, model probability, and top SHAP features into a natural-language underwriting memorandum for credit committee review.
            </p>
          </div>

          <div className="rounded-lg border border-slate-200 p-3 bg-slate-50/50">
            <div className="flex items-center gap-1.5 text-slate-900 font-bold mb-1">
              <Sparkles className="h-3.5 w-3.5 text-slate-500" />
              Stage 4: Fraud Engine (Planned)
            </div>
            <p className="text-slate-600 text-[11px] leading-relaxed">
              Transaction anomaly detection and synthetic identity screening operating concurrently with default scoring.
            </p>
          </div>
        </div>

        {/* Technical Documentation of Stage 3 Extension Interface */}
        <div className="rounded-lg bg-slate-900 p-4 text-xs font-mono text-slate-300 space-y-2">
          <div className="flex items-center justify-between text-slate-400 text-[11px] border-b border-slate-800 pb-2">
            <span>Stage 3 LLM Service Interface Contract (Planned Integration Point)</span>
            <span className="text-blue-400 font-sans">POST /assistant/underwrite</span>
          </div>
          <pre className="text-[11px] text-blue-300 overflow-x-auto leading-relaxed">
{`// Input to future Stage 3 LLM service:
{
  "application": LoanApplicationInput,
  "prediction": LoanPredictionResponse,
  "top_shap_features": ShapLocalFeature[],
  "requested_tone": "institutional_underwriting_memo"
}
// Outputs: Structured underwriter memorandum, risk caveats, policy flags.`}
          </pre>
        </div>
      </div>
    </div>
  );
};
