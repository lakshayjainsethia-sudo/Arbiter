export interface LoanApplicationInput {
  revenue: number;
  dti_n: number;
  loan_amnt: number;
  fico_n: number;
  experience_c: number;
  emp_length: string;
  purpose: string;
  home_ownership_n: string;
  addr_state: string;
}

export type DecisionCategory =
  | 'lower_predicted_default_risk'
  | 'manual_review'
  | 'higher_predicted_default_risk'
  | string;

export interface LoanPredictionResponse {
  project: string;
  default_probability: number;
  predicted_default: number;
  decision: DecisionCategory;
  review_band: [number, number];
  note: string;
}

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  explainer_loaded: boolean;
  artifacts_status: {
    robustness_report: boolean;
    shap_global: boolean;
    shap_local: boolean;
    zip_ablation: boolean;
  };
}

export interface PresetProfile {
  id: string;
  label: string;
  description: string;
  data: LoanApplicationInput;
}

export interface ModelMetadata {
  project: string;
  model: {
    type: string;
    variant: string;
    pipeline: string;
    base_threshold: number;
    review_band: [number, number];
    target: string;
  };
  supported_categories: {
    emp_length: string[];
    purpose: string[];
    home_ownership_n: string[];
    addr_state: string[];
  };
  numerical_fields: Record<
    string,
    {
      label: string;
      min: number;
      max?: number;
      default: number;
      step: number;
    }
  >;
  preset_profiles: PresetProfile[];
}

export interface VerifiedSummary {
  model_type: string;
  dataset: string;
  model_variant: string;
  clean_baseline_roc_auc: number;
  clean_baseline_pr_auc: number;
  evaluation_sample_size: number;
  default_cases_in_sample: number;
  clean_review_rate: number;
  clean_reviewed_defaults: number;
  clean_automatic_false_negatives: number;
  review_band: [number, number];
}

export interface RobustnessScenarioRow {
  scenario: string;
  roc_auc: number;
  pr_auc: number;
  precision: number;
  recall: number;
  f1: number;
  original_false_negatives: number;
  review_count: number;
  review_rate: number;
  reviewed_defaults: number;
  reviewed_nondefaults: number;
  automatic_false_negatives: number;
  automatic_decision_rate: number;
  class_flip_rate_vs_clean: number;
}

export interface StabilitySummaryRow {
  feature: string;
  direction: string;
  requested_change: number;
  prediction_change_rate: number;
  mean_absolute_probability_change: number;
  changed_to_default: number;
  changed_to_non_default: number;
}

export interface ZipAblationRow {
  model: string;
  roc_auc: number;
  pr_auc: number;
  default_precision: number;
  default_recall: number;
  default_f1: number;
}

export interface MetricsResponse {
  verified_summary: VerifiedSummary;
  robustness_sample_clean: RobustnessScenarioRow | null;
  full_test_baseline: {
    scenario: string;
    roc_auc: number;
    pr_auc: number;
    default_precision: number;
    default_recall: number;
    default_f1: number;
    positive_predictions: number;
  } | null;
  zip_ablation_comparison: ZipAblationRow[];
  training_metrics: {
    model: string;
    rows: number;
    roc_auc: number;
    pr_auc: number;
    training_time_seconds: number;
  } | null;
  limitations: string[];
}

export interface RobustnessResponse {
  comparison: RobustnessScenarioRow[];
  stability_summary: StabilitySummaryRow[];
  review_metrics: any[];
  perturbation_specifications: Record<
    string,
    {
      type: string;
      delta?: number;
      delta_pct?: number;
      description: string;
    }
  >;
  policy: {
    review_low: number;
    review_high: number;
    clean_review_rate: number;
    note: string;
  };
}

export interface ShapGlobalFeature {
  feature: string;
  mean_absolute_shap: number;
}

export interface GlobalExplainabilityResponse {
  features: ShapGlobalFeature[];
  total_features: number;
  method: string;
  interpretation: string;
}

export interface ShapLocalFeature {
  feature: string;
  feature_value?: number;
  shap_value: number;
  absolute_shap: number;
}

export interface LimeFeatureCondition {
  feature_condition: string;
  lime_weight: number;
}

export interface LocalExampleResponse {
  benchmark_profile: LoanApplicationInput;
  shap_contributions: ShapLocalFeature[];
  lime_weights: LimeFeatureCondition[];
  note: string;
}

export interface DynamicLocalExplainResponse {
  status: string;
  total_features: number;
  top_contributions: ShapLocalFeature[];
  all_contributions: ShapLocalFeature[];
  note: string;
}

// ── Indian Loan Research types [EXPERIMENTAL] ─────────────────
export interface IndianLoanApplicationInput {
  age: number;
  gender: string;
  marital_status: string;
  city_tier: string;
  state: string;
  employment_type: string;
  bank_account_vintage_years: number;
  monthly_income_inr: number;
  existing_loans_count: number;
  existing_emi_monthly_inr: number;
  credit_utilization_ratio_pct: number;
  num_credit_inquiries_last_6m: number;
  late_payments_last_12m: number;
  loan_amount_requested_inr: number;
  loan_tenure_months: number;
  loan_purpose: string;
  collateral_provided: string;
  cibil_score: number;
  interest_rate_offered_pct: number;
}

export type IndianClassification =
  | 'lower_predicted_default_risk'
  | 'higher_predicted_default_risk'
  | string;

export interface IndianLoanPredictionResponse {
  project: string;
  model: {
    name: string;
    version: string;
    type: string;
    experiment_label: string;
    dataset_caveat: string;
  };
  prediction: {
    default_probability_raw: number;
    predicted_class: number;
    cv_threshold: number;
    classification: IndianClassification;
    probability_label: string;
  };
  features_used: number;
  excluded_features: string[];
  governance: {
    gender_in_model: boolean;
    fairness_review_required: boolean;
    real_world_validation: boolean;
    warnings: string[];
  };
  note: string;
}

export interface IndianShapFeature {
  rank: number;
  feature: string;
  mean_abs_shap: number;
  direction: string;
}

export interface IndianModelMetadata {
  model: {
    name: string;
    version: string;
    experiment_label: string;
    dataset_caveat: string;
    cv_threshold: number;
    features_used: number;
    excluded: string[];
    pipeline: string;
    model_loaded: boolean;
  };
  valid_categories: {
    gender: string[];
    marital_status: string[];
    city_tier: string[];
    state: string[];
    employment_type: string[];
    loan_purpose: string[];
    collateral_provided: string[];
  };
  numeric_ranges: Record<
    string,
    { min: number; max: number; default: number; step: number; label: string }
  >;
  shap_top_features: IndianShapFeature[];
  governance: {
    gender_in_model: boolean;
    fairness_review_required: boolean;
    real_world_validation: boolean;
    calibration_status: string;
    warnings: string[];
  };
}

