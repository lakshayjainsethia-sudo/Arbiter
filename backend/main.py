from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, model_validator

# Project root: the parent directory of backend/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
ABLATION_DIR = ARTIFACTS_DIR / "ablation"
EXPLAINABILITY_DIR = ARTIFACTS_DIR / "explainability"
ROBUSTNESS_DIR = ARTIFACTS_DIR / "robustness"
FINAL_EVAL_DIR = ROBUSTNESS_DIR / "final_evaluation"

MODEL_PATH = ABLATION_DIR / "loan_xgboost_without_zip.joblib"

app = FastAPI(
    title="ARBITER API",
    description="Adversarially Robust and Bias-Informed Transaction Evaluation and Risk Scoring System",
    version="0.2.0",
)

# Enable CORS for local frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained model once when the API starts.
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

model = joblib.load(MODEL_PATH)
preprocessor = model.named_steps["preprocessor"]
classifier = model.named_steps["classifier"]

# Initialize TreeExplainer for dynamic local SHAP explanations
tree_explainer: Optional[shap.TreeExplainer] = None
try:
    tree_explainer = shap.TreeExplainer(classifier)
except Exception as e:
    print(f"Warning: Could not initialize TreeExplainer: {e}")

# ─────────────────────────────────────────────────────────────
# Indian Loan Research Model  [EXPERIMENTAL]
# Completely separate from the US LendingClub model.
# The two models must never cross-contaminate inputs or outputs.
# ─────────────────────────────────────────────────────────────
INDIAN_MODEL_DIR  = ARTIFACTS_DIR / "indian_loan"
INDIAN_MODEL_PATH = INDIAN_MODEL_DIR / "logistic_regression_pipeline.joblib"
INDIAN_EXPLAIN_DIR = INDIAN_MODEL_DIR / "explainability"

INDIAN_CV_THRESHOLD   = 0.7623
INDIAN_MODEL_VERSION  = "3B-experimental"
INDIAN_DATASET_CAVEAT = (
    "Dataset provenance is unverified — distribution is consistent with "
    "synthetic or stratified generation. Not validated for real lending decisions."
)
INDIAN_EXPERIMENT_LABEL = "EXPERIMENTAL"

# Features the Indian model accepts and those it explicitly excludes
INDIAN_BASELINE_FEATURES = [
    "age", "gender", "marital_status", "city_tier", "state",
    "employment_type", "bank_account_vintage_years", "monthly_income_inr",
    "existing_loans_count", "existing_emi_monthly_inr",
    "credit_utilization_ratio_pct", "num_credit_inquiries_last_6m",
    "late_payments_last_12m", "loan_amount_requested_inr",
    "loan_tenure_months", "loan_purpose", "collateral_provided",
    "cibil_score", "interest_rate_offered_pct",
]
INDIAN_NUMERIC_FEATURES = [
    "age", "bank_account_vintage_years", "monthly_income_inr",
    "existing_loans_count", "existing_emi_monthly_inr",
    "credit_utilization_ratio_pct", "num_credit_inquiries_last_6m",
    "late_payments_last_12m", "loan_amount_requested_inr",
    "loan_tenure_months", "cibil_score", "interest_rate_offered_pct",
]
INDIAN_CATEGORICAL_FEATURES = [
    "gender", "marital_status", "city_tier", "state",
    "employment_type", "loan_purpose", "collateral_provided",
]
INDIAN_EXCLUDED_FEATURES = ["applicant_id", "default_risk_score"]

INDIAN_VALID_CATEGORIES: Dict[str, List[str]] = {
    "gender":       ["Male", "Female", "Other"],
    "marital_status": ["Single", "Married", "Divorced", "Widowed"],
    "city_tier":    ["Tier 1", "Tier 2", "Tier 3"],
    "state": [
        "Andhra Pradesh", "Bihar", "Delhi", "Gujarat", "Karnataka",
        "Kerala", "Madhya Pradesh", "Maharashtra", "Odisha", "Punjab",
        "Rajasthan", "Tamil Nadu", "Telangana", "Uttar Pradesh", "West Bengal",
    ],
    "employment_type": ["Salaried", "Self-Employed", "Business Owner", "Unemployed"],
    "loan_purpose": [
        "Home Improvement", "Education", "Medical", "Debt Consolidation",
        "Business", "Vehicle", "Personal", "Travel", "Other",
    ],
    "collateral_provided": ["Yes", "No"],
}

INDIAN_NUMERIC_RANGES: Dict[str, Dict[str, Any]] = {
    "age":                          {"min": 18,  "max": 70,        "default": 35,     "step": 1,      "label": "Age (years)"},
    "bank_account_vintage_years":   {"min": 0,   "max": 40,        "default": 5,      "step": 1,      "label": "Bank Account Vintage (years)"},
    "monthly_income_inr":           {"min": 5000,"max": 1000000,   "default": 75000,  "step": 1000,   "label": "Monthly Income (INR)"},
    "existing_loans_count":         {"min": 0,   "max": 10,        "default": 1,      "step": 1,      "label": "Existing Loans Count"},
    "existing_emi_monthly_inr":     {"min": 0,   "max": 200000,    "default": 8000,   "step": 500,    "label": "Existing EMI / Month (INR)"},
    "credit_utilization_ratio_pct": {"min": 0,   "max": 100,       "default": 35.0,   "step": 0.5,    "label": "Credit Utilization (%)"},
    "num_credit_inquiries_last_6m": {"min": 0,   "max": 20,        "default": 2,      "step": 1,      "label": "Credit Inquiries (last 6 months)"},
    "late_payments_last_12m":       {"min": 0,   "max": 12,        "default": 0,      "step": 1,      "label": "Late Payments (last 12 months)"},
    "loan_amount_requested_inr":    {"min": 10000,"max": 10000000, "default": 500000, "step": 5000,   "label": "Loan Amount Requested (INR)"},
    "loan_tenure_months":           {"min": 6,   "max": 360,       "default": 36,     "step": 6,      "label": "Loan Tenure (months)"},
    "cibil_score":                  {"min": 300, "max": 900,       "default": 740,    "step": 1,      "label": "CIBIL Score (300–900)"},
    "interest_rate_offered_pct":    {"min": 6.0, "max": 36.0,      "default": 11.5,   "step": 0.25,   "label": "Interest Rate Offered (%)"},
}

# Top SHAP features (pre-computed in Stage 3C — association only, not causal)
INDIAN_SHAP_TOP_FEATURES = [
    {"rank": 1,  "feature": "cibil_score",                  "mean_abs_shap": 1.0417, "direction": "higher CIBIL → lower risk"},
    {"rank": 2,  "feature": "monthly_income_inr",           "mean_abs_shap": 0.2805, "direction": "higher income → lower risk"},
    {"rank": 3,  "feature": "existing_emi_monthly_inr",     "mean_abs_shap": 0.2531, "direction": "higher EMI → higher risk"},
    {"rank": 4,  "feature": "late_payments_last_12m",       "mean_abs_shap": 0.2476, "direction": "more late payments → higher risk"},
    {"rank": 5,  "feature": "loan_amount_requested_inr",    "mean_abs_shap": 0.1861, "direction": "higher loan → slightly higher risk"},
    {"rank": 6,  "feature": "collateral_provided",          "mean_abs_shap": 0.1296, "direction": "collateral=Yes → lower risk"},
    {"rank": 7,  "feature": "num_credit_inquiries_last_6m", "mean_abs_shap": 0.1199, "direction": "more inquiries → higher risk"},
    {"rank": 8,  "feature": "credit_utilization_ratio_pct", "mean_abs_shap": 0.0816, "direction": "higher utilization → small effect"},
    {"rank": 9,  "feature": "gender",                       "mean_abs_shap": 0.0688, "direction": "FAIRNESS REVIEW REQUIRED — see governance"},
    {"rank": 10, "feature": "interest_rate_offered_pct",    "mean_abs_shap": 0.0561, "direction": "higher rate → slightly lower risk (leakage risk — monitor)"},
]

# Load Indian model at startup
indian_lr_pipeline: Optional[Any] = None
indian_preprocessor: Optional[Any] = None
indian_classifier:   Optional[Any] = None
indian_shap_explainer: Optional[shap.LinearExplainer] = None

if INDIAN_MODEL_PATH.exists():
    try:
        indian_lr_pipeline  = joblib.load(INDIAN_MODEL_PATH)
        indian_preprocessor = indian_lr_pipeline.named_steps["preprocessor"]
        indian_classifier   = indian_lr_pipeline.named_steps["classifier"]
        print(f"[ARBITER] Indian LR pipeline loaded from {INDIAN_MODEL_PATH}")
    except Exception as _e:
        print(f"[ARBITER] Warning: Could not load Indian model: {_e}")
else:
    print(f"[ARBITER] Warning: Indian model not found at {INDIAN_MODEL_PATH}")


# Extract supported categories dynamically from the fitted OneHotEncoder
def get_supported_categories() -> Dict[str, List[str]]:
    try:
        categorical_transformer = preprocessor.transformers_[1]
        encoder = categorical_transformer[1].named_steps["encoder"]
        categorical_cols = categorical_transformer[2]
        return {
            col: [str(val) for val in cats]
            for col, cats in zip(categorical_cols, encoder.categories_)
        }
    except Exception as exc:
        print(f"Warning: Failed to extract categories: {exc}")
        return {
            "emp_length": [
                "1 year", "10+ years", "2 years", "3 years", "4 years", "5 years",
                "6 years", "7 years", "8 years", "9 years", "< 1 year", "NI"
            ],
            "purpose": [
                "car", "credit_card", "debt_consolidation", "home_improvement", "house",
                "major_purchase", "medical", "moving", "other", "renewable_energy",
                "small_business", "vacation"
            ],
            "home_ownership_n": ["MORTGAGE", "OTHER", "OWN", "RENT"],
            "addr_state": [
                "AK", "AL", "AR", "AZ", "CA", "CO", "CT", "DC", "DE", "FL", "GA", "HI",
                "IL", "IN", "KS", "KY", "LA", "MA", "MD", "ME", "MI", "MN", "MO", "MS",
                "MT", "NC", "ND", "NE", "NH", "NJ", "NM", "NV", "NY", "OH", "OK", "OR",
                "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VA", "VT", "WA", "WI", "WV", "WY"
            ]
        }

SUPPORTED_CATEGORIES = get_supported_categories()


class LoanApplication(BaseModel):
    revenue: float = Field(ge=0, description="Annual verifiable revenue / income in USD")
    dti_n: float = Field(ge=0, description="Debt-to-income ratio (%)")
    loan_amnt: float = Field(gt=0, description="Requested loan amount in USD")
    fico_n: float = Field(ge=300, le=850, description="FICO credit score (300 - 850)")
    experience_c: float = Field(description="Years of professional credit experience")
    emp_length: str = Field(description="Employment length string")
    purpose: str = Field(description="Loan purpose category")
    home_ownership_n: str = Field(description="Home ownership status")
    addr_state: str = Field(description="US state code (2 letters)")


def safe_read_csv(filepath: Path) -> List[Dict[str, Any]]:
    if not filepath.exists():
        return []
    df = pd.read_csv(filepath)
    df = df.replace({np.nan: None})
    return df.to_dict(orient="records")


@app.get("/")
def home():
    return {
        "project": "ARBITER",
        "name": "Adversarially Robust and Bias-Informed Transaction Evaluation and Risk Scoring System",
        "status": "running",
        "model": "XGBoost loan risk model (ZIP-code excluded)",
        "version": "0.2.0",
    }


@app.get("/health")
def health():
    final_robustness_path = FINAL_EVAL_DIR / "final_robustness_comparison.csv"
    shap_global_path = EXPLAINABILITY_DIR / "shap_global_importance.csv"
    shap_local_path = EXPLAINABILITY_DIR / "shap_local_explanation.csv"
    ablation_path = ABLATION_DIR / "zipcode_ablation_comparison.csv"

    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "explainer_loaded": tree_explainer is not None,
        "artifacts_status": {
            "robustness_report": final_robustness_path.exists(),
            "shap_global": shap_global_path.exists(),
            "shap_local": shap_local_path.exists(),
            "zip_ablation": ablation_path.exists(),
        },
    }


@app.get("/metadata")
def metadata():
    return {
        "project": "ARBITER",
        "model": {
            "type": "XGBoost Classifier",
            "variant": "ZIP-code-excluded model",
            "pipeline": "ColumnTransformer (Imputer + OneHotEncoder) -> XGBClassifier",
            "base_threshold": 0.50,
            "review_band": [0.45, 0.55],
            "target": "Default (1 = Default, 0 = Non-default)",
        },
        "supported_categories": SUPPORTED_CATEGORIES,
        "numerical_fields": {
            "revenue": {"label": "Annual Revenue ($)", "min": 0, "default": 65000, "step": 1000},
            "dti_n": {"label": "Debt-to-Income Ratio (%)", "min": 0, "max": 100, "default": 16.5, "step": 0.1},
            "loan_amnt": {"label": "Loan Amount ($)", "min": 500, "max": 100000, "default": 12000, "step": 500},
            "fico_n": {"label": "FICO Credit Score", "min": 300, "max": 850, "default": 710, "step": 1},
            "experience_c": {"label": "Credit History / Experience (Years)", "min": 0, "max": 60, "default": 8, "step": 1},
        },
        "preset_profiles": [
            {
                "id": "benchmark_local_shap",
                "label": "Benchmark Sample #0 (Saved SHAP/LIME Profile)",
                "description": "The exact borrower record from the research test set used to generate the saved SHAP & LIME local artifacts.",
                "data": {
                    "revenue": 52000.0,
                    "dti_n": 5.7,
                    "loan_amnt": 11000.0,
                    "fico_n": 662.0,
                    "experience_c": 1.0,
                    "emp_length": "5 years",
                    "purpose": "debt_consolidation",
                    "home_ownership_n": "MORTGAGE",
                    "addr_state": "CA",
                },
            },
            {
                "id": "prime_low_risk",
                "label": "Prime Borrower (Low Default Risk)",
                "description": "High revenue, excellent FICO score (790), low DTI, stable 10+ year employment.",
                "data": {
                    "revenue": 140000.0,
                    "dti_n": 8.5,
                    "loan_amnt": 15000.0,
                    "fico_n": 790.0,
                    "experience_c": 14.0,
                    "emp_length": "10+ years",
                    "purpose": "debt_consolidation",
                    "home_ownership_n": "MORTGAGE",
                    "addr_state": "WA",
                },
            },
            {
                "id": "borderline_review",
                "label": "Borderline Applicant (Within Review Band)",
                "description": "Moderate FICO score, medium DTI, typical loan amount falling into the 0.45-0.55 manual review zone.",
                "data": {
                    "revenue": 48000.0,
                    "dti_n": 21.0,
                    "loan_amnt": 16000.0,
                    "fico_n": 655.0,
                    "experience_c": 4.0,
                    "emp_length": "3 years",
                    "purpose": "debt_consolidation",
                    "home_ownership_n": "RENT",
                    "addr_state": "FL",
                },
            },
            {
                "id": "elevated_risk",
                "label": "Elevated Risk Profile (High Default Risk)",
                "description": "High loan request relative to revenue, sub-620 FICO score, elevated debt ratio.",
                "data": {
                    "revenue": 32000.0,
                    "dti_n": 34.0,
                    "loan_amnt": 25000.0,
                    "fico_n": 605.0,
                    "experience_c": 2.0,
                    "emp_length": "< 1 year",
                    "purpose": "small_business",
                    "home_ownership_n": "RENT",
                    "addr_state": "NY",
                },
            },
        ],
    }


@app.get("/metrics")
def get_metrics():
    # 1. Clean final robustness evaluation sample (2,000 rows)
    final_robustness = safe_read_csv(FINAL_EVAL_DIR / "final_robustness_comparison.csv")
    clean_2k = next((row for row in final_robustness if row.get("scenario") == "clean"), None)

    # 2. Clean 10,000-row test partition baseline
    clean_10k_list = safe_read_csv(ROBUSTNESS_DIR / "clean_baseline_metrics.csv")
    clean_10k = clean_10k_list[0] if clean_10k_list else None

    # 3. ZIP-code ablation experiment comparison
    zip_ablation = safe_read_csv(ABLATION_DIR / "zipcode_ablation_comparison.csv")

    # 4. Training metrics
    training_metrics = safe_read_csv(ARTIFACTS_DIR / "loan_xgboost_metrics.csv")

    return {
        "verified_summary": {
            "model_type": "XGBoost",
            "dataset": "LendingClub Loan Granting Dataset",
            "model_variant": "ZIP-code-excluded model",
            "clean_baseline_roc_auc": 0.6679,
            "clean_baseline_pr_auc": 0.3318,
            "evaluation_sample_size": 2000,
            "default_cases_in_sample": 401,
            "clean_review_rate": 0.2430,
            "clean_reviewed_defaults": 98,
            "clean_automatic_false_negatives": 117,
            "review_band": [0.45, 0.55],
        },
        "robustness_sample_clean": clean_2k,
        "full_test_baseline": clean_10k,
        "zip_ablation_comparison": zip_ablation,
        "training_metrics": training_metrics[0] if training_metrics else None,
        "limitations": [
            "Evaluation sample uses a stratified 2,000-row subset of the LendingClub test partition for robustness stress testing.",
            "Historical test set evaluated at 10,000 rows showed ROC-AUC 0.6744 and PR-AUC 0.3403; these are distinct evaluation samples.",
            "Input perturbations represent controlled local sensitivity testing, not universal mathematical robustness guarantees.",
            "Review band (0.45-0.55) routes ambiguous classifications to human underwriters, mitigating automatic false negatives by 40% but requiring manual capacity.",
            "State-level geographic features exhibit significant SHAP weights and require dedicated fairness auditing.",
        ],
    }


@app.get("/robustness")
def get_robustness():
    comparison = safe_read_csv(FINAL_EVAL_DIR / "final_robustness_comparison.csv")
    stability_summary = safe_read_csv(ROBUSTNESS_DIR / "stability_defense_summary.csv")
    review_metrics = safe_read_csv(ROBUSTNESS_DIR / "review_policy_metrics.csv")

    return {
        "comparison": comparison,
        "stability_summary": stability_summary,
        "review_metrics": review_metrics,
        "perturbation_specifications": {
            "fico_n": {"type": "additive", "delta": 5.0, "description": "±5 FICO score points"},
            "dti_n": {"type": "additive", "delta": 0.5, "description": "±0.5% Debt-to-Income ratio"},
            "revenue": {"type": "proportional", "delta_pct": 5.0, "description": "±5% relative median income ($3,260)"},
            "loan_amnt": {"type": "proportional", "delta_pct": 5.0, "description": "±5% relative median loan amount ($600)"},
        },
        "policy": {
            "review_low": 0.45,
            "review_high": 0.55,
            "clean_review_rate": 0.2430,
            "note": "A review band routes borderline cases for human loan officer scrutiny. It does not eliminate false negatives.",
        },
    }


@app.get("/explainability/global")
def get_global_explainability():
    features = safe_read_csv(EXPLAINABILITY_DIR / "shap_global_importance.csv")
    return {
        "features": features,
        "total_features": len(features),
        "method": "TreeSHAP (mean absolute SHAP value across representative test set)",
        "interpretation": "Describes average model feature dependency magnitude across training distribution; does not imply real-world causation.",
    }


@app.get("/explainability/local/example")
def get_local_example_explainability():
    shap_data = safe_read_csv(EXPLAINABILITY_DIR / "shap_local_explanation.csv")
    lime_data = safe_read_csv(EXPLAINABILITY_DIR / "lime_loan_explanation.csv")

    return {
        "benchmark_profile": {
            "revenue": 52000.0,
            "dti_n": 5.7,
            "loan_amnt": 11000.0,
            "fico_n": 662.0,
            "experience_c": 1.0,
            "emp_length": "5 years",
            "purpose": "debt_consolidation",
            "home_ownership_n": "MORTGAGE",
            "addr_state": "CA",
        },
        "shap_contributions": shap_data,
        "lime_weights": lime_data,
        "note": "Pre-computed explainability artifacts for test benchmark sample #0.",
    }


@app.post("/explainability/local")
def explain_local(application: LoanApplication):
    if tree_explainer is None:
        raise HTTPException(
            status_code=503,
            detail="SHAP TreeExplainer is not available on this server.",
        )

    try:
        loan_data = pd.DataFrame(
            [
                {
                    "revenue": application.revenue,
                    "dti_n": application.dti_n,
                    "loan_amnt": application.loan_amnt,
                    "fico_n": application.fico_n,
                    "experience_c": application.experience_c,
                    "emp_length": application.emp_length,
                    "purpose": application.purpose,
                    "home_ownership_n": application.home_ownership_n,
                    "addr_state": application.addr_state,
                }
            ]
        )

        transformed = preprocessor.transform(loan_data)
        feature_names = preprocessor.get_feature_names_out()

        if hasattr(transformed, "toarray"):
            transformed_dense = transformed.toarray()
        else:
            transformed_dense = np.asarray(transformed)

        shap_res = tree_explainer(transformed_dense, check_additivity=False)
        values = np.asarray(shap_res.values)

        if values.ndim == 3:
            values = values[0, :, 1]
        elif values.ndim == 2:
            values = values[0]

        row_dense = transformed_dense[0]

        contributions = []
        for name, val, sv in zip(feature_names, row_dense, values):
            contributions.append({
                "feature": name,
                "feature_value": float(val),
                "shap_value": round(float(sv), 6),
                "absolute_shap": round(abs(float(sv)), 6),
            })

        contributions.sort(key=lambda x: x["absolute_shap"], reverse=True)

        return {
            "status": "success",
            "total_features": len(contributions),
            "top_contributions": contributions[:20],
            "all_contributions": contributions,
            "note": "Dynamic TreeSHAP explanation computed for the submitted applicant inputs.",
        }

    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Failed to calculate local explanation: {exc}",
        ) from exc


@app.post("/predict/loan")
def predict_loan(application: LoanApplication):
    try:
        # Feature names match the training pipeline schema
        loan_data = pd.DataFrame(
            [
                {
                    "revenue": application.revenue,
                    "dti_n": application.dti_n,
                    "loan_amnt": application.loan_amnt,
                    "fico_n": application.fico_n,
                    "experience_c": application.experience_c,
                    "emp_length": application.emp_length,
                    "purpose": application.purpose,
                    "home_ownership_n": application.home_ownership_n,
                    "addr_state": application.addr_state,
                }
            ]
        )

        probabilities = model.predict_proba(loan_data)[0]
        classes = list(model.classes_)
        default_index = classes.index(1)
        default_probability = float(probabilities[default_index])

        # Research prototype policy: manual review band [0.45, 0.55]
        lower_review_bound = 0.45
        upper_review_bound = 0.55

        if lower_review_bound <= default_probability <= upper_review_bound:
            decision = "manual_review"
        elif default_probability > upper_review_bound:
            decision = "higher_predicted_default_risk"
        else:
            decision = "lower_predicted_default_risk"

        return {
            "project": "ARBITER",
            "default_probability": round(default_probability, 4),
            "predicted_default": int(default_probability >= 0.5),
            "decision": decision,
            "review_band": [
                lower_review_bound,
                upper_review_bound,
            ],
            "note": (
                "Research prototype only. This prediction is not a "
                "loan approval or rejection recommendation."
            ),
        }

    except (ValueError, KeyError) as error:
        raise HTTPException(
            status_code=422,
            detail=f"Could not process loan application: {error}",
        ) from error


# ═══════════════════════════════════════════════════════════════
# Indian Loan Research Endpoints  [EXPERIMENTAL]
# These endpoints are completely isolated from the US model.
# ═══════════════════════════════════════════════════════════════

class IndianLoanApplication(BaseModel):
    """Input schema for the Indian loan research model.

    Excluded (will raise validation error if sent):
      - applicant_id      (identifier — never a model feature)
      - default_risk_score (unverified provenance, leakage suspected)
    """
    age:                          int   = Field(..., ge=18, le=70,         description="Applicant age (18–70)")
    gender:                       str   = Field(...,                        description="Gender: Male / Female / Other")
    marital_status:               str   = Field(...,                        description="Marital status")
    city_tier:                    str   = Field(...,                        description="City tier: Tier 1 / Tier 2 / Tier 3")
    state:                        str   = Field(...,                        description="Indian state name")
    employment_type:              str   = Field(...,                        description="Employment type")
    bank_account_vintage_years:   float = Field(..., ge=0, le=40,          description="Years since account opening")
    monthly_income_inr:           float = Field(..., gt=0,                  description="Monthly gross income (INR)")
    existing_loans_count:         int   = Field(..., ge=0, le=20,          description="Number of existing active loans")
    existing_emi_monthly_inr:     float = Field(..., ge=0,                  description="Total existing EMI per month (INR)")
    credit_utilization_ratio_pct: float = Field(..., ge=0, le=100,         description="Credit utilization percentage (0–100)")
    num_credit_inquiries_last_6m: int   = Field(..., ge=0, le=50,          description="Hard credit inquiries in last 6 months")
    late_payments_last_12m:       int   = Field(..., ge=0, le=12,          description="Late payments in last 12 months")
    loan_amount_requested_inr:    float = Field(..., gt=0,                  description="Requested loan amount (INR)")
    loan_tenure_months:           int   = Field(..., ge=6, le=360,         description="Loan tenure in months")
    loan_purpose:                 str   = Field(...,                        description="Loan purpose category")
    collateral_provided:          str   = Field(...,                        description="Collateral provided: Yes / No")
    cibil_score:                  int   = Field(..., ge=300, le=900,       description="CIBIL credit score (300–900)")
    interest_rate_offered_pct:    float = Field(..., ge=0, le=100,         description="Interest rate offered (%)")

    @model_validator(mode="after")
    def validate_categories(self) -> "IndianLoanApplication":
        errors = []
        cat_fields = {
            "gender":            self.gender,
            "marital_status":    self.marital_status,
            "city_tier":         self.city_tier,
            "state":             self.state,
            "employment_type":   self.employment_type,
            "loan_purpose":      self.loan_purpose,
            "collateral_provided": self.collateral_provided,
        }
        for field, value in cat_fields.items():
            valid = INDIAN_VALID_CATEGORIES[field]
            if value not in valid:
                errors.append(f"'{field}' value '{value}' is not recognised. Valid values: {valid}")
        if errors:
            raise ValueError("; ".join(errors))
        return self


@app.get("/predict/indian-loan/metadata")
def indian_loan_metadata():
    """Return Indian model metadata, valid category values, numeric ranges and SHAP importance.

    This endpoint does NOT return a prediction. It is intended for UI form population and
    documentation purposes.

    [EXPERIMENTAL] — dataset provenance unverified.
    """
    return {
        "model": {
            "name":             "indian_loan_logistic_regression",
            "version":          INDIAN_MODEL_VERSION,
            "experiment_label": INDIAN_EXPERIMENT_LABEL,
            "dataset_caveat":   INDIAN_DATASET_CAVEAT,
            "cv_threshold":     INDIAN_CV_THRESHOLD,
            "features_used":    len(INDIAN_BASELINE_FEATURES),
            "excluded":         INDIAN_EXCLUDED_FEATURES,
            "pipeline":         "ColumnTransformer(SimpleImputer+StandardScaler | SimpleImputer+OHE) → LogisticRegression(C=0.1, class_weight=balanced)",
            "model_loaded":     indian_lr_pipeline is not None,
        },
        "valid_categories": INDIAN_VALID_CATEGORIES,
        "numeric_ranges":   INDIAN_NUMERIC_RANGES,
        "shap_top_features": INDIAN_SHAP_TOP_FEATURES,
        "governance": {
            "gender_in_model":           True,
            "fairness_review_required":  True,
            "real_world_validation":     False,
            "calibration_status":        "UNCALIBRATED — raw LR probabilities are over-dispersed (Brier 0.153). Isotonic calibration reduces Brier to ~0.047 but calibrator is not deployed here.",
            "warnings": [
                "EXPERIMENTAL: Dataset provenance is unverified.",
                "Gender is an active model feature. Fairness impact has not been assessed.",
                "Do not use for real lending decisions.",
                "Probability outputs are raw and uncalibrated — not suitable as calibrated scores.",
            ],
        },
    }


@app.post("/predict/indian-loan")
def predict_indian_loan(application: IndianLoanApplication):
    """Indian loan default research prediction.  [EXPERIMENTAL]

    Uses the Stage-3B Logistic Regression pipeline trained on an unverified dataset.
    The output is an experimental research result only — not a lending decision.

    Key governance notes:
    - applicant_id and default_risk_score are excluded and must not be submitted.
    - Gender is an active model feature; fairness review is required.
    - Probabilities are raw (uncalibrated). Isotonic calibration available in research artifacts.
    - The CV threshold 0.7623 was selected on OOF cross-validation, never on test labels.
    - Do not route this output to an automatic loan approval or rejection system.
    """
    if indian_lr_pipeline is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Indian loan research model is not available. "
                f"Expected path: {INDIAN_MODEL_PATH}"
            ),
        )

    try:
        # Build DataFrame strictly from the 19 approved baseline features.
        # This construction explicitly omits applicant_id and default_risk_score.
        input_df = pd.DataFrame([{
            "age":                          application.age,
            "gender":                       application.gender,
            "marital_status":               application.marital_status,
            "city_tier":                    application.city_tier,
            "state":                        application.state,
            "employment_type":              application.employment_type,
            "bank_account_vintage_years":   application.bank_account_vintage_years,
            "monthly_income_inr":           application.monthly_income_inr,
            "existing_loans_count":         application.existing_loans_count,
            "existing_emi_monthly_inr":     application.existing_emi_monthly_inr,
            "credit_utilization_ratio_pct": application.credit_utilization_ratio_pct,
            "num_credit_inquiries_last_6m": application.num_credit_inquiries_last_6m,
            "late_payments_last_12m":       application.late_payments_last_12m,
            "loan_amount_requested_inr":    application.loan_amount_requested_inr,
            "loan_tenure_months":           application.loan_tenure_months,
            "loan_purpose":                 application.loan_purpose,
            "collateral_provided":          application.collateral_provided,
            "cibil_score":                  application.cibil_score,
            "interest_rate_offered_pct":    application.interest_rate_offered_pct,
        }])

        # Verify column set matches expected features exactly
        assert set(input_df.columns) == set(INDIAN_BASELINE_FEATURES), (
            f"Feature mismatch: got {set(input_df.columns)}"
        )
        # Reorder to match training order
        input_df = input_df[INDIAN_BASELINE_FEATURES]

        probabilities = indian_lr_pipeline.predict_proba(input_df)[0]
        default_prob  = float(probabilities[1])
        predicted_cls = int(default_prob >= INDIAN_CV_THRESHOLD)

        if predicted_cls == 1:
            classification = "higher_predicted_default_risk"
        else:
            classification = "lower_predicted_default_risk"

        return {
            "project": "ARBITER",
            "model": {
                "name":             "indian_loan_logistic_regression",
                "version":          INDIAN_MODEL_VERSION,
                "type":             "Logistic Regression (class_weight=balanced, C=0.1, lbfgs)",
                "experiment_label": INDIAN_EXPERIMENT_LABEL,
                "dataset_caveat":   INDIAN_DATASET_CAVEAT,
            },
            "prediction": {
                "default_probability_raw": round(default_prob, 4),
                "predicted_class":         predicted_cls,
                "cv_threshold":            INDIAN_CV_THRESHOLD,
                "classification":          classification,
                "probability_label": (
                    "RAW_UNCALIBRATED — Logistic Regression output. "
                    "Brier score on test set: 0.153 (uncalibrated) vs ~0.047 (isotonic). "
                    "Do not interpret as a validated probability."
                ),
            },
            "features_used":     len(INDIAN_BASELINE_FEATURES),
            "excluded_features": INDIAN_EXCLUDED_FEATURES,
            "governance": {
                "gender_in_model":          True,
                "fairness_review_required": True,
                "real_world_validation":    False,
                "warnings": [
                    "EXPERIMENTAL: Dataset provenance is unverified — possibly synthetic or stratified.",
                    "Gender is an active model feature and requires fairness review before any deployment.",
                    "This output is not a loan approval, rejection, pricing or eligibility decision.",
                    "Do not use for real lending decisions.",
                ],
            },
            "note": (
                "Research prototype only. "
                "All results are experimental and must not be used for real lending decisions."
            ),
        }

    except HTTPException:
        raise
    except (ValueError, AssertionError, KeyError) as error:
        raise HTTPException(
            status_code=422,
            detail=f"Could not process Indian loan application: {error}",
        ) from error
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Internal error in Indian loan model: {error}",
        ) from error
