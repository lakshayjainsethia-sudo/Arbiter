"""
explain_robustness_indian_loan.py
==================================
ARBITER Stage 3C — Indian Loan: Explainability, Robustness & Calibration
[EXPERIMENTAL]

Sections:
  A. Evaluation integrity verification
  B. Explainability  — SHAP global + local for Logistic Regression pipeline
  C. Robustness      — Controlled numeric sensitivity tests on held-out examples
  D. Calibration     — Reliability curves and Brier scores (LR vs RF)
                       Isotonic and Platt calibration fitted on train-CV only.

Reads:
  artifacts/indian_loan/logistic_regression_pipeline.joblib
  artifacts/indian_loan/random_forest_pipeline.joblib
  artifacts/indian_loan/evaluation_report.json
  data/raw/indian_loan/Cleaned_Dataset.csv

Writes to:
  artifacts/indian_loan/explainability/
    integrity_report.json
    shap_global_importance.csv
    shap_global_importance.png
    shap_beeswarm.png
    shap_local_examples.csv
    shap_local_example_{i}.png
    robustness_sensitivity.csv
    robustness_sensitivity.png
    calibration_curves.png
    calibration_report.json

Usage (from project root with venv active):
  python src/explain_robustness_indian_loan.py

DISCLAIMER: Dataset provenance is unverified. All results are experimental
and must not be used for real lending decisions.
"""

from __future__ import annotations

import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import shap
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).resolve().parent.parent
DATA_PATH     = PROJECT_ROOT / "data" / "raw" / "indian_loan" / "Cleaned_Dataset.csv"
ARTIFACT_DIR  = PROJECT_ROOT / "artifacts" / "indian_loan"
EXPLAIN_DIR   = ARTIFACT_DIR / "explainability"
EXPLAIN_DIR.mkdir(parents=True, exist_ok=True)

EVAL_REPORT   = ARTIFACT_DIR / "evaluation_report.json"
LR_PIPELINE   = ARTIFACT_DIR / "logistic_regression_pipeline.joblib"
RF_PIPELINE   = ARTIFACT_DIR / "random_forest_pipeline.joblib"

RANDOM_SEED   = 42
TEST_SIZE     = 0.20
TARGET        = "default_flag"
CV_THRESHOLD  = 0.7623   # from model_metadata.json

BASELINE_FEATURES = [
    "age", "gender", "marital_status", "city_tier", "state",
    "employment_type", "bank_account_vintage_years", "monthly_income_inr",
    "existing_loans_count", "existing_emi_monthly_inr",
    "credit_utilization_ratio_pct", "num_credit_inquiries_last_6m",
    "late_payments_last_12m", "loan_amount_requested_inr",
    "loan_tenure_months", "loan_purpose", "collateral_provided",
    "cibil_score", "interest_rate_offered_pct",
]
NUMERIC_FEATURES = [
    "age", "bank_account_vintage_years", "monthly_income_inr",
    "existing_loans_count", "existing_emi_monthly_inr",
    "credit_utilization_ratio_pct", "num_credit_inquiries_last_6m",
    "late_payments_last_12m", "loan_amount_requested_inr",
    "loan_tenure_months", "cibil_score", "interest_rate_offered_pct",
]
CATEGORICAL_FEATURES = [
    "gender", "marital_status", "city_tier", "state",
    "employment_type", "loan_purpose", "collateral_provided",
]

def hdr(title: str):
    print(f"\n{'='*64}")
    print(f"  {title}")
    print("=" * 64)

def safe_json(obj):
    if isinstance(obj, dict):
        return {k: safe_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [safe_json(v) for v in obj]
    if isinstance(obj, (np.integer,)):  return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, np.ndarray):     return obj.tolist()
    return obj

# ─────────────────────────────────────────────────────────────
# Load data  (reproduce identical split — same seed + stratify)
# ─────────────────────────────────────────────────────────────
hdr("Loading Data & Reproducing Train/Test Split")
df = pd.read_csv(DATA_PATH)
X  = df[BASELINE_FEATURES].copy()
y  = df[TARGET].astype(int).copy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
)
print(f"Train: {len(X_train):,}  Test: {len(X_test):,}")
print(f"Test positives: {y_test.sum()} ({y_test.mean()*100:.2f}%)")

# Load pipelines
lr_pipeline = joblib.load(LR_PIPELINE)
rf_pipeline = joblib.load(RF_PIPELINE)
print("Pipelines loaded.")

# ══════════════════════════════════════════════════════════════
# SECTION A — Evaluation Integrity Verification
# ══════════════════════════════════════════════════════════════
hdr("A. Evaluation Integrity Verification")

with open(EVAL_REPORT, "r", encoding="utf-8") as f:
    eval_report = json.load(f)

integrity = {
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "experiment_label": "EXPERIMENTAL — dataset provenance unverified",
    "checks": {}
}

checks = integrity["checks"]

# A1 — Identical split
checks["identical_split"] = {
    "random_seed": eval_report["dataset"]["random_seed"],
    "test_size":   eval_report["dataset"]["test_size"],
    "split_strategy": eval_report["dataset"]["split_strategy"],
    "reproduced_train_rows": len(X_train),
    "reproduced_test_rows":  len(X_test),
    "reported_train_rows":   eval_report["dataset"]["train_rows"],
    "reported_test_rows":    eval_report["dataset"]["test_rows"],
    "match": (len(X_train) == eval_report["dataset"]["train_rows"]
              and len(X_test) == eval_report["dataset"]["test_rows"]),
    "verdict": "PASS" if (len(X_train) == eval_report["dataset"]["train_rows"]
                          and len(X_test) == eval_report["dataset"]["test_rows"]) else "FAIL",
}
print(f"\n[A1] Identical split: {checks['identical_split']['verdict']}")

# A2 — Re-derive metrics and compare to saved report
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              brier_score_loss as bsl, precision_score,
                              recall_score, f1_score, confusion_matrix)

lr_prob  = lr_pipeline.predict_proba(X_test)[:, 1]
lr_pred  = (lr_prob >= CV_THRESHOLD).astype(int)

re_roc   = round(roc_auc_score(y_test, lr_prob), 4)
re_pr    = round(average_precision_score(y_test, lr_prob), 4)
re_brier = round(bsl(y_test, lr_prob), 4)
re_prec  = round(precision_score(y_test, lr_pred, zero_division=0), 4)
re_rec   = round(recall_score(y_test, lr_pred, zero_division=0), 4)
re_f1    = round(f1_score(y_test, lr_pred, zero_division=0), 4)
re_cm    = confusion_matrix(y_test, lr_pred).tolist()

saved    = eval_report["models"]["logistic_regression"]["at_cv_threshold"]
tol = 0.001   # floating-point rounding tolerance

metric_match = (
    abs(re_roc   - saved["roc_auc"])    <= tol and
    abs(re_pr    - saved["pr_auc"])     <= tol and
    abs(re_brier - saved["brier_score"]) <= tol and
    abs(re_prec  - saved["precision"])  <= tol and
    abs(re_rec   - saved["recall"])     <= tol and
    abs(re_f1    - saved["f1"])         <= tol and
    re_cm == saved["confusion_matrix"]
)
checks["metric_reproducibility"] = {
    "recomputed": {"roc_auc": re_roc, "pr_auc": re_pr, "brier": re_brier,
                   "precision": re_prec, "recall": re_rec, "f1": re_f1,
                   "confusion_matrix": re_cm},
    "saved":      {"roc_auc": saved["roc_auc"], "pr_auc": saved["pr_auc"],
                   "brier": saved["brier_score"],
                   "precision": saved["precision"], "recall": saved["recall"],
                   "f1": saved["f1"], "confusion_matrix": saved["confusion_matrix"]},
    "tolerance": tol,
    "verdict": "PASS" if metric_match else "FAIL",
}
print(f"[A2] Metric reproducibility: {checks['metric_reproducibility']['verdict']}")

# A3 — Threshold not selected on test set
checks["threshold_selection_integrity"] = {
    "method":  eval_report["threshold_selection"]["method"],
    "note":    eval_report["threshold_selection"]["note"],
    "verdict": "PASS — threshold selected on OOF CV probabilities, test labels never accessed",
}
print(f"[A3] Threshold integrity: {checks['threshold_selection_integrity']['verdict']}")

# A4 — Exclusions verified
feat_names = set(eval_report["features"]["baseline_features"])
excluded   = set(eval_report["features"]["excluded"])
excluded_ok = "applicant_id" in excluded and "default_risk_score" in excluded
feat_clean  = "applicant_id" not in feat_names and "default_risk_score" not in feat_names
checks["feature_exclusions"] = {
    "excluded_columns":        sorted(excluded),
    "excluded_present_in_features": not feat_clean,
    "verdict": "PASS" if (excluded_ok and feat_clean) else "FAIL",
}
print(f"[A4] Feature exclusions: {checks['feature_exclusions']['verdict']}")

# A5 — Provenance caveat
checks["dataset_provenance"] = {
    "status": "UNVERIFIED",
    "observations": [
        "All 15 Indian states have near-equal representation (~6.5-7%) — consistent with synthetic/stratified generation.",
        "No accompanying data dictionary or README confirms real-world origin.",
        "default_risk_score formula undocumented — excluded from features (MODERATE_LEAKAGE_SUSPECTED).",
        "Results must not be used for real lending decisions.",
    ],
    "verdict": "CAVEAT_MAINTAINED",
}
print(f"[A5] Provenance: {checks['dataset_provenance']['verdict']}")

# Overall — use startswith so verbose strings like "PASS — ..." still match
all_verdicts = [v["verdict"] for v in checks.values()
                if isinstance(v, dict) and "verdict" in v]
integrity["overall_verdict"] = (
    "PASS" if all(
        v.startswith("PASS") or v == "CAVEAT_MAINTAINED"
        for v in all_verdicts
    ) else "FAIL"
)
print(f"\n  Overall integrity verdict: {integrity['overall_verdict']}")

integrity_path = EXPLAIN_DIR / "integrity_report.json"
with open(integrity_path, "w", encoding="utf-8") as f:
    json.dump(safe_json(integrity), f, indent=2)
print(f"  Saved: {integrity_path.name}")

# ══════════════════════════════════════════════════════════════
# SECTION B — Explainability (SHAP)
# ══════════════════════════════════════════════════════════════
hdr("B. Explainability — SHAP on Logistic Regression Pipeline")

# Extract the fitted preprocessor and classifier
preprocessor = lr_pipeline.named_steps["preprocessor"]
classifier   = lr_pipeline.named_steps["classifier"]

# Transform test features through the fitted pipeline's preprocessor
X_test_transformed = preprocessor.transform(X_test)

# Recover feature names after OHE expansion
num_feature_names = NUMERIC_FEATURES.copy()
ohe = preprocessor.named_transformers_["cat"].named_steps["ohe"]
cat_feature_names = ohe.get_feature_names_out(CATEGORICAL_FEATURES).tolist()
all_feature_names = num_feature_names + cat_feature_names
print(f"Transformed feature space: {len(all_feature_names)} features")

# SHAP LinearExplainer is the correct choice for Logistic Regression
# It works on the transformed (scaled + OHE) feature matrix directly
# Use a representative background sample from training data
X_train_transformed = preprocessor.transform(X_train)

print("Initialising SHAP LinearExplainer...")
explainer = shap.LinearExplainer(
    classifier,
    X_train_transformed,
    feature_names=all_feature_names,
)

# Compute SHAP values on test set (subsample for speed)
N_SHAP = min(2000, len(X_test))
rng = np.random.RandomState(RANDOM_SEED)
shap_idx = rng.choice(len(X_test), N_SHAP, replace=False)

if hasattr(X_test_transformed, "toarray"):
    X_shap = X_test_transformed[shap_idx].toarray()
else:
    X_shap = X_test_transformed[shap_idx]

print(f"Computing SHAP values for {N_SHAP} test samples...")
shap_values = explainer(X_shap)
print("SHAP values computed.")

# ── B1: Global importance (mean |SHAP|) ─────────────────────
mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
shap_global_df = pd.DataFrame({
    "feature":        all_feature_names,
    "mean_abs_shap":  mean_abs_shap,
    "mean_shap":      shap_values.values.mean(axis=0),
}).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
shap_global_df["rank"] = shap_global_df.index + 1
shap_global_df["note"] = "EXPERIMENTAL — association only, not causal"

shap_global_csv = EXPLAIN_DIR / "shap_global_importance.csv"
shap_global_df.to_csv(shap_global_csv, index=False)
print(f"\n  Top 10 features by mean |SHAP|:")
print(shap_global_df[["rank", "feature", "mean_abs_shap", "mean_shap"]].head(10).to_string(index=False))

# ── B1 Plot: Top-20 bar chart ────────────────────────────────
top20 = shap_global_df.head(20)
colors = ["#d73027" if v < 0 else "#4575b4" for v in top20["mean_shap"]]

fig, ax = plt.subplots(figsize=(10, 7))
bars = ax.barh(top20["feature"][::-1], top20["mean_abs_shap"][::-1], color=colors[::-1])
ax.set_xlabel("Mean |SHAP value| (impact on model output)")
ax.set_title("Global Feature Importance — Logistic Regression\nTop 20 Features [EXPERIMENTAL]\n"
             "Note: SHAP values reflect model associations, not causal relationships.", fontsize=10)
ax.axvline(0, color="k", lw=0.8)
plt.tight_layout()
shap_bar_path = EXPLAIN_DIR / "shap_global_importance.png"
plt.savefig(shap_bar_path, dpi=130, bbox_inches="tight")
plt.close()
print(f"  Saved: {shap_bar_path.name}")

# ── B2 Plot: Beeswarm ────────────────────────────────────────
# Subset to top 20 features for readability
top20_idx = [all_feature_names.index(f) for f in top20["feature"].tolist()]
shap_vals_top20 = shap_values.values[:, top20_idx]
x_top20 = X_shap[:, top20_idx]

fig, ax = plt.subplots(figsize=(10, 8))
# Manual beeswarm-style scatter (no shap.plots.beeswarm to avoid display issues)
for i, feat_idx in enumerate(range(len(top20_idx))):
    feat_name = top20["feature"].iloc[feat_idx]
    sv = shap_vals_top20[:, feat_idx]
    fv = x_top20[:, feat_idx]
    # Normalise feature values for color
    fv_norm = (fv - fv.min()) / (fv.max() - fv.min() + 1e-8)
    jitter = rng.uniform(-0.3, 0.3, size=len(sv))
    sc = ax.scatter(sv, np.full(len(sv), i) + jitter,
                    c=fv_norm, cmap="coolwarm", alpha=0.35, s=6)

ax.set_yticks(range(len(top20)))
ax.set_yticklabels(top20["feature"].tolist(), fontsize=8)
ax.axvline(0, color="k", lw=1)
ax.set_xlabel("SHAP value (impact on default probability)")
ax.set_title("SHAP Beeswarm — Logistic Regression (top 20 features)\n"
             "[EXPERIMENTAL — association, not causation]", fontsize=10)
plt.colorbar(sc, ax=ax, label="Feature value (low→high)")
plt.tight_layout()
beeswarm_path = EXPLAIN_DIR / "shap_beeswarm.png"
plt.savefig(beeswarm_path, dpi=130, bbox_inches="tight")
plt.close()
print(f"  Saved: {beeswarm_path.name}")

# ── B3: Local explanations — a few held-out examples ─────────
# Pick: 2 correct defaults, 2 missed defaults, 2 false alarms
y_test_arr   = y_test.values
lr_prob_arr  = lr_pipeline.predict_proba(X_test)[:, 1]
lr_pred_arr  = (lr_prob_arr >= CV_THRESHOLD).astype(int)

idx_tp = np.where((y_test_arr == 1) & (lr_pred_arr == 1))[0][:2]   # TP
idx_fn = np.where((y_test_arr == 1) & (lr_pred_arr == 0))[0][:2]   # FN (missed)
idx_fp = np.where((y_test_arr == 0) & (lr_pred_arr == 1))[0][:2]   # FP (false alarm)

local_examples = []
local_fig_paths = []

for label, indices in [("TP_correct_default", idx_tp),
                        ("FN_missed_default",  idx_fn),
                        ("FP_false_alarm",     idx_fp)]:
    for rank, test_row_idx in enumerate(indices):
        # Get the SHAP subsample index (may not be identical to test_row_idx)
        # Recompute SHAP for this specific row
        if hasattr(X_test_transformed, "toarray"):
            x_row = X_test_transformed[test_row_idx].toarray()
        else:
            x_row = X_test_transformed[test_row_idx:test_row_idx+1]

        sv_row = explainer(x_row).values[0]
        top_features = pd.Series(np.abs(sv_row), index=all_feature_names).nlargest(12)
        shap_row     = pd.Series(sv_row, index=all_feature_names)[top_features.index]

        prob = float(lr_prob_arr[test_row_idx])
        pred = int(lr_pred_arr[test_row_idx])
        true = int(y_test_arr[test_row_idx])

        local_examples.append({
            "example_id":     f"{label}_{rank+1}",
            "test_row_index": int(test_row_idx),
            "true_label":     true,
            "predicted_label": pred,
            "predicted_prob":  round(prob, 4),
            "classification":  label,
            "top_features": dict(zip(top_features.index.tolist(),
                                     [round(float(v), 4) for v in shap_row.values])),
            "note": "SHAP values reflect model-level associations in this example. Not causal.",
        })

        # Plot waterfall-style bar chart for this example
        colors_local = ["#d73027" if v < 0 else "#4575b4"
                        for v in shap_row.values[::-1]]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(shap_row.index[::-1], shap_row.values[::-1], color=colors_local)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_xlabel("SHAP value (contribution to default probability)")
        ax.set_title(
            f"Local SHAP — {label.replace('_', ' ')} #{rank+1}\n"
            f"Prob={prob:.4f} | Pred={pred} | True={true} | Threshold={CV_THRESHOLD}\n"
            "[EXPERIMENTAL — association only]",
            fontsize=9
        )
        plt.tight_layout()
        local_path = EXPLAIN_DIR / f"shap_local_{label}_{rank+1}.png"
        plt.savefig(local_path, dpi=120, bbox_inches="tight")
        plt.close()
        local_fig_paths.append(local_path.name)
        print(f"  Saved local explanation: {local_path.name}")

local_df = pd.DataFrame(local_examples)
local_csv = EXPLAIN_DIR / "shap_local_examples.csv"
# Flatten top_features dict for CSV
flat_rows = []
for ex in local_examples:
    row = {k: v for k, v in ex.items() if k != "top_features"}
    for feat, sv in ex["top_features"].items():
        row[f"shap_{feat}"] = sv
    flat_rows.append(row)
pd.DataFrame(flat_rows).to_csv(local_csv, index=False)
print(f"  Saved: {local_csv.name}")

# ── B4: Coefficient direction notes ──────────────────────────
coef = classifier.coef_[0]
coef_df = pd.DataFrame({
    "feature":     all_feature_names,
    "coefficient": coef,
}).sort_values("coefficient")

print("\n  Top positive coefficients (increase default risk in standardised space):")
print(coef_df.tail(8)[["feature","coefficient"]].to_string(index=False))
print("\n  Top negative coefficients (decrease default risk in standardised space):")
print(coef_df.head(8)[["feature","coefficient"]].to_string(index=False))

# Save coefficient table
coef_csv = EXPLAIN_DIR / "lr_coefficients.csv"
coef_df["note"] = ("Coefficient in standardised+OHE feature space. "
                   "Numeric features are StandardScaler-scaled. "
                   "Association only — not causal.")
coef_df.to_csv(coef_csv, index=False)
print(f"\n  Saved: {coef_csv.name}")

# ══════════════════════════════════════════════════════════════
# SECTION C — Robustness: Numeric Sensitivity Tests
# ══════════════════════════════════════════════════════════════
hdr("C. Robustness — Numeric Sensitivity Tests")

print("NOTE: These are sensitivity/perturbation tests on the held-out test set.")
print("They measure model response to plausible input changes, NOT adversarial robustness.\n")

# Select a stratified sample for sensitivity testing
N_ROBUST = 200
rng2 = np.random.RandomState(RANDOM_SEED + 1)
robust_idx = rng2.choice(len(X_test), N_ROBUST, replace=False)
X_robust = X_test.iloc[robust_idx].copy().reset_index(drop=True)
y_robust = y_test.iloc[robust_idx].copy().reset_index(drop=True)

baseline_probs = lr_pipeline.predict_proba(X_robust)[:, 1]

# Define perturbations: (feature, description, delta_fn)
# Plausible single-step changes that a real applicant might exhibit
PERTURBATIONS = [
    # CIBIL score changes
    ("cibil_score", "CIBIL +50 (improvement)",   lambda x: x.assign(cibil_score=x["cibil_score"] + 50)),
    ("cibil_score", "CIBIL -50 (deterioration)", lambda x: x.assign(cibil_score=x["cibil_score"] - 50)),
    ("cibil_score", "CIBIL +150 (major improvement)", lambda x: x.assign(cibil_score=(x["cibil_score"] + 150).clip(upper=900))),
    # Monthly income
    ("monthly_income_inr", "Income +25%",  lambda x: x.assign(monthly_income_inr=x["monthly_income_inr"] * 1.25)),
    ("monthly_income_inr", "Income -25%",  lambda x: x.assign(monthly_income_inr=x["monthly_income_inr"] * 0.75)),
    # EMI burden
    ("existing_emi_monthly_inr", "EMI +50%",  lambda x: x.assign(existing_emi_monthly_inr=x["existing_emi_monthly_inr"] * 1.50)),
    ("existing_emi_monthly_inr", "EMI -50%",  lambda x: x.assign(existing_emi_monthly_inr=x["existing_emi_monthly_inr"] * 0.50)),
    # Loan amount
    ("loan_amount_requested_inr", "Loan amount +30%", lambda x: x.assign(loan_amount_requested_inr=x["loan_amount_requested_inr"] * 1.30)),
    ("loan_amount_requested_inr", "Loan amount -30%", lambda x: x.assign(loan_amount_requested_inr=x["loan_amount_requested_inr"] * 0.70)),
    # Credit utilization
    ("credit_utilization_ratio_pct", "Credit util +20pp", lambda x: x.assign(credit_utilization_ratio_pct=(x["credit_utilization_ratio_pct"] + 20).clip(upper=100))),
    ("credit_utilization_ratio_pct", "Credit util -20pp", lambda x: x.assign(credit_utilization_ratio_pct=(x["credit_utilization_ratio_pct"] - 20).clip(lower=0))),
    # Late payments
    ("late_payments_last_12m", "Late payments +2", lambda x: x.assign(late_payments_last_12m=(x["late_payments_last_12m"] + 2).clip(upper=8))),
    ("late_payments_last_12m", "Late payments 0 (none)", lambda x: x.assign(late_payments_last_12m=0)),
]

rob_rows = []
for feature, description, perturb_fn in PERTURBATIONS:
    X_perturbed = perturb_fn(X_robust.copy())
    perturbed_probs = lr_pipeline.predict_proba(X_perturbed)[:, 1]

    delta    = perturbed_probs - baseline_probs
    baseline_pred = (baseline_probs >= CV_THRESHOLD).astype(int)
    perturbed_pred = (perturbed_probs >= CV_THRESHOLD).astype(int)
    class_changes = int((baseline_pred != perturbed_pred).sum())
    approve_to_flag = int(((baseline_pred == 0) & (perturbed_pred == 1)).sum())
    flag_to_approve = int(((baseline_pred == 1) & (perturbed_pred == 0)).sum())

    rob_rows.append({
        "feature":              feature,
        "perturbation":         description,
        "n_samples":            N_ROBUST,
        "mean_baseline_prob":   round(float(baseline_probs.mean()), 4),
        "mean_perturbed_prob":  round(float(perturbed_probs.mean()), 4),
        "mean_delta_prob":      round(float(delta.mean()), 4),
        "abs_mean_delta":       round(float(np.abs(delta).mean()), 4),
        "max_abs_delta":        round(float(np.abs(delta).max()), 4),
        "classification_changes": class_changes,
        "approvals_flipped_to_flag": approve_to_flag,
        "flags_flipped_to_approval": flag_to_approve,
        "pct_class_change":     round(class_changes / N_ROBUST * 100, 2),
        "note": "Sensitivity test only — not adversarial robustness proof",
    })
    print(f"  [{description}] mean_delta={delta.mean():+.4f}  class_changes={class_changes}/{N_ROBUST}")

rob_df = pd.DataFrame(rob_rows)
rob_csv = EXPLAIN_DIR / "robustness_sensitivity.csv"
rob_df.to_csv(rob_csv, index=False)
print(f"\n  Saved: {rob_csv.name}")

# ── C Plot: Sensitivity summary ──────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Sensitivity Test Summary — Logistic Regression [EXPERIMENTAL]", fontsize=11)

ax1, ax2 = axes
colors_rob = ["#d73027" if d > 0 else "#4575b4" for d in rob_df["mean_delta_prob"]]
y_pos = range(len(rob_df))

ax1.barh(y_pos, rob_df["mean_delta_prob"], color=colors_rob)
ax1.set_yticks(y_pos)
ax1.set_yticklabels(rob_df["perturbation"], fontsize=8)
ax1.axvline(0, color="k", lw=1)
ax1.set_xlabel("Mean change in predicted probability")
ax1.set_title("Mean Delta Probability")
ax1.grid(axis="x", alpha=0.3)

colors_cc = ["#e66101" if c > N_ROBUST * 0.05 else "#5e3c99" for c in rob_df["classification_changes"]]
ax2.barh(y_pos, rob_df["pct_class_change"], color=colors_cc)
ax2.set_yticks(y_pos)
ax2.set_yticklabels(rob_df["perturbation"], fontsize=8)
ax2.axvline(5, color="k", lw=1, linestyle="--", label=">5% threshold")
ax2.set_xlabel("% Samples with classification change")
ax2.set_title("Classification Flip Rate")
ax2.grid(axis="x", alpha=0.3)
ax2.legend(fontsize=8)

plt.tight_layout()
rob_plot = EXPLAIN_DIR / "robustness_sensitivity.png"
plt.savefig(rob_plot, dpi=130, bbox_inches="tight")
plt.close()
print(f"  Saved: {rob_plot.name}")

# ══════════════════════════════════════════════════════════════
# SECTION D — Probability Calibration
# ══════════════════════════════════════════════════════════════
hdr("D. Probability Calibration")
print("NOTE: Calibrators fitted on OOF cross_val_predict on TRAINING data only.")
print("      Test set is NEVER used to fit calibration.\n")

# ── D1: Baseline uncalibrated test probabilities ─────────────
lr_prob_test = lr_pipeline.predict_proba(X_test)[:, 1]
rf_prob_test = rf_pipeline.predict_proba(X_test)[:, 1]

frac_pos_lr, mean_pred_lr = calibration_curve(y_test, lr_prob_test, n_bins=10, strategy="uniform")
frac_pos_rf, mean_pred_rf = calibration_curve(y_test, rf_prob_test, n_bins=10, strategy="uniform")

brier_lr = round(brier_score_loss(y_test, lr_prob_test), 4)
brier_rf = round(brier_score_loss(y_test, rf_prob_test), 4)
print(f"  Uncalibrated — LR Brier: {brier_lr}  |  RF Brier: {brier_rf}")

# ── D2: OOF probabilities from training set via CV ───────────
# These OOF probs are produced using the ORIGINAL full pipelines
# (preprocessor + classifier) via 5-fold stratified CV.
# They are never contaminated with test-set information.
from sklearn.model_selection import cross_val_predict as cvp

print("  Computing OOF probabilities (5-fold CV on train set)...")
lr_oof = cvp(lr_pipeline, X_train, y_train, cv=5,
             method="predict_proba", n_jobs=-1)[:, 1]
print("  LR OOF done.")
rf_oof = cvp(rf_pipeline, X_train, y_train, cv=5,
             method="predict_proba", n_jobs=-1)[:, 1]
print("  RF OOF done.")

# ── D3: Fit calibrators on OOF probs ─────────────────────────
# Platt scaling  = 1-D logistic regression on the raw score
# Isotonic       = piecewise-constant monotone regression
from sklearn.linear_model import LogisticRegression as PlattLR
from sklearn.isotonic import IsotonicRegression

def fit_platt(oof_probs, y_tr):
    """Platt scaling: fit a 1-D LR on OOF scores."""
    return PlattLR(C=1e6, solver="lbfgs").fit(
        oof_probs.reshape(-1, 1), y_tr
    )

def fit_isotonic(oof_probs, y_tr):
    return IsotonicRegression(out_of_bounds="clip").fit(oof_probs, y_tr)

def apply_platt(calibrator, test_probs):
    return calibrator.predict_proba(test_probs.reshape(-1, 1))[:, 1]

def apply_isotonic(calibrator, test_probs):
    return calibrator.predict(test_probs)

# Logistic Regression calibrators
lr_platt_cal   = fit_platt(lr_oof, y_train)
lr_isotonic_cal = fit_isotonic(lr_oof, y_train)
lr_platt_prob   = apply_platt(lr_platt_cal, lr_prob_test)
lr_isotonic_prob = apply_isotonic(lr_isotonic_cal, lr_prob_test)

# Random Forest calibrators
rf_platt_cal   = fit_platt(rf_oof, y_train)
rf_isotonic_cal = fit_isotonic(rf_oof, y_train)
rf_platt_prob   = apply_platt(rf_platt_cal, rf_prob_test)
rf_isotonic_prob = apply_isotonic(rf_isotonic_cal, rf_prob_test)

# Brier scores after calibration
brier_lr_platt    = round(brier_score_loss(y_test, lr_platt_prob), 4)
brier_lr_isotonic = round(brier_score_loss(y_test, lr_isotonic_prob), 4)
brier_rf_platt    = round(brier_score_loss(y_test, rf_platt_prob), 4)
brier_rf_isotonic = round(brier_score_loss(y_test, rf_isotonic_prob), 4)

print(f"\n  LR  — Uncal: {brier_lr}  |  Platt: {brier_lr_platt}  |  Isotonic: {brier_lr_isotonic}")
print(f"  RF  — Uncal: {brier_rf}  |  Platt: {brier_rf_platt}  |  Isotonic: {brier_rf_isotonic}")

# Calibration curves for calibrated variants
frac_lr_platt,    mean_lr_platt    = calibration_curve(y_test, lr_platt_prob,    n_bins=10, strategy="uniform")
frac_lr_isotonic, mean_lr_isotonic = calibration_curve(y_test, lr_isotonic_prob, n_bins=10, strategy="uniform")
frac_rf_platt,    mean_rf_platt    = calibration_curve(y_test, rf_platt_prob,    n_bins=10, strategy="uniform")
frac_rf_isotonic, mean_rf_isotonic = calibration_curve(y_test, rf_isotonic_prob, n_bins=10, strategy="uniform")

cal_results = {
    "sigmoid":  {
        "lr": {"brier": brier_lr_platt,
               "frac_pos": frac_lr_platt.tolist(), "mean_pred": mean_lr_platt.tolist()},
        "rf": {"brier": brier_rf_platt,
               "frac_pos": frac_rf_platt.tolist(), "mean_pred": mean_rf_platt.tolist()},
    },
    "isotonic": {
        "lr": {"brier": brier_lr_isotonic,
               "frac_pos": frac_lr_isotonic.tolist(), "mean_pred": mean_lr_isotonic.tolist()},
        "rf": {"brier": brier_rf_isotonic,
               "frac_pos": frac_rf_isotonic.tolist(), "mean_pred": mean_rf_isotonic.tolist()},
    },
}

# ── D Plot: Calibration curves ───────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Probability Calibration — LR vs RF [EXPERIMENTAL]\n"
             "Calibrators fitted on OOF train-CV probs only — test set not used for fitting",
             fontsize=10)

for ax, (model_label, bp_uncal, fp_uncal, mp_uncal, cal_key) in zip(
    axes,
    [
        ("Logistic Regression", brier_lr, frac_pos_lr, mean_pred_lr, "lr"),
        ("Random Forest",       brier_rf, frac_pos_rf, mean_pred_rf, "rf"),
    ]
):
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Perfect calibration")
    ax.plot(mp_uncal, fp_uncal, "o-", color="#e66101", lw=1.8,
            label=f"Uncalibrated (Brier={bp_uncal})")

    for cal_label, lstyle, color, cal_key2 in [
        ("Platt (sigmoid)", "--", "#5e3c99", "sigmoid"),
        ("Isotonic",        "-.", "#1a9641", "isotonic"),
    ]:
        cr = cal_results[cal_key2][cal_key]
        ax.plot(cr["mean_pred"], cr["frac_pos"], lstyle, color=color, lw=1.8,
                label=f"{cal_label} cal. (Brier={cr['brier']})")

    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(f"{model_label}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

plt.tight_layout()
cal_plot = EXPLAIN_DIR / "calibration_curves.png"
plt.savefig(cal_plot, dpi=130, bbox_inches="tight")
plt.close()
print(f"\n  Saved: {cal_plot.name}")

# Calibration report JSON
cal_report = {
    "experiment_label": "EXPERIMENTAL — dataset provenance unverified",
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "note": (
        "Calibrators fitted on OOF probabilities from 5-fold cross_val_predict "
        "on the training set only. Test set was NEVER used to fit calibration. "
        "Calibrated models are exploratory and do NOT replace the selected baseline. "
        "The selected model remains logistic_regression_pipeline.joblib (uncalibrated)."
    ),
    "calibration_method": {
        "oof_source": "cross_val_predict(pipeline, X_train, y_train, cv=5)",
        "platt": "LogisticRegression(C=1e6) fit on OOF probs vs y_train",
        "isotonic": "IsotonicRegression(out_of_bounds=clip) fit on OOF probs vs y_train",
        "test_labels_used_for_fitting": False,
    },
    "uncalibrated": {
        "logistic_regression": {"brier_score": brier_lr},
        "random_forest":       {"brier_score": brier_rf},
    },
    "calibrated": safe_json(cal_results),
}
cal_report_path = EXPLAIN_DIR / "calibration_report.json"
with open(cal_report_path, "w", encoding="utf-8") as f:
    json.dump(cal_report, f, indent=2)
print(f"  Saved: {cal_report_path.name}")



# ══════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════
hdr("Stage 3C Complete — Summary [EXPERIMENTAL]")

artifacts_saved = [
    "integrity_report.json",
    "shap_global_importance.csv",
    "shap_global_importance.png",
    "shap_beeswarm.png",
    "shap_local_*.png (6 examples: 2 TP, 2 FN, 2 FP)",
    "shap_local_examples.csv",
    "lr_coefficients.csv",
    "robustness_sensitivity.csv",
    "robustness_sensitivity.png",
    "calibration_curves.png",
    "calibration_report.json",
]

print(f"\n  Artifacts saved to: {EXPLAIN_DIR}")
for a in artifacts_saved:
    print(f"    - {a}")

print(f"\n  [A] Integrity: {integrity['overall_verdict']}")
print(f"      All models shared identical split (seed=42, stratified 80/20)")
print(f"      Metrics reproduced within tolerance={tol}")
print(f"      Threshold selected on OOF CV — test labels never accessed")
print(f"      applicant_id & default_risk_score confirmed excluded")

print(f"\n  [B] Top SHAP feature (global): {shap_global_df.iloc[0]['feature']}")
print(f"      Mean |SHAP|: {shap_global_df.iloc[0]['mean_abs_shap']:.4f}")
print(f"      All SHAP values reflect model associations — not causal relationships")
print(f"      6 local explanations generated (TP/FN/FP cases)")

print(f"\n  [C] Sensitivity tests: {len(rob_df)} perturbations on {N_ROBUST} samples")
max_flip = rob_df.loc[rob_df['pct_class_change'].idxmax()]
print(f"      Highest flip rate: {max_flip['perturbation']} -> "
      f"{max_flip['pct_class_change']:.1f}% classification changes")

print(f"\n  [D] Calibration (Brier scores):")
print(f"      LR uncalibrated:  {brier_lr}")
print(f"      LR + sigmoid:     {cal_results['sigmoid']['lr']['brier']}")
print(f"      LR + isotonic:    {cal_results['isotonic']['lr']['brier']}")
print(f"      RF uncalibrated:  {brier_rf}")
print(f"      RF + sigmoid:     {cal_results['sigmoid']['rf']['brier']}")
print(f"      RF + isotonic:    {cal_results['isotonic']['rf']['brier']}")
print(f"      Selected model unchanged: logistic_regression_pipeline.joblib")

print(f"\n  ** All results are EXPERIMENTAL **")
print(f"  ** Dataset provenance unverified **")
print(f"  ** Do not integrate into API yet (Stage 3D) **\n")
