"""
train_indian_loan_models.py
============================
ARBITER Stage 3B — Indian Loan Default Model Training  [EXPERIMENTAL]

Trains three baseline classifiers on the audit-approved feature set:
  1. Logistic Regression  (class_weight='balanced')
  2. Random Forest        (class_weight='balanced_subsample')
  3. XGBoost              (scale_pos_weight)

All models share an identical stratified 80/20 split (seed=42).
Preprocessing is fitted on training data only.

NOTE: Dataset provenance is unverified. Results are experimental and
must not be used for any real lending decisions.

Outputs (artifacts/indian_loan/):
  model_comparison.csv
  evaluation_report.json
  confusion_matrix_<model>.png
  roc_pr_curves.png
  <model_name>_pipeline.joblib
  model_metadata.json

Usage (from project root with venv active):
  python src/train_indian_loan_models.py
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe on all platforms
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH    = PROJECT_ROOT / "data" / "raw" / "indian_loan" / "Cleaned_Dataset.csv"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "indian_loan"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED  = 42
TEST_SIZE    = 0.20
TARGET       = "default_flag"

# Audit-approved features (from audit_summary.json)
EXCLUDE_COLS = ["applicant_id", "default_risk_score", TARGET]
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

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def hdr(title: str):
    print(f"\n{'='*62}")
    print(f"  {title}")
    print('='*62)


def subhdr(msg: str):
    print(f"\n  -- {msg}")


def safe_json(obj):
    """Recursively convert numpy scalars to Python natives for JSON."""
    if isinstance(obj, dict):
        return {k: safe_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [safe_json(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# ─────────────────────────────────────────────────────────────
# 1. Load & split
# ─────────────────────────────────────────────────────────────
hdr("1. Loading Data")
if not DATA_PATH.exists():
    sys.exit(f"ERROR: Dataset not found at {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
print(f"Loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")

# Verify all expected features present
missing_cols = [c for c in BASELINE_FEATURES if c not in df.columns]
if missing_cols:
    sys.exit(f"ERROR: Expected feature columns missing from CSV: {missing_cols}")

X = df[BASELINE_FEATURES].copy()
y = df[TARGET].astype(int).copy()

print(f"Features: {len(BASELINE_FEATURES)} | Target: '{TARGET}'")
print(f"Class distribution — 0: {(y==0).sum():,} | 1: {(y==1).sum():,}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_SEED,
    stratify=y,
)
print(f"\nTrain: {len(X_train):,} rows | Test: {len(X_test):,} rows")
print(f"Train positives: {y_train.sum():,} ({y_train.mean()*100:.2f}%)")
print(f"Test  positives: {y_test.sum():,}  ({y_test.mean()*100:.2f}%)")

test_support   = int(y_test.sum())
test_prevalence = float(y_test.mean())

# ─────────────────────────────────────────────────────────────
# 2. Preprocessing pipeline
# ─────────────────────────────────────────────────────────────
hdr("2. Building Preprocessor")

numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler()),
])

categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ohe",     OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float32)),
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipe,      NUMERIC_FEATURES),
        ("cat", categorical_pipe,  CATEGORICAL_FEATURES),
    ],
    remainder="drop",
    sparse_threshold=0.3,
)
print("ColumnTransformer built.")

# ─────────────────────────────────────────────────────────────
# 3. Class-imbalance weight for XGBoost
# ─────────────────────────────────────────────────────────────
neg_count = int((y_train == 0).sum())
pos_count = int((y_train == 1).sum())
scale_pos_weight = neg_count / pos_count
print(f"\nClass-weight ratio (neg/pos): {scale_pos_weight:.2f}")

# ─────────────────────────────────────────────────────────────
# 4. Define classifiers
# ─────────────────────────────────────────────────────────────
hdr("3. Defining Classifiers")

try:
    import xgboost as xgb
    xgb_available = True
    print(f"XGBoost {xgb.__version__} available.")
except ImportError:
    xgb_available = False
    print("WARNING: xgboost not installed. XGBoost model will be skipped.")

classifiers: dict[str, object] = {
    "logistic_regression": LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        solver="lbfgs",
        C=0.1,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    ),
    "random_forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=10,
        class_weight="balanced_subsample",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    ),
}

if xgb_available:
    classifiers["xgboost"] = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbosity=0,
    )


# ─────────────────────────────────────────────────────────────
# 5. Threshold selection via 5-fold CV on training set only
# ─────────────────────────────────────────────────────────────
def select_threshold_cv(pipeline: Pipeline, X_tr: pd.DataFrame,
                         y_tr: pd.Series, n_splits: int = 5) -> float:
    """
    Select threshold that maximises F1 on OOF probabilities from CV.
    Uses only training data — test set is never touched.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    oof_probs = cross_val_predict(pipeline, X_tr, y_tr, cv=cv,
                                  method="predict_proba", n_jobs=1)[:, 1]
    thresholds = np.linspace(0.05, 0.80, 200)
    best_t, best_f1 = 0.5, 0.0
    for t in thresholds:
        preds = (oof_probs >= t).astype(int)
        f1 = f1_score(y_tr, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t  = t
    return float(round(best_t, 4))


# ─────────────────────────────────────────────────────────────
# 6. Train, evaluate, save
# ─────────────────────────────────────────────────────────────
hdr("4. Training & Evaluation  [EXPERIMENTAL]")

results: dict[str, dict] = {}
trained_pipelines: dict[str, Pipeline] = {}
all_probas: dict[str, np.ndarray] = {}

DISPLAY_NAMES = {
    "logistic_regression": "Logistic Regression",
    "random_forest":       "Random Forest",
    "xgboost":             "XGBoost",
}

for name, clf in classifiers.items():
    display = DISPLAY_NAMES[name]
    subhdr(f"Training: {display}")

    full_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier",   clf),
    ])

    # --- Threshold selection (train-only CV) ---
    print(f"    Selecting threshold via 5-fold CV on training data...")
    t0 = time.time()
    cv_threshold = select_threshold_cv(full_pipeline, X_train, y_train)
    cv_time = time.time() - t0
    print(f"    CV-selected threshold: {cv_threshold:.4f}  ({cv_time:.1f}s)")

    # --- Fit on full training set ---
    print(f"    Fitting on full training set...")
    t0 = time.time()
    full_pipeline.fit(X_train, y_train)
    fit_time = time.time() - t0
    print(f"    Fit time: {fit_time:.1f}s")

    # --- Predict on test set ---
    y_prob = full_pipeline.predict_proba(X_test)[:, 1]
    all_probas[name] = y_prob

    def eval_at_threshold(threshold: float) -> dict:
        y_pred = (y_prob >= threshold).astype(int)
        cm     = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        return {
            "threshold":   round(threshold, 4),
            "roc_auc":     round(roc_auc_score(y_test, y_prob), 4),
            "pr_auc":      round(average_precision_score(y_test, y_prob), 4),
            "brier_score": round(brier_score_loss(y_test, y_prob), 4),
            "precision":   round(precision_score(y_test, y_pred, zero_division=0), 4),
            "recall":      round(recall_score(y_test, y_pred, zero_division=0), 4),
            "f1":          round(f1_score(y_test, y_pred, zero_division=0), 4),
            "confusion_matrix": cm.tolist(),
            "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        }

    metrics_cv   = eval_at_threshold(cv_threshold)
    metrics_0_50 = eval_at_threshold(0.50)

    print(f"\n    [{display}] CV threshold={cv_threshold:.4f}")
    print(f"      ROC-AUC   : {metrics_cv['roc_auc']}")
    print(f"      PR-AUC    : {metrics_cv['pr_auc']}")
    print(f"      Precision : {metrics_cv['precision']}")
    print(f"      Recall    : {metrics_cv['recall']}")
    print(f"      F1        : {metrics_cv['f1']}")
    print(f"      Brier     : {metrics_cv['brier_score']}")
    print(f"      CM        : {metrics_cv['confusion_matrix']}")
    print(f"\n    [{display}] @0.50 threshold")
    print(f"      Precision : {metrics_0_50['precision']}")
    print(f"      Recall    : {metrics_0_50['recall']}")
    print(f"      F1        : {metrics_0_50['f1']}")

    results[name] = {
        "display_name":   display,
        "cv_threshold":   cv_threshold,
        "at_cv_threshold": metrics_cv,
        "at_0_50":        metrics_0_50,
        "fit_time_s":     round(fit_time, 2),
        "cv_time_s":      round(cv_time, 2),
    }
    trained_pipelines[name] = full_pipeline

    # --- Confusion matrix plot ---
    cm_arr = np.array(metrics_cv["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm_arr, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_title(f"{display}\nConfusion Matrix (threshold={cv_threshold:.2f})\n[EXPERIMENTAL]",
                 fontsize=10)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    tick_labels = ["Non-default (0)", "Default (1)"]
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(tick_labels, fontsize=8)
    ax.set_yticklabels(tick_labels, fontsize=8)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm_arr[i, j]),
                    ha="center", va="center",
                    color="white" if cm_arr[i, j] > cm_arr.max() / 2 else "black",
                    fontsize=13, fontweight="bold")
    plt.tight_layout()
    cm_path = ARTIFACT_DIR / f"confusion_matrix_{name}.png"
    plt.savefig(cm_path, dpi=120)
    plt.close()
    print(f"    Confusion matrix saved: {cm_path.name}")

    # --- Save pipeline ---
    model_path = ARTIFACT_DIR / f"{name}_pipeline.joblib"
    joblib.dump(full_pipeline, model_path, compress=3)
    print(f"    Pipeline saved: {model_path.name}")


# ─────────────────────────────────────────────────────────────
# 7. ROC + PR curves (combined plot)
# ─────────────────────────────────────────────────────────────
hdr("5. ROC & Precision-Recall Curves")

COLORS = {
    "logistic_regression": "#e66101",
    "random_forest":       "#5e3c99",
    "xgboost":             "#1a9641",
}

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Indian Loan Default — ROC & PR Curves  [EXPERIMENTAL]", fontsize=12)

ax_roc, ax_pr = axes

# Baseline reference
ax_roc.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC=0.50)")
ax_pr.axhline(test_prevalence, color="k", linestyle="--", lw=1,
              label=f"No-skill (AP={test_prevalence:.3f})")

for name in classifiers:
    y_prob = all_probas[name]
    display = DISPLAY_NAMES[name]
    color   = COLORS[name]
    roc_auc = results[name]["at_cv_threshold"]["roc_auc"]
    pr_auc  = results[name]["at_cv_threshold"]["pr_auc"]

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    ax_roc.plot(fpr, tpr, color=color, lw=1.8,
                label=f"{display} (AUC={roc_auc:.4f})")

    prec, rec, _ = precision_recall_curve(y_test, y_prob)
    ax_pr.plot(rec, prec, color=color, lw=1.8,
               label=f"{display} (AP={pr_auc:.4f})")

    # Mark CV threshold operating point on ROC
    cv_t = results[name]["cv_threshold"]
    y_pred_cv = (y_prob >= cv_t).astype(int)
    op_recall = recall_score(y_test, y_pred_cv, zero_division=0)
    op_fpr    = float((y_pred_cv[(y_test == 0)] == 1).sum() / (y_test == 0).sum())
    ax_roc.scatter([op_fpr], [op_recall], color=color, s=60, zorder=5)

ax_roc.set_xlabel("False Positive Rate")
ax_roc.set_ylabel("True Positive Rate")
ax_roc.set_title("ROC Curve")
ax_roc.legend(fontsize=8)
ax_roc.grid(alpha=0.3)

ax_pr.set_xlabel("Recall")
ax_pr.set_ylabel("Precision")
ax_pr.set_title("Precision-Recall Curve")
ax_pr.legend(fontsize=8)
ax_pr.grid(alpha=0.3)

plt.tight_layout()
roc_pr_path = ARTIFACT_DIR / "roc_pr_curves.png"
plt.savefig(roc_pr_path, dpi=120)
plt.close()
print(f"ROC & PR curves saved: {roc_pr_path}")


# ─────────────────────────────────────────────────────────────
# 8. Model comparison CSV
# ─────────────────────────────────────────────────────────────
hdr("6. Model Comparison CSV")

rows = []
for name, res in results.items():
    m = res["at_cv_threshold"]
    rows.append({
        "model":          res["display_name"],
        "threshold_used": res["cv_threshold"],
        "roc_auc":        m["roc_auc"],
        "pr_auc":         m["pr_auc"],
        "brier_score":    m["brier_score"],
        "precision":      m["precision"],
        "recall":         m["recall"],
        "f1":             m["f1"],
        "tp":             m["tp"],
        "fp":             m["fp"],
        "tn":             m["tn"],
        "fn":             m["fn"],
        "fit_time_s":     res["fit_time_s"],
        "note":           "EXPERIMENTAL",
    })

comparison_df = pd.DataFrame(rows)
csv_path = ARTIFACT_DIR / "model_comparison.csv"
comparison_df.to_csv(csv_path, index=False)
print(f"model_comparison.csv saved: {csv_path}")
print("\n" + comparison_df.to_string(index=False))


# ─────────────────────────────────────────────────────────────
# 9. Model selection
# ─────────────────────────────────────────────────────────────
hdr("7. Model Selection")

# Primary criterion: PR-AUC (most informative for imbalanced classes)
# Secondary: ROC-AUC, then F1
best_name = max(
    results.keys(),
    key=lambda n: (
        results[n]["at_cv_threshold"]["pr_auc"],
        results[n]["at_cv_threshold"]["roc_auc"],
        results[n]["at_cv_threshold"]["f1"],
    )
)
best_display = DISPLAY_NAMES[best_name]
best_metrics = results[best_name]["at_cv_threshold"]

print(f"\nSelected baseline model: {best_display}")
print(f"  PR-AUC    : {best_metrics['pr_auc']}")
print(f"  ROC-AUC   : {best_metrics['roc_auc']}")
print(f"  F1 score  : {best_metrics['f1']}")
print(f"  Threshold : {results[best_name]['cv_threshold']}")


# ─────────────────────────────────────────────────────────────
# 10. Evaluation report JSON
# ─────────────────────────────────────────────────────────────
hdr("8. Evaluation Report JSON")

report = {
    "experiment_label": "EXPERIMENTAL — dataset provenance unverified",
    "generated_at":     datetime.now(timezone.utc).isoformat(),
    "dataset": {
        "path":            str(DATA_PATH),
        "total_rows":      len(df),
        "train_rows":      len(X_train),
        "test_rows":       len(X_test),
        "random_seed":     RANDOM_SEED,
        "test_size":       TEST_SIZE,
        "split_strategy":  "stratified",
        "test_positives":  test_support,
        "test_prevalence": round(test_prevalence, 4),
    },
    "features": {
        "target":              TARGET,
        "excluded":            ["applicant_id", "default_risk_score"],
        "exclusion_reason":    {
            "applicant_id":      "Identifier — never a feature",
            "default_risk_score": "Unverified provenance, MODERATE_LEAKAGE_SUSPECTED",
        },
        "baseline_features":   BASELINE_FEATURES,
        "numeric_features":    NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "n_features":          len(BASELINE_FEATURES),
    },
    "threshold_selection": {
        "method": "OOF cross-validation on training set (5-fold StratifiedKFold)",
        "criterion": "maximise F1 on OOF probabilities",
        "note": "Test set was never used for threshold selection",
    },
    "models": safe_json(results),
    "selected_model": {
        "name":         best_name,
        "display_name": best_display,
        "selection_criteria": "Highest PR-AUC (primary), ROC-AUC (secondary), F1 (tertiary)",
        "metrics_at_cv_threshold": safe_json(best_metrics),
    },
}

report_path = ARTIFACT_DIR / "evaluation_report.json"
with open(report_path, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)
print(f"Evaluation report saved: {report_path}")


# ─────────────────────────────────────────────────────────────
# 11. Model metadata JSON
# ─────────────────────────────────────────────────────────────
hdr("9. Model Metadata")

metadata = {
    "experiment_label":   "EXPERIMENTAL",
    "generated_at":       datetime.now(timezone.utc).isoformat(),
    "selected_model":     best_name,
    "display_name":       best_display,
    "pipeline_file":      f"{best_name}_pipeline.joblib",
    "target":             TARGET,
    "random_seed":        RANDOM_SEED,
    "test_size":          TEST_SIZE,
    "split_strategy":     "StratifiedShuffleSplit (train_test_split stratify=y)",
    "cv_threshold":       results[best_name]["cv_threshold"],
    "default_threshold":  0.50,
    "feature_names":      BASELINE_FEATURES,
    "numeric_features":   NUMERIC_FEATURES,
    "categorical_features": CATEGORICAL_FEATURES,
    "excluded_features":  ["applicant_id", "default_risk_score"],
    "preprocessing": {
        "numeric":     "SimpleImputer(median) + StandardScaler",
        "categorical": "SimpleImputer(most_frequent) + OneHotEncoder(handle_unknown=ignore)",
    },
    "notes": [
        "Dataset provenance is unverified; distribution suggests synthetic or stratified generation.",
        "default_risk_score excluded due to MODERATE_LEAKAGE_SUSPECTED verdict in Stage 3A audit.",
        "6.12% positive-class prevalence — imbalanced; models use class-weight compensation.",
        "Do not use for real lending decisions.",
    ],
}

meta_path = ARTIFACT_DIR / "model_metadata.json"
with open(meta_path, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)
print(f"Model metadata saved: {meta_path}")


# ─────────────────────────────────────────────────────────────
# 12. Final summary
# ─────────────────────────────────────────────────────────────
hdr("STAGE 3B COMPLETE  [EXPERIMENTAL]")

print(f"\n  Dataset     : {DATA_PATH.name}")
print(f"  Train / Test: {len(X_train):,} / {len(X_test):,} rows")
print(f"  Test pos    : {test_support} ({test_prevalence*100:.2f}%)")
print(f"\n  {'Model':<24} {'Threshold':>9} {'ROC-AUC':>8} {'PR-AUC':>8} {'F1':>7} {'Recall':>7} {'Precision':>9}")
print(f"  {'-'*24} {'-'*9} {'-'*8} {'-'*8} {'-'*7} {'-'*7} {'-'*9}")
for name, res in results.items():
    m = res["at_cv_threshold"]
    marker = " <-- SELECTED" if name == best_name else ""
    print(f"  {res['display_name']:<24} {res['cv_threshold']:>9.4f} "
          f"{m['roc_auc']:>8.4f} {m['pr_auc']:>8.4f} "
          f"{m['f1']:>7.4f} {m['recall']:>7.4f} {m['precision']:>9.4f}{marker}")

print(f"\n  Selected: {best_display}")
print(f"  Criterion: PR-AUC (primary), ROC-AUC (secondary), F1 (tertiary)")
print(f"\n  Artifacts saved to: {ARTIFACT_DIR}")
print(f"\n  ** All results are EXPERIMENTAL **")
print(f"  ** Do not integrate into API yet (Stage 3C) **\n")
