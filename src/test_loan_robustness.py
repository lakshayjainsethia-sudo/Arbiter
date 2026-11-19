
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = Path("data/raw/lendingclub/LC_loans_granting_model_dataset.csv")
MODEL_PATH = Path("artifacts/ablation/loan_xgboost_without_zip.joblib")
OUTPUT_DIR = Path("artifacts/robustness")

TARGET = "Default"
MAX_ROWS = 50000
TEST_SIZE = 0.20
RANDOM_STATE = 42
THRESHOLD = 0.50

# Test one feature at a time, with controlled relative changes.
# These are sensitivity scenarios, not claims of realistic attacks.
PERTURBATION_LEVELS = [0.05, 0.10]


# ============================================================
# LOAD MODEL AND TEST DATA
# ============================================================

def load_test_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    print("Loading saved ZIP-code-free XGBoost model...")
    model = joblib.load(MODEL_PATH)

    print("Loading dataset...")
    df = pd.read_csv(
        DATA_PATH,
        nrows=MAX_ROWS,
        low_memory=False,
        dtype={
            "zip_code": "category",
            "addr_state": "category",
        },
    )

    df = df.dropna(subset=[TARGET]).copy()
    df[TARGET] = pd.to_numeric(df[TARGET], errors="raise")

    if not set(df[TARGET].unique()).issubset({0, 1}):
        raise ValueError("Default must contain only 0 and 1.")

    y = df[TARGET].astype("int8")

    X = df.drop(
        columns=["id", "issue_d", "title", "desc", TARGET],
        errors="ignore",
    )

    X = X.drop(columns=["zip_code"], errors="ignore")

    # Reproduce the original train/test split.
    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("Test rows:", len(X_test))
    return model, X_test, y_test


# ============================================================
# EVALUATION
# ============================================================

def calculate_metrics(name, y_true, probabilities):
    predictions = (probabilities >= THRESHOLD).astype(int)

    return {
        "scenario": name,
        "roc_auc": roc_auc_score(y_true, probabilities),
        "pr_auc": average_precision_score(y_true, probabilities),
        "default_precision": precision_score(
            y_true, predictions, zero_division=0
        ),
        "default_recall": recall_score(
            y_true, predictions, zero_division=0
        ),
        "default_f1": f1_score(
            y_true, predictions, zero_division=0
        ),
        "positive_predictions": int(predictions.sum()),
    }


# ============================================================
# CONTROLLED NUMERICAL SENSITIVITY TEST
# ============================================================

def perturb_feature(X, feature, relative_change):
    """
    Return a copy with one numeric feature shifted by a relative amount.

    Values are clipped to the observed training-sample test range.
    This tests model sensitivity, not necessarily a feasible real-world
    change to a borrower's financial circumstances.
    """
    modified = X.copy()

    original = pd.to_numeric(modified[feature], errors="coerce")
    valid = original.dropna()

    if valid.empty:
        return None

    lower = valid.min()
    upper = valid.max()

    # Avoid dividing by zero when values can be zero.
    scale = max(abs(float(valid.median())), 1.0)
    delta = scale * relative_change

    modified[feature] = (original + delta).clip(
        lower=lower,
        upper=upper,
    )

    return modified


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 65)
    print("ARBITER - LOAN ROBUSTNESS SENSITIVITY TEST")
    print("=" * 65)

    model, X_test, y_test = load_test_data()

    print("\nComputing clean baseline predictions...")
    clean_probabilities = model.predict_proba(X_test)[:, 1]
    clean_predictions = (clean_probabilities >= THRESHOLD).astype(int)

    results = [
        calculate_metrics(
            "clean_baseline",
            y_test,
            clean_probabilities,
        )
    ]

    # These are numerical features available to this model.
    candidate_features = [
        feature
        for feature in ["fico_n", "dti_n", "revenue", "loan_amnt", "experience_c"]
        if feature in X_test.columns
        and pd.api.types.is_numeric_dtype(X_test[feature])
    ]

    if not candidate_features:
        raise ValueError("No candidate numerical features were found.")

    # A deterministic subset keeps the first experiment lightweight.
    X_sample = X_test.iloc[:2000].copy()
    clean_sample_probabilities = model.predict_proba(X_sample)[:, 1]
    clean_sample_predictions = (
        clean_sample_probabilities >= THRESHOLD
    ).astype(int)

    print("\nCandidate features:", candidate_features)
    print("Rows used for perturbation tests:", len(X_sample))

    perturbation_results = []

    for feature in candidate_features:
        for level in PERTURBATION_LEVELS:
            X_modified = perturb_feature(X_sample, feature, level)

            if X_modified is None:
                continue

            modified_probabilities = model.predict_proba(X_modified)[:, 1]
            modified_predictions = (
                modified_probabilities >= THRESHOLD
            ).astype(int)

            # Count how often the thresholded prediction changes.
            changed = modified_predictions != clean_sample_predictions

            # Count cleanly detected defaults that become missed defaults
            # under the perturbation.
            sample_y = y_test.iloc[:len(X_sample)].to_numpy()
            newly_missed = (
                (sample_y == 1)
                & (clean_sample_predictions == 1)
                & (modified_predictions == 0)
            )

            perturbation_results.append({
                "feature": feature,
                "relative_shift": level,
                "mean_absolute_probability_change": float(
                    np.mean(
                        np.abs(
                            modified_probabilities
                            - clean_sample_probabilities
                        )
                    )
                ),
                "prediction_change_rate": float(np.mean(changed)),
                "newly_missed_defaults": int(newly_missed.sum()),
                "sample_rows": len(X_sample),
            })

            print(
                f"{feature}, shift={level:.0%}: "
                f"mean probability change="
                f"{perturbation_results[-1]['mean_absolute_probability_change']:.4f}, "
                f"prediction changes="
                f"{perturbation_results[-1]['prediction_change_rate']:.2%}"
            )

    # Save full-test clean metrics.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    metrics_path = OUTPUT_DIR / "clean_baseline_metrics.csv"
    pd.DataFrame(results).to_csv(metrics_path, index=False)

    perturbation_path = OUTPUT_DIR / "perturbation_results.csv"
    pd.DataFrame(perturbation_results).to_csv(
        perturbation_path,
        index=False,
    )

    print("\nClean baseline metrics:")
    print(pd.DataFrame(results).to_string(index=False))

    print("\nSaved clean metrics:", metrics_path.resolve())
    print("Saved perturbation results:", perturbation_path.resolve())
    print("\nSensitivity testing completed.")
    print(
        "These perturbations are controlled experiments. "
        "They are not yet a validated adversarial attack or defense."
    )


if __name__ == "__main__":
    main()
