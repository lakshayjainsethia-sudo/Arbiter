
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
OUTPUT_DIR = Path("artifacts/robustness/final_evaluation")

TARGET = "Default"
MAX_ROWS = 50000
TEST_SIZE = 0.20
RANDOM_STATE = 42

BASE_THRESHOLD = 0.50
REVIEW_LOW = 0.45
REVIEW_HIGH = 0.55

# Match the previous stability experiment.
SAMPLE_SIZE = 2000

PERTURBATIONS = {
    "fico_n": 5.0,
    "dti_n": 0.5,
    "revenue": 0.05,
    "loan_amnt": 0.05,
}


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    print("=" * 70)
    print("ARBITER - FINAL ROBUSTNESS EVALUATION")
    print("=" * 70)

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    model = joblib.load(MODEL_PATH)

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
        columns=["id", "issue_d", "title", "desc", TARGET, "zip_code"],
        errors="ignore",
    )

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    X_test = X_test.iloc[:SAMPLE_SIZE].copy()
    y_test = y_test.iloc[:SAMPLE_SIZE].copy()

    print("Test rows:", len(X_test))
    print("Default cases:", int((y_test == 1).sum()))
    print("Model:", MODEL_PATH)

    return model, X_test, y_test


# ============================================================
# POLICY DEFINITIONS
# ============================================================

def get_class(probabilities):
    return (probabilities >= BASE_THRESHOLD).astype(int)


def get_review_flags(probabilities):
    return (
        (probabilities >= REVIEW_LOW)
        & (probabilities <= REVIEW_HIGH)
    )


# ============================================================
# PERTURBATION HELPERS
# ============================================================

def get_delta(X, feature, amount):
    if feature in ("fico_n", "dti_n"):
        return float(amount)

    values = pd.to_numeric(X[feature], errors="coerce")
    return max(abs(float(values.median())), 1.0) * amount


def make_variant(X, feature, delta):
    modified = X.copy()

    values = pd.to_numeric(modified[feature], errors="coerce")
    valid = values.dropna()

    if valid.empty:
        return None

    modified[feature] = (values + delta).clip(
        lower=valid.min(),
        upper=valid.max(),
    )

    return modified


# ============================================================
# EVALUATE ONE SCENARIO
# ============================================================

def evaluate_scenario(
    scenario,
    y_true,
    probabilities,
    clean_predictions,
):
    actual = np.asarray(y_true)
    predictions = get_class(probabilities)
    review = get_review_flags(probabilities)
    automatic = ~review

    actual_default = actual == 1
    predicted_default = predictions == 1

    # Defaults automatically predicted as non-default.
    # Reviewed cases are reported separately, not counted as
    # automatic decisions.
    automatic_false_negatives = (
        actual_default & automatic & ~predicted_default
    )

    reviewed_defaults = actual_default & review

    return {
        "scenario": scenario,
        "roc_auc": roc_auc_score(actual, probabilities),
        "pr_auc": average_precision_score(actual, probabilities),
        "precision": precision_score(
            actual, predictions, zero_division=0
        ),
        "recall": recall_score(
            actual, predictions, zero_division=0
        ),
        "f1": f1_score(
            actual, predictions, zero_division=0
        ),
        "original_false_negatives": int(
            (actual_default & ~predicted_default).sum()
        ),
        "review_count": int(review.sum()),
        "review_rate": float(review.mean()),
        "reviewed_defaults": int(reviewed_defaults.sum()),
        "reviewed_nondefaults": int(
            ((actual == 0) & review).sum()
        ),
        "automatic_false_negatives": int(
            automatic_false_negatives.sum()
        ),
        "automatic_decision_rate": float(automatic.mean()),
        "class_flip_rate_vs_clean": float(
            (predictions != clean_predictions).mean()
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main():
    model, X, y = load_data()

    print("\nCalculating clean predictions...")
    clean_probabilities = model.predict_proba(X)[:, 1]
    clean_predictions = get_class(clean_probabilities)

    results = [
        evaluate_scenario(
            "clean",
            y,
            clean_probabilities,
            clean_predictions,
        )
    ]

    for feature, amount in PERTURBATIONS.items():
        if feature not in X.columns:
            print(f"Skipping unavailable feature: {feature}")
            continue

        if not pd.api.types.is_numeric_dtype(X[feature]):
            print(f"Skipping non-numeric feature: {feature}")
            continue

        delta = get_delta(X, feature, amount)

        for direction in (-1, 1):
            variant = make_variant(
                X,
                feature,
                direction * delta,
            )

            if variant is None:
                continue

            probabilities = model.predict_proba(variant)[:, 1]

            scenario = (
                f"{feature}_"
                f"{'decrease' if direction < 0 else 'increase'}"
            )

            results.append(
                evaluate_scenario(
                    scenario,
                    y,
                    probabilities,
                    clean_predictions,
                )
            )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame(results)
    output_path = OUTPUT_DIR / "final_robustness_comparison.csv"
    results_df.to_csv(output_path, index=False)

    print("\n" + "=" * 70)
    print("FINAL ROBUSTNESS COMPARISON")
    print("=" * 70)

    display_columns = [
        "scenario",
        "roc_auc",
        "pr_auc",
        "recall",
        "review_rate",
        "reviewed_defaults",
        "automatic_false_negatives",
        "class_flip_rate_vs_clean",
    ]

    print(
        results_df[display_columns]
        .round(4)
        .to_string(index=False)
    )

    clean = results_df.iloc[0]

    print("\nCLEAN BASELINE SUMMARY")
    print(f"ROC-AUC: {clean['roc_auc']:.4f}")
    print(f"PR-AUC: {clean['pr_auc']:.4f}")
    print(f"Defaults reviewed: {int(clean['reviewed_defaults'])}")
    print(
        "Defaults automatically predicted as non-default:",
        int(clean["automatic_false_negatives"]),
    )
    print(f"Review rate: {clean['review_rate']:.2%}")

    print("\nSaved report:", output_path.resolve())
    print(
        "\nNote: review counts and automatic false negatives describe "
        "the chosen review policy. They do not prove a defense against "
        "all adversarial attacks."
    )


if __name__ == "__main__":
    main()
