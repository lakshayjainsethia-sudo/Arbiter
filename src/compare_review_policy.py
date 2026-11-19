
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

# Existing classification threshold
BASE_THRESHOLD = 0.50

# Experimental review band
REVIEW_LOW = 0.45
REVIEW_HIGH = 0.55

# Controlled variations used for stability testing
PERTURBATIONS = {
    "fico_n": 5.0,       # +/- 5 points
    "dti_n": 0.5,        # +/- 0.5 units
    "revenue": 0.05,     # +/- 5% of sample median
    "loan_amnt": 0.05,   # +/- 5% of sample median
}


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    print("=" * 65)
    print("ARBITER - REVIEW POLICY COMPARISON")
    print("=" * 65)

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

    # Use the same first 2,000 test rows as the stability experiment.
    X_sample = X_test.iloc[:2000].copy()
    y_sample = y_test.iloc[:2000].copy()

    print("Test sample:", len(X_sample))
    print("Model:", MODEL_PATH)

    return model, X_sample, y_sample


# ============================================================
# PREDICTION POLICIES
# ============================================================

def classify(probabilities):
    return (probabilities >= BASE_THRESHOLD).astype(int)


def review_flags(probabilities):
    # The review policy abstains from an automatic decision
    # when probability falls inside the experimental review band.
    return (
        (probabilities >= REVIEW_LOW)
        & (probabilities <= REVIEW_HIGH)
    )


def summarize_policy(name, y_true, probabilities):
    predictions = classify(probabilities)
    review = review_flags(probabilities)
    automatic = ~review

    y_array = np.asarray(y_true)
    default_class = 1

    result = {
        "scenario": name,
        "roc_auc": roc_auc_score(y_array, probabilities),
        "pr_auc": average_precision_score(y_array, probabilities),
        "base_precision": precision_score(
            y_array, predictions, zero_division=0
        ),
        "base_recall": recall_score(
            y_array, predictions, zero_division=0
        ),
        "base_f1": f1_score(
            y_array, predictions, zero_division=0
        ),
        "review_count": int(review.sum()),
        "review_rate": float(review.mean()),
        "automatic_decision_rate": float(automatic.mean()),
        "defaults_in_review": int(
            ((y_array == default_class) & review).sum()
        ),
        "defaults_auto_predicted_nondefault": int(
            (
                (y_array == default_class)
                & automatic
                & (predictions == 0)
            ).sum()
        ),
        "nondefaults_in_review": int(
            ((y_array == 0) & review).sum()
        ),
    }

    return result


# ============================================================
# CREATE CONTROLLED PERTURBATIONS
# ============================================================

def make_variant(X, feature, delta):
    modified = X.copy()

    original = pd.to_numeric(modified[feature], errors="coerce")
    valid = original.dropna()

    if valid.empty:
        return None

    modified[feature] = (original + delta).clip(
        lower=valid.min(),
        upper=valid.max(),
    )

    return modified


def perturbation_delta(X, feature, amount):
    if feature in ("fico_n", "dti_n"):
        return float(amount)

    numeric = pd.to_numeric(X[feature], errors="coerce")
    median = numeric.median()

    return max(abs(float(median)), 1.0) * float(amount)


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def main():
    model, X, y = load_data()

    print("\nGenerating clean probabilities...")
    clean_prob = model.predict_proba(X)[:, 1]

    rows = []
    clean_result = summarize_policy("clean", y, clean_prob)
    rows.append(clean_result)

    clean_class = classify(clean_prob)
    clean_review = review_flags(clean_prob)

    perturbation_rows = []

    for feature, amount in PERTURBATIONS.items():
        if feature not in X.columns:
            print(f"Skipping unavailable feature: {feature}")
            continue

        if not pd.api.types.is_numeric_dtype(X[feature]):
            print(f"Skipping non-numeric feature: {feature}")
            continue

        delta = perturbation_delta(X, feature, amount)

        for direction in (-1, 1):
            variant = make_variant(
                X,
                feature,
                direction * delta,
            )

            if variant is None:
                continue

            probability = model.predict_proba(variant)[:, 1]
            prediction = classify(probability)
            review = review_flags(probability)

            changed_class = prediction != clean_class
            changed_review_status = review != clean_review

            perturbation_rows.append({
                "feature": feature,
                "direction": "increase" if direction > 0 else "decrease",
                "requested_change": abs(delta),
                "mean_absolute_probability_change": float(
                    np.mean(np.abs(probability - clean_prob))
                ),
                "class_change_rate_vs_clean": float(
                    changed_class.mean()
                ),
                "review_status_change_rate_vs_clean": float(
                    changed_review_status.mean()
                ),
                "perturbed_review_rate": float(review.mean()),
                "rows": len(X),
            })

            rows.append(
                summarize_policy(
                    f"{feature}_{'plus' if direction > 0 else 'minus'}",
                    y,
                    probability,
                )
            )

    # Save detailed outputs
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    policy_path = OUTPUT_DIR / "review_policy_metrics.csv"
    pd.DataFrame(rows).to_csv(policy_path, index=False)

    perturbation_path = OUTPUT_DIR / "review_policy_perturbations.csv"
    pd.DataFrame(perturbation_rows).to_csv(
        perturbation_path,
        index=False,
    )

    row_path = OUTPUT_DIR / "review_policy_rows.csv"

    pd.DataFrame({
        "actual_default": np.asarray(y),
        "clean_probability": clean_prob,
        "clean_prediction": clean_class,
        "clean_review_flag": clean_review,
    }).to_csv(row_path, index=False)

    print("\n" + "=" * 65)
    print("CLEAN POLICY RESULTS")
    print("=" * 65)
    print(pd.DataFrame([clean_result]).to_string(index=False))

    print("\n" + "=" * 65)
    print("PERTURBATION COMPARISON")
    print("=" * 65)
    print(pd.DataFrame(perturbation_rows).to_string(index=False))

    print("\nSaved policy metrics:", policy_path.resolve())
    print("Saved perturbation comparison:", perturbation_path.resolve())
    print("Saved row-level decisions:", row_path.resolve())

    print(
        "\nThe review band is experimental. Review flags indicate "
        "uncertainty under the chosen threshold policy; they do not "
        "prove adversarial activity or improve the underlying model "
        "automatically."
    )


if __name__ == "__main__":
    main()
