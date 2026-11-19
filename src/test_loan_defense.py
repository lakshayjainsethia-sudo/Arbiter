
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split


# ============================================================
# 1. CONFIGURATION
# ============================================================

DATA_PATH = Path("data/raw/lendingclub/LC_loans_granting_model_dataset.csv")
MODEL_PATH = Path("artifacts/ablation/loan_xgboost_without_zip.joblib")
OUTPUT_DIR = Path("artifacts/robustness")

TARGET = "Default"
MAX_ROWS = 50000
TEST_SIZE = 0.20
RANDOM_STATE = 42
THRESHOLD = 0.50

# Small, controlled changes for this initial experiment.
# These are test scenarios, not claims about real-world attacks.
PERTURBATIONS = {
    "fico_n": 5.0,        # +/- 5 FICO points
    "dti_n": 0.5,          # +/- 0.5 DTI units
    "revenue": 0.05,       # +/- 5%
    "loan_amnt": 0.05,     # +/- 5%
}


# ============================================================
# 2. LOAD MODEL AND DATA
# ============================================================

def load_data():
    print("=" * 65)
    print("ARBITER - PREDICTION STABILITY DEFENSE")
    print("=" * 65)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

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

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Match the 2,000-row sample used in the initial sensitivity test.
    X_sample = X_test.iloc[:2000].copy()
    y_sample = y_test.iloc[:2000].copy()

    print("Test sample rows:", len(X_sample))
    print("Model loaded:", MODEL_PATH)

    return model, X_sample, y_sample


# ============================================================
# 3. CREATE BOUNDED VARIATIONS
# ============================================================

def make_perturbed_copy(X, feature, delta):
    """
    Perturb one numeric feature while keeping it inside the
    observed sample's minimum and maximum.

    This is a controlled sensitivity test. The resulting row
    may not represent a feasible real-world financial situation.
    """
    modified = X.copy()

    original = pd.to_numeric(modified[feature], errors="coerce")
    valid = original.dropna()

    if valid.empty:
        return None

    lower = valid.min()
    upper = valid.max()

    modified[feature] = (original + delta).clip(
        lower=lower,
        upper=upper,
    )

    return modified


def get_delta(series, amount):
    """
    Absolute changes are used for FICO and DTI.
    Relative changes are used for revenue and loan amount.
    """
    if series.name in ("fico_n", "dti_n"):
        return float(amount)

    numeric = pd.to_numeric(series, errors="coerce")
    median_value = numeric.median()

    scale = max(abs(float(median_value)), 1.0)
    return scale * float(amount)


# ============================================================
# 4. RUN STABILITY TEST
# ============================================================

def run_stability_test(model, X, y):
    print("\nGenerating clean predictions...")

    clean_prob = model.predict_proba(X)[:, 1]
    clean_pred = (clean_prob >= THRESHOLD).astype(int)

    # A prediction is flagged if any tested variation changes
    # the predicted class. This is a review signal, not a new
    # risk classification.
    any_flip = np.zeros(len(X), dtype=bool)
    max_probability_change = np.zeros(len(X), dtype=float)

    scenario_results = []

    for feature, amount in PERTURBATIONS.items():
        if feature not in X.columns:
            print(f"Skipping unavailable feature: {feature}")
            continue

        if not pd.api.types.is_numeric_dtype(X[feature]):
            print(f"Skipping non-numeric feature: {feature}")
            continue

        base_delta = get_delta(X[feature], amount)

        for direction in (-1, 1):
            signed_delta = direction * base_delta

            X_variant = make_perturbed_copy(
                X,
                feature,
                signed_delta,
            )

            if X_variant is None:
                continue

            variant_prob = model.predict_proba(X_variant)[:, 1]
            variant_pred = (variant_prob >= THRESHOLD).astype(int)

            flipped = variant_pred != clean_pred
            probability_change = np.abs(variant_prob - clean_prob)

            any_flip |= flipped
            max_probability_change = np.maximum(
                max_probability_change,
                probability_change,
            )

            scenario_results.append({
                "feature": feature,
                "direction": "increase" if direction > 0 else "decrease",
                "requested_change": abs(signed_delta),
                "prediction_change_rate": float(flipped.mean()),
                "mean_absolute_probability_change": float(
                    probability_change.mean()
                ),
                "changed_to_default": int(
                    ((clean_pred == 0) & (variant_pred == 1)).sum()
                ),
                "changed_to_non_default": int(
                    ((clean_pred == 1) & (variant_pred == 0)).sum()
                ),
            })

            print(
                f"{feature} ({'+' if direction > 0 else '-'}): "
                f"prediction changes={flipped.mean():.2%}, "
                f"mean probability change={probability_change.mean():.4f}"
            )

    results = pd.DataFrame({
        "actual_default": y.to_numpy(),
        "clean_probability": clean_prob,
        "clean_prediction": clean_pred,
        "stability_flag": any_flip,
        "max_probability_change": max_probability_change,
    })

    results["review_recommended"] = results["stability_flag"]

    return results, scenario_results


# ============================================================
# 5. SUMMARIZE RESULTS
# ============================================================

def summarize(results, y):
    clean_pred = results["clean_prediction"].to_numpy()
    clean_prob = results["clean_probability"].to_numpy()
    actual = y.to_numpy()

    flagged = results["stability_flag"].to_numpy()
    stable = ~flagged

    print("\n" + "=" * 65)
    print("STABILITY DEFENSE RESULTS")
    print("=" * 65)

    print(f"Rows tested: {len(results)}")
    print(f"Flagged for review: {flagged.sum()}")
    print(f"Flag rate: {flagged.mean():.2%}")
    print(f"Stable predictions: {stable.sum()}")
    print(f"Stable rate: {stable.mean():.2%}")

    print("\nClean model metrics:")
    print(f"ROC-AUC: {roc_auc_score(actual, clean_prob):.4f}")
    print(f"PR-AUC: {average_precision_score(actual, clean_prob):.4f}")
    print(
        "Default recall:",
        f"{recall_score(actual, clean_pred, zero_division=0):.4f}",
    )

    print("\nFlagged cases by actual outcome:")
    print(
        results.groupby("actual_default")["stability_flag"]
        .agg(["count", "sum", "mean"])
        .rename(columns={
            "count": "rows",
            "sum": "flagged_rows",
            "mean": "flag_rate",
        })
        .to_string()
    )

    print(
        "\nInterpretation: a stability flag means the model's "
        "thresholded prediction changed under at least one tested "
        "variation. It does not prove the original prediction is "
        "wrong or that the variation is a feasible attack."
    )


# ============================================================
# 6. SAVE OUTPUTS
# ============================================================

def main():
    model, X_sample, y_sample = load_data()

    results, scenario_results = run_stability_test(
        model,
        X_sample,
        y_sample,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    row_path = OUTPUT_DIR / "stability_defense_rows.csv"
    results.to_csv(row_path, index=False)

    scenario_path = OUTPUT_DIR / "stability_defense_summary.csv"
    pd.DataFrame(scenario_results).to_csv(
        scenario_path,
        index=False,
    )

    summarize(results, y_sample)

    print("\nSaved row-level results:", row_path.resolve())
    print("Saved scenario summary:", scenario_path.resolve())
    print("\nStability-monitoring experiment completed.")


if __name__ == "__main__":
    main()
