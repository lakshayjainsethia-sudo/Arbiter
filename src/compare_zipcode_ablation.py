
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = Path("data/raw/lendingclub/LC_loans_granting_model_dataset.csv")
BASELINE_PATH = Path("artifacts/loan_xgboost.joblib")
OUTPUT_DIR = Path("artifacts/ablation")

TARGET = "Default"
MAX_ROWS = 50000
RANDOM_STATE = 42
TEST_SIZE = 0.20
THRESHOLD = 0.50


# ============================================================
# LOAD DATA AND CREATE THE SAME SPLIT
# ============================================================

def load_data():
    print("=" * 65)
    print("ARBITER - ZIP-CODE ABLATION EXPERIMENT")
    print("=" * 65)

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    if not BASELINE_PATH.exists():
        raise FileNotFoundError(f"Baseline model not found: {BASELINE_PATH}")

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

    empty_columns = X.columns[X.isna().all()].tolist()
    X = X.drop(columns=empty_columns)

    # The same split settings as the original XGBoost script.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("Total rows:", len(X))
    print("Training rows:", len(X_train))
    print("Testing rows:", len(X_test))

    return X_train, X_test, y_train, y_test


# ============================================================
# BUILD A MODEL PIPELINE
# ============================================================

def build_model(X_train, y_train):
    categorical_columns = X_train.select_dtypes(
        include=["object", "category", "bool", "string"]
    ).columns.tolist()

    numeric_columns = X_train.select_dtypes(
        include=["number"]
    ).columns.tolist()

    for column in ["zip_code", "addr_state"]:
        if column in X_train.columns and column not in categorical_columns:
            categorical_columns.append(column)
            if column in numeric_columns:
                numeric_columns.remove(column)

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore",
            min_frequency=20,
            sparse_output=True,
            dtype="float32",
        )),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
        sparse_threshold=1.0,
    )

    negatives = int((y_train == 0).sum())
    positives = int((y_train == 1).sum())

    classifier = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        scale_pos_weight=negatives / positives,
        random_state=RANDOM_STATE,
        n_jobs=2,
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])


# ============================================================
# EVALUATE MODEL
# ============================================================

def evaluate(name, model, X_test, y_test):
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= THRESHOLD).astype(int)

    result = {
        "model": name,
        "roc_auc": roc_auc_score(y_test, probabilities),
        "pr_auc": average_precision_score(y_test, probabilities),
        "default_precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "default_recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "default_f1": f1_score(
            y_test, predictions, zero_division=0
        ),
    }

    print(f"\n{name}")
    print("-" * 45)

    for metric, value in result.items():
        if metric != "model":
            print(f"{metric}: {value:.4f}")

    return result


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def main():
    X_train, X_test, y_train, y_test = load_data()

    # Evaluate the existing model on the same test rows.
    print("\nLoading existing baseline model...")
    baseline = joblib.load(BASELINE_PATH)

    results = [
        evaluate(
            "XGBoost with ZIP code",
            baseline,
            X_test,
            y_test,
        )
    ]

    # Remove ZIP code from both training and testing features.
    if "zip_code" not in X_train.columns:
        print("\nZIP-code column is absent; nothing to ablate.")
        return

    X_train_without_zip = X_train.drop(columns=["zip_code"])
    X_test_without_zip = X_test.drop(columns=["zip_code"])

    print("\nTraining a new XGBoost model without ZIP code...")
    start = time.time()

    ablated_model = build_model(
        X_train_without_zip,
        y_train,
    )

    ablated_model.fit(X_train_without_zip, y_train)

    training_seconds = time.time() - start
    print(f"Training time: {training_seconds:.2f} seconds")

    results.append(
        evaluate(
            "XGBoost without ZIP code",
            ablated_model,
            X_test_without_zip,
            y_test,
        )
    )

    # Save the ablated model and comparison.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model_path = OUTPUT_DIR / "loan_xgboost_without_zip.joblib"
    joblib.dump(ablated_model, model_path, compress=3)

    results_df = pd.DataFrame(results)
    results_path = OUTPUT_DIR / "zipcode_ablation_comparison.csv"
    results_df.to_csv(results_path, index=False)

    print("\n" + "=" * 65)
    print("ABLATION COMPARISON")
    print("=" * 65)
    print(results_df.to_string(index=False))

    print("\nSaved ablated model:", model_path.resolve())
    print("Saved comparison:", results_path.resolve())

    print(
        "\nInterpretation: compare the metrics across both models. "
        "A performance drop after removing ZIP code suggests it "
        "contributed predictive information, but does not establish "
        "that ZIP code is causal or appropriate to use."
    )


if __name__ == "__main__":
    main()
