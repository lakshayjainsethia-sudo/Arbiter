
from pathlib import Path
import time

import joblib
import pandas as pd
import xgboost as xgb

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ============================================================
# 1. CONFIGURATION
# ============================================================

DATA_PATH = Path("data/raw/lendingclub/LC_loans_granting_model_dataset.csv")
MODEL_PATH = Path("artifacts/loan_xgboost.joblib")

TARGET = "Default"
MAX_ROWS = 50000
TEST_SIZE = 0.20
RANDOM_STATE = 42


# ============================================================
# 2. LOAD DATA
# ============================================================

def load_data():
    print("=" * 60)
    print("ARBITER - XGBOOST LOAN DEFAULT PREDICTION")
    print("=" * 60)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH.resolve()}"
        )

    print("\nLoading dataset...")

    options = {
        "low_memory": False,
        "dtype": {
            "zip_code": "category",
            "addr_state": "category",
        },
    }

    if MAX_ROWS is not None:
        options["nrows"] = MAX_ROWS

    df = pd.read_csv(DATA_PATH, **options)

    print("Dataset shape:", df.shape)

    if TARGET not in df.columns:
        raise ValueError(f"Missing target column: {TARGET}")

    return df


# ============================================================
# 3. PREPARE FEATURES
# ============================================================

def prepare_data(df):
    print("\nPreparing features...")

    df = df.dropna(subset=[TARGET]).copy()
    df[TARGET] = pd.to_numeric(df[TARGET], errors="raise")

    if not set(df[TARGET].unique()).issubset({0, 1}):
        raise ValueError("Default must contain only 0 and 1.")

    y = df[TARGET].astype("int8")

    # Exclude identifiers, free text, date and target.
    drop_columns = [
        "id",
        "issue_d",
        "title",
        "desc",
        TARGET,
    ]

    X = df.drop(columns=drop_columns, errors="ignore")

    empty_columns = X.columns[X.isna().all()].tolist()
    if empty_columns:
        X = X.drop(columns=empty_columns)

    categorical_columns = X.select_dtypes(
        include=["object", "category", "bool", "string"]
    ).columns.tolist()

    numeric_columns = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    # ZIP codes are categories, not numeric measurements.
    for column in ["zip_code", "addr_state"]:
        if column in X.columns and column not in categorical_columns:
            categorical_columns.append(column)
            if column in numeric_columns:
                numeric_columns.remove(column)

    print("Rows:", len(X))
    print("Numerical features:", len(numeric_columns))
    print("Categorical features:", len(categorical_columns))
    print("\nTarget distribution:")
    print(y.value_counts().sort_index())

    return X, y, numeric_columns, categorical_columns


# ============================================================
# 4. PREPROCESSING
# ============================================================

def build_preprocessor(numeric_columns, categorical_columns):
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

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
        sparse_threshold=1.0,
    )


# ============================================================
# 5. TRAIN XGBOOST
# ============================================================

def train_model(X, y, numeric_columns, categorical_columns):
    print("\nSplitting data...")

    # Same split settings as the Random Forest baseline.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("Training rows:", len(X_train))
    print("Testing rows:", len(X_test))

    preprocessor = build_preprocessor(
        numeric_columns,
        categorical_columns,
    )

    # Compensate for class imbalance using training data only.
    negative_count = int((y_train == 0).sum())
    positive_count = int((y_train == 1).sum())
    class_weight_ratio = negative_count / positive_count

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
        scale_pos_weight=class_weight_ratio,
        random_state=RANDOM_STATE,
        n_jobs=2,
    )

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])

    print("\nTraining XGBoost...")
    start_time = time.time()

    model.fit(X_train, y_train)

    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds.")

    return model, X_test, y_test, training_time


# ============================================================
# 6. EVALUATION
# ============================================================

def evaluate_model(model, X_test, y_test):
    print("\n" + "=" * 60)
    print("XGBOOST EVALUATION")
    print("=" * 60)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        predictions,
        target_names=["Non-default (0)", "Default (1)"],
        zero_division=0,
    ))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    roc_auc = roc_auc_score(y_test, probabilities)
    pr_auc = average_precision_score(y_test, probabilities)

    print(f"\nROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC (Average Precision): {pr_auc:.4f}")

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
    }


# ============================================================
# 7. SAVE MODEL AND METRICS
# ============================================================

def main():
    df = load_data()

    X, y, numeric_columns, categorical_columns = prepare_data(df)

    model, X_test, y_test, training_time = train_model(
        X,
        y,
        numeric_columns,
        categorical_columns,
    )

    metrics = evaluate_model(model, X_test, y_test)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH, compress=3)

    print("\nModel saved to:", MODEL_PATH.resolve())

    metrics_path = Path("artifacts/loan_xgboost_metrics.csv")
    pd.DataFrame([{
        "model": "XGBoost",
        "rows": len(df),
        "roc_auc": metrics["roc_auc"],
        "pr_auc": metrics["pr_auc"],
        "training_time_seconds": training_time,
    }]).to_csv(metrics_path, index=False)

    print("Metrics saved to:", metrics_path.resolve())
    print("\nXGBoost training completed successfully.")


if __name__ == "__main__":
    main()
