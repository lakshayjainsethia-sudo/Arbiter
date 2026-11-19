
from pathlib import Path
import time

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
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
MODEL_PATH = Path("artifacts/loan_random_forest.joblib")

TARGET = "Default"
TEST_SIZE = 0.20
RANDOM_STATE = 42

# Set to 50000 for a smaller initial test run.
# Set to None to use the complete dataset.
MAX_ROWS = 50000


# ============================================================
# 2. LOAD DATA
# ============================================================

def load_data():
    print("=" * 60)
    print("ARBITER - LOAN DEFAULT PREDICTION")
    print("=" * 60)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH.resolve()}\n"
            "Check that the CSV exists at the specified path."
        )

    print("\nLoading dataset...")

    read_options = {
        "low_memory": False,
        "dtype": {
            "zip_code": "category",
            "addr_state": "category",
        },
    }

    if MAX_ROWS is not None:
        print(f"Initial run: loading up to {MAX_ROWS:,} rows.")
        read_options["nrows"] = MAX_ROWS
    else:
        print("Loading the complete dataset. This may require substantial RAM.")

    df = pd.read_csv(DATA_PATH, **read_options)

    print(f"Dataset shape: {df.shape}")
    print("\nFirst five rows:")
    print(df.head())

    if TARGET not in df.columns:
        raise ValueError(
            f"Target column '{TARGET}' was not found. "
            f"Available columns: {df.columns.tolist()}"
        )

    return df


# ============================================================
# 3. PREPARE FEATURES AND TARGET
# ============================================================

def prepare_data(df):
    print("\nPreparing data...")

    # Remove rows without a known target.
    df = df.dropna(subset=[TARGET]).copy()

    # Validate the target before converting it to integers.
    df[TARGET] = pd.to_numeric(df[TARGET], errors="raise")

    if not set(df[TARGET].unique()).issubset({0, 1}):
        raise ValueError(
            f"'{TARGET}' must contain only 0 and 1. "
            f"Found: {df[TARGET].unique()}"
        )

    y = df[TARGET].astype("int8")

    # Exclude identifiers, dates, free-text fields and target.
    # issue_d can be considered later for a time-based evaluation.
    columns_to_drop = [
        "id",
        "issue_d",
        "title",
        "desc",
        TARGET,
    ]

    X = df.drop(columns=columns_to_drop, errors="ignore")

    # Remove columns that contain no usable values.
    empty_columns = X.columns[X.isna().all()].tolist()

    if empty_columns:
        print("Removing entirely empty columns:", empty_columns)
        X = X.drop(columns=empty_columns)

    # Explicitly identify categorical and numerical features.
    categorical_columns = X.select_dtypes(
        include=["object", "category", "bool", "string"]
    ).columns.tolist()

    numeric_columns = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    # If the original dataset stores ZIP codes as numbers, treat them
    # as categories rather than quantities.
    for column in ["zip_code", "addr_state"]:
        if column in X.columns and column not in categorical_columns:
            categorical_columns.append(column)
            if column in numeric_columns:
                numeric_columns.remove(column)

    print(f"Numerical features: {len(numeric_columns)}")
    print(f"Categorical features: {len(categorical_columns)}")
    print(f"Rows available: {len(X):,}")

    print("\nTarget distribution:")
    print(y.value_counts().sort_index())
    print("\nTarget proportions:")
    print(y.value_counts(normalize=True).sort_index().round(4))

    return X, y, numeric_columns, categorical_columns


# ============================================================
# 4. BUILD PREPROCESSING PIPELINE
# ============================================================

def build_preprocessor(numeric_columns, categorical_columns):
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=20,
                    sparse_output=True,
                    dtype="float32",
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
        sparse_threshold=1.0,
    )

    return preprocessor


# ============================================================
# 5. TRAIN RANDOM FOREST
# ============================================================

def train_model(X, y, numeric_columns, categorical_columns):
    print("\nSplitting data into training and testing sets...")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"Training rows: {len(X_train):,}")
    print(f"Testing rows:  {len(X_test):,}")

    preprocessor = build_preprocessor(
        numeric_columns,
        categorical_columns,
    )

    classifier = RandomForestClassifier(
        n_estimators=30,
        max_depth=10,
        min_samples_leaf=5,
        max_features="sqrt",
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=2,
        verbose=1,
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )

    print("\nTraining Random Forest...")
    start_time = time.time()

    model.fit(X_train, y_train)

    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds.")

    return model, X_test, y_test, training_time


# ============================================================
# 6. EVALUATE MODEL
# ============================================================

def evaluate_model(model, X_test, y_test):
    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=["Non-default (0)", "Default (1)"],
            zero_division=0,
        )
    )

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    print(
        "\nROC-AUC:",
        round(roc_auc_score(y_test, probabilities), 4),
    )

    print(
        "PR-AUC (Average Precision):",
        round(average_precision_score(y_test, probabilities), 4),
    )

    print("\nNote: predictions currently use a 0.50 threshold.")
    print("The threshold can be tuned later using validation data.")

    return probabilities


# ============================================================
# 7. SAVE MODEL
# ============================================================

def save_model(model):
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_PATH, compress=3)

    print(f"\nModel saved to: {MODEL_PATH.resolve()}")


# ============================================================
# 8. MAIN
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

    evaluate_model(model, X_test, y_test)

    save_model(model)

    print("\n" + "=" * 60)
    print("ARBITER LOAN MODEL RUN COMPLETED")
    print("=" * 60)
    print(f"Rows used: {len(df):,}")
    print(f"Training time: {training_time:.2f} seconds")
    print(f"Saved artifact: {MODEL_PATH}")


if __name__ == "__main__":
    main()
