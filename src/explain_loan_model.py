
from pathlib import Path

import joblib
import lime
import lime.lime_tabular
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from scipy import sparse
from sklearn.model_selection import train_test_split


# ============================================================
# 1. CONFIGURATION
# ============================================================

DATA_PATH = Path("data/raw/lendingclub/LC_loans_granting_model_dataset.csv")
MODEL_PATH = Path(
    "artifacts/ablation/loan_xgboost_without_zip.joblib"
)
OUTPUT_DIR = Path("artifacts/explainability")

TARGET = "Default"
MAX_ROWS = 50000
RANDOM_STATE = 42

# Number of loans used for the SHAP summary.
SHAP_SAMPLE_SIZE = 100

# Number of training rows used as the LIME reference sample.
LIME_REFERENCE_SIZE = 500


# ============================================================
# 2. LOAD MODEL AND DATA
# ============================================================

def load_model_and_data():
    print("=" * 60)
    print("ARBITER - SHAP AND LIME EXPLAINABILITY")
    print("=" * 60)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH.resolve()}"
        )

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH.resolve()}"
        )

    print("\nLoading saved XGBoost model...")
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

    empty_columns = X.columns[X.isna().all()].tolist()
    if empty_columns:
        X = X.drop(columns=empty_columns)

    categorical_columns = X.select_dtypes(
        include=["object", "category", "bool", "string"]
    ).columns.tolist()

    numeric_columns = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    for column in ["zip_code", "addr_state"]:
        if column in X.columns and column not in categorical_columns:
            categorical_columns.append(column)
            if column in numeric_columns:
                numeric_columns.remove(column)

    # Reproduce the same split used by the XGBoost training script.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("Dataset loaded:", X.shape)
    print("Test rows:", len(X_test))

    return model, X_train, X_test, y_test


# ============================================================
# 3. TRANSFORM FEATURES
# ============================================================

def transform_features(model, X_train, X_test):
    print("\nTransforming features using the saved pipeline...")

    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    # Use the already-fitted preprocessor; do not fit it again.
    X_train_transformed = preprocessor.transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    feature_names = preprocessor.get_feature_names_out()

    # Explain a small test sample to limit memory use.
    sample_size = min(SHAP_SAMPLE_SIZE, len(X_test))
    X_explain = X_test_transformed[:sample_size]

    # Dense arrays are needed by the plotting and LIME routines.
    if sparse.issparse(X_explain):
        X_explain_dense = X_explain.toarray()
    else:
        X_explain_dense = np.asarray(X_explain)

    reference_size = min(LIME_REFERENCE_SIZE, len(X_train))
    X_reference = X_train_transformed[:reference_size]

    if sparse.issparse(X_reference):
        X_reference_dense = X_reference.toarray()
    else:
        X_reference_dense = np.asarray(X_reference)

    return (
        classifier,
        feature_names,
        X_explain,
        X_explain_dense,
        X_reference_dense,
    )


# ============================================================
# 4. SHAP EXPLANATIONS
# ============================================================

def run_shap(classifier, feature_names, X_explain_dense):
    print("\nCalculating SHAP values...")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Explain the fitted XGBoost classifier.
    explainer = shap.TreeExplainer(classifier)

    shap_output = explainer(
        X_explain_dense,
        check_additivity=False,
    )

    values = np.asarray(shap_output.values)

    # Accommodate SHAP versions that return both class outputs.
    if values.ndim == 3:
        values = values[:, :, 1]

    if values.ndim != 2:
        raise ValueError(
            f"Unexpected SHAP output shape: {values.shape}"
        )

    # --------------------------------------------------------
    # Global importance: mean absolute SHAP value per feature.
    # --------------------------------------------------------

    importance = np.abs(values).mean(axis=0)

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "mean_absolute_shap": importance,
    }).sort_values(
        "mean_absolute_shap",
        ascending=False,
    )

    importance_path = OUTPUT_DIR / "shap_global_importance.csv"
    importance_df.to_csv(importance_path, index=False)

    top = importance_df.head(15).sort_values(
        "mean_absolute_shap"
    )

    plt.figure(figsize=(10, 7))
    plt.barh(
        top["feature"],
        top["mean_absolute_shap"],
    )
    plt.xlabel("Mean absolute SHAP value")
    plt.title("ARBITER - Global SHAP Feature Importance")
    plt.tight_layout()

    global_plot = OUTPUT_DIR / "shap_global_importance.png"
    plt.savefig(global_plot, dpi=160, bbox_inches="tight")
    plt.close()

    # --------------------------------------------------------
    # Local importance for the first explained test loan.
    # --------------------------------------------------------

    local_values = values[0]
    local_row = X_explain_dense[0]

    local_df = pd.DataFrame({
        "feature": feature_names,
        "feature_value": local_row,
        "shap_value": local_values,
    })

    local_df["absolute_shap"] = np.abs(local_df["shap_value"])
    local_df = local_df.sort_values(
        "absolute_shap",
        ascending=False,
    )

    local_path = OUTPUT_DIR / "shap_local_explanation.csv"
    local_df.to_csv(local_path, index=False)

    local_top = local_df.head(15).sort_values("shap_value")

    plt.figure(figsize=(10, 7))
    plt.barh(
        local_top["feature"],
        local_top["shap_value"],
    )
    plt.axvline(0, linewidth=1)
    plt.xlabel("SHAP value (model output contribution)")
    plt.title("ARBITER - SHAP Explanation for One Loan")
    plt.tight_layout()

    local_plot = OUTPUT_DIR / "shap_local_explanation.png"
    plt.savefig(local_plot, dpi=160, bbox_inches="tight")
    plt.close()

    print("Global SHAP plot:", global_plot.resolve())
    print("Global SHAP values:", importance_path.resolve())
    print("Local SHAP plot:", local_plot.resolve())
    print("Local SHAP values:", local_path.resolve())

    print("\nTop five global features:")
    print(importance_df.head(5).to_string(index=False))


# ============================================================
# 5. LIME EXPLANATION
# ============================================================

def run_lime(
    classifier,
    feature_names,
    X_reference_dense,
    X_explain_dense,
):
    print("\nGenerating LIME explanation...")

    # LIME operates on the encoded feature space in this version.
    # Its conditions therefore refer to encoded features.
    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_reference_dense,
        feature_names=list(feature_names),
        class_names=["Non-default", "Default"],
        mode="classification",
        discretize_continuous=True,
        random_state=RANDOM_STATE,
    )

    def predict_proba_for_lime(data):
        # The XGBoost classifier was trained on the transformed features.
        data_sparse = sparse.csr_matrix(data)
        return classifier.predict_proba(data_sparse)

    loan = X_explain_dense[0]

    explanation = lime_explainer.explain_instance(
        data_row=loan,
        predict_fn=predict_proba_for_lime,
        labels=(1,),
        num_features=10,
        num_samples=2000,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    html_path = OUTPUT_DIR / "lime_loan_explanation.html"
    explanation.save_to_file(str(html_path))

    # Save the LIME feature contributions as CSV too.
    lime_values = explanation.as_list(label=1)

    lime_df = pd.DataFrame(
        lime_values,
        columns=["feature_condition", "lime_weight"],
    )

    csv_path = OUTPUT_DIR / "lime_loan_explanation.csv"
    lime_df.to_csv(csv_path, index=False)

    print("LIME HTML:", html_path.resolve())
    print("LIME values:", csv_path.resolve())
    print("\nLIME explanation for Default (class 1):")

    for condition, weight in lime_values:
        print(f"{condition}: {weight:.4f}")


# ============================================================
# 6. REPORT MODEL PREDICTION
# ============================================================

def report_prediction(model, X_test):
    print("\n" + "=" * 60)
    print("EXAMPLE LOAN PREDICTION")
    print("=" * 60)

    loan = X_test.iloc[[0]]
    probability = model.predict_proba(loan)[0, 1]
    prediction = int(probability >= 0.5)

    print("Predicted default probability:", round(float(probability), 4))
    print("Prediction threshold: 0.50")
    print("Predicted class:", prediction)

    if prediction == 1:
        print("Model result: Flagged as potential default.")
    else:
        print("Model result: Predicted non-default.")

    print(
        "This is a model prediction, not a definitive judgment "
        "about the borrower."
    )


# ============================================================
# 7. MAIN
# ============================================================

def main():
    (
        model,
        X_train,
        X_test,
        y_test,
    ) = load_model_and_data()

    (
        classifier,
        feature_names,
        X_explain,
        X_explain_dense,
        X_reference_dense,
    ) = transform_features(model, X_train, X_test)

    report_prediction(model, X_test)

    run_shap(
        classifier,
        feature_names,
        X_explain_dense,
    )

    run_lime(
        classifier,
        feature_names,
        X_reference_dense,
        X_explain_dense,
    )

    print("\n" + "=" * 60)
    print("SHAP AND LIME EXPLANATIONS COMPLETED")
    print("Output directory:", OUTPUT_DIR.resolve())
    print("=" * 60)


if __name__ == "__main__":
    main()
