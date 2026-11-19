"""
audit_india_loan.py
====================
Stage 3A — Indian Loan Dataset Audit

Performs a reproducible data-quality audit on
  data/raw/indian_loan/Cleaned_Dataset.csv

Outputs:
  artifacts/india_loan/audit_report.md
  artifacts/india_loan/audit_summary.json

Usage (from project root, with venv active):
  python src/audit_india_loan.py
"""

import json
import os
import sys
import warnings
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH    = PROJECT_ROOT / "data" / "raw" / "indian_loan" / "Cleaned_Dataset.csv"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "india_loan"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_MD   = ARTIFACT_DIR / "audit_report.md"
SUMMARY_JSON = ARTIFACT_DIR / "audit_summary.json"

# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
def section(title: str) -> str:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)
    return f"\n## {title}\n"

def banner(msg: str):
    print(f"\n  >> {msg}")

# ──────────────────────────────────────────────
# Load
# ──────────────────────────────────────────────
print(f"\n[audit_india_loan] Loading {DATA_PATH}")
if not DATA_PATH.exists():
    sys.exit(f"ERROR: File not found at {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
print(f"Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

# ──────────────────────────────────────────────
# Build report
# ──────────────────────────────────────────────
md_lines: list[str] = []
summary: dict = {"generated_at": datetime.now(timezone.utc).isoformat()}

# ── 1. Shape & columns ──────────────────────────
md_lines.append(section("1. Shape & Columns"))
md_lines.append(f"- **Rows:** {df.shape[0]:,}")
md_lines.append(f"- **Columns:** {df.shape[1]}")
md_lines.append(f"\n**Column list:**\n")
md_lines.append("| # | Column | dtype |")
md_lines.append("|---|--------|-------|")
for i, col in enumerate(df.columns):
    md_lines.append(f"| {i+1} | `{col}` | {df[col].dtype} |")
summary["shape"] = {"rows": df.shape[0], "cols": df.shape[1]}
summary["columns"] = list(df.columns)

banner(f"Columns: {list(df.columns)}")

# ── 2. Target detection ──────────────────────────
md_lines.append(section("2. Target Column Detection"))
EXPECTED_TARGET = "default_flag"
found_target = EXPECTED_TARGET if EXPECTED_TARGET in df.columns else None

if found_target:
    vc = df[found_target].value_counts(dropna=False).to_dict()
    total = len(df)
    pct = {str(k): round(v/total*100, 2) for k, v in vc.items()}
    banner(f"Target '{found_target}' found. Distribution: {vc}")
    md_lines.append(f"- Expected target `{EXPECTED_TARGET}`: ✅ **found**")
    md_lines.append(f"\n**Target distribution (`{found_target}`):**\n")
    md_lines.append("| Value | Count | % |")
    md_lines.append("|-------|-------|---|")
    for k, v in sorted(vc.items(), key=lambda x: str(x[0])):
        md_lines.append(f"| {k} | {v:,} | {pct[str(k)]}% |")
    null_target = int(df[found_target].isna().sum())
    md_lines.append(f"\n- Null target values: **{null_target}**")
    # unique values
    uniq = sorted(df[found_target].dropna().unique().tolist())
    md_lines.append(f"- Unique values: `{uniq}`")
    if set(uniq) == {0, 1}:
        md_lines.append("- ✅ Binary 0/1 — looks like a proper default flag")
    else:
        md_lines.append("- ⚠️ Unexpected values; inspect manually")
else:
    banner(f"WARNING: '{EXPECTED_TARGET}' NOT found in columns!")
    md_lines.append(f"- ❌ Expected target `{EXPECTED_TARGET}` **NOT found**")
    md_lines.append(f"- Available columns: {list(df.columns)}")

summary["target"] = {
    "column": found_target,
    "distribution": {str(k): int(v) for k, v in df[found_target].value_counts(dropna=False).items()} if found_target else None
}

# ── 3. Missing values ───────────────────────────
md_lines.append(section("3. Missing Values"))
null_counts = df.isnull().sum()
null_pct    = (null_counts / len(df) * 100).round(2)
null_df     = pd.DataFrame({"missing_count": null_counts, "missing_pct": null_pct})
null_df     = null_df[null_df["missing_count"] > 0].sort_values("missing_count", ascending=False)

if null_df.empty:
    md_lines.append("- ✅ No missing values detected in any column.")
    banner("No missing values")
else:
    banner(f"Columns with missing values:\n{null_df}")
    md_lines.append("| Column | Missing Count | Missing % |")
    md_lines.append("|--------|--------------|-----------|")
    for col, row in null_df.iterrows():
        md_lines.append(f"| `{col}` | {int(row['missing_count']):,} | {row['missing_pct']}% |")

summary["missing"] = {
    "total_null_cells": int(null_counts.sum()),
    "columns_with_nulls": null_df.to_dict("index")
}

# ── 4. Duplicate rows & applicant IDs ─────────────
md_lines.append(section("4. Duplicate Rows & Applicant IDs"))
dup_rows = int(df.duplicated().sum())
banner(f"Duplicate rows: {dup_rows}")
md_lines.append(f"- **Duplicate rows:** {dup_rows}")

APPLICANT_ID_COL = "applicant_id"
if APPLICANT_ID_COL in df.columns:
    dup_ids  = int(df[APPLICANT_ID_COL].duplicated().sum())
    null_ids = int(df[APPLICANT_ID_COL].isna().sum())
    banner(f"Duplicate applicant_id values: {dup_ids}")
    md_lines.append(f"- **Duplicate `applicant_id` values:** {dup_ids}")
    md_lines.append(f"- **Null `applicant_id` values:** {null_ids}")
    md_lines.append(f"\n> `applicant_id` is an identifier column and **must be excluded from model features**.")
    summary["applicant_id"] = {"duplicate_count": dup_ids, "null_count": null_ids}
else:
    md_lines.append(f"- ℹ️ No `{APPLICANT_ID_COL}` column found.")

summary["duplicate_rows"] = dup_rows

# ── 5. default_risk_score provenance ────────────
md_lines.append(section("5. `default_risk_score` — Leakage Investigation"))
DRS_COL = "default_risk_score"
TARGET  = found_target

if DRS_COL in df.columns and TARGET:
    corr_pearson = df[[DRS_COL, TARGET]].corr().loc[DRS_COL, TARGET]
    corr_spearman = df[[DRS_COL, TARGET]].corr(method="spearman").loc[DRS_COL, TARGET]
    banner(f"Pearson corr({DRS_COL}, {TARGET}) = {corr_pearson:.4f}")
    banner(f"Spearman corr({DRS_COL}, {TARGET}) = {corr_spearman:.4f}")

    drs_by_target = df.groupby(TARGET)[DRS_COL].describe().round(3)
    banner(f"{DRS_COL} stats by target:\n{drs_by_target}")

    md_lines.append(f"Column `{DRS_COL}` is present. Checking correlation with `{TARGET}`:\n")
    md_lines.append(f"- Pearson r = **{corr_pearson:.4f}**")
    md_lines.append(f"- Spearman ρ = **{corr_spearman:.4f}**\n")
    md_lines.append(f"**`{DRS_COL}` stats grouped by `{TARGET}`:**\n")
    # Build markdown table manually (no tabulate dependency needed)
    hdr = "| " + " | ".join([drs_by_target.index.name or "index"] + list(drs_by_target.columns)) + " |"
    sep = "| " + " | ".join(["---"] * (1 + len(drs_by_target.columns))) + " |"
    md_lines.append(hdr)
    md_lines.append(sep)
    for idx_val, row in drs_by_target.iterrows():
        row_str = "| " + str(idx_val) + " | " + " | ".join(f"{v:.3f}" for v in row) + " |"
        md_lines.append(row_str)
    md_lines.append("")

    # Leakage threshold
    if abs(corr_pearson) > 0.7 or abs(corr_spearman) > 0.7:
        verdict = "STRONG_LEAKAGE_SUSPECTED"
        md_lines.append(
            f"\n> ⚠️ **LEAKAGE RISK (HIGH):** Correlation |r| = {abs(corr_pearson):.4f} is very high. "
            "`default_risk_score` should be **excluded** from all baseline models until its formula is verified. "
            "If computed using post-loan outcome, it is a direct information leak."
        )
    elif abs(corr_pearson) > 0.4 or abs(corr_spearman) > 0.4:
        verdict = "MODERATE_LEAKAGE_SUSPECTED"
        md_lines.append(
            f"\n> ⚠️ **LEAKAGE RISK (MODERATE):** Correlation |r| = {abs(corr_pearson):.4f}. "
            "Exclude until provenance is confirmed."
        )
    else:
        verdict = "LOW_CORRELATION_BUT_UNVERIFIED"
        md_lines.append(
            f"\n> ℹ️ Low correlation with target ({corr_pearson:.4f}), but provenance is unverified. "
            "Exclude from baseline as a precaution."
        )

    # Check if score perfectly separates classes
    drs_overlap = (
        df[df[TARGET]==0][DRS_COL].max() > df[df[TARGET]==1][DRS_COL].min() or
        df[df[TARGET]==1][DRS_COL].max() > df[df[TARGET]==0][DRS_COL].min()
    )
    md_lines.append(f"- Class score ranges overlap: **{drs_overlap}**")
    md_lines.append(f"\n**Verdict:** `{verdict}` — Excluded from baseline model features.\n")
    summary["default_risk_score"] = {
        "pearson_corr_with_target": round(float(corr_pearson), 4),
        "spearman_corr_with_target": round(float(corr_spearman), 4),
        "leakage_verdict": verdict
    }
elif DRS_COL not in df.columns:
    md_lines.append(f"- ℹ️ `{DRS_COL}` column not found in dataset.")
    summary["default_risk_score"] = {"present": False}

# ── 6. Data types & impossible values ───────────
md_lines.append(section("6. Data Types & Value Range Checks"))
issues = []

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols     = df.select_dtypes(include=["object", "category"]).columns.tolist()

md_lines.append(f"- **Numeric columns ({len(numeric_cols)}):** {numeric_cols}")
md_lines.append(f"- **Categorical columns ({len(cat_cols)}):** {cat_cols}")
md_lines.append("")

# Numeric summary
md_lines.append("**Numeric column summary (min / max / mean / std):**\n")
md_lines.append("| Column | min | max | mean | std | neg_count |")
md_lines.append("|--------|-----|-----|------|-----|-----------|")

EXPECTED_NON_NEGATIVE = [
    "age", "monthly_income_inr", "existing_loans_count",
    "existing_emi_monthly_inr", "credit_utilization_ratio_pct",
    "num_credit_inquiries_last_6m", "late_payments_last_12m",
    "loan_amount_requested_inr", "loan_tenure_months",
    "cibil_score", "interest_rate_offered_pct", "default_risk_score"
]

impossible_found = []
for col in numeric_cols:
    col_min  = df[col].min()
    col_max  = df[col].max()
    col_mean = df[col].mean()
    col_std  = df[col].std()
    neg_cnt  = int((df[col] < 0).sum())
    md_lines.append(f"| `{col}` | {col_min:.2f} | {col_max:.2f} | {col_mean:.2f} | {col_std:.2f} | {neg_cnt} |")
    if col in EXPECTED_NON_NEGATIVE and neg_cnt > 0:
        impossible_found.append(f"{col}: {neg_cnt} negative values")
        issues.append(f"IMPOSSIBLE_VALUE: {col} has {neg_cnt} negative values")

# Age bounds
if "age" in df.columns:
    age_min = df["age"].min()
    age_max = df["age"].max()
    if age_min < 18:
        issues.append(f"IMPOSSIBLE_VALUE: age has values below 18 (min={age_min})")
        impossible_found.append(f"age: min={age_min} < 18")
    if age_max > 90:
        issues.append(f"IMPOSSIBLE_VALUE: age has values above 90 (max={age_max})")
        impossible_found.append(f"age: max={age_max} > 90")

# CIBIL score bounds (India: 300–900)
if "cibil_score" in df.columns:
    cs_min = df["cibil_score"].min()
    cs_max = df["cibil_score"].max()
    if cs_min < 300 or cs_max > 900:
        issues.append(f"IMPOSSIBLE_VALUE: cibil_score outside 300-900 range (min={cs_min}, max={cs_max})")
        impossible_found.append(f"cibil_score: range [{cs_min}, {cs_max}], expected [300, 900]")

# Credit utilization (0–100%)
if "credit_utilization_ratio_pct" in df.columns:
    cu_min = df["credit_utilization_ratio_pct"].min()
    cu_max = df["credit_utilization_ratio_pct"].max()
    if cu_min < 0 or cu_max > 100:
        issues.append(f"IMPOSSIBLE_VALUE: credit_utilization_ratio_pct outside 0-100% (min={cu_min}, max={cu_max})")
        impossible_found.append(f"credit_utilization_ratio_pct: range [{cu_min}, {cu_max}]")

md_lines.append("")
if impossible_found:
    md_lines.append("**⚠️ Impossible/suspicious values found:**\n")
    for item in impossible_found:
        md_lines.append(f"- {item}")
else:
    md_lines.append("- ✅ No impossible values detected in numeric columns.")

summary["impossible_values"] = {"issues": impossible_found}

# ── 7. Categorical distributions ─────────────────
md_lines.append(section("7. Categorical Column Distributions"))
for col in cat_cols:
    if col == APPLICANT_ID_COL:
        continue
    vc = df[col].value_counts(dropna=False)
    n_unique = df[col].nunique(dropna=False)
    md_lines.append(f"\n**`{col}`** — {n_unique} unique values:")
    md_lines.append("")
    md_lines.append("| Value | Count | % |")
    md_lines.append("|-------|-------|---|")
    for val, cnt in vc.items():
        pct_val = round(cnt / len(df) * 100, 2)
        md_lines.append(f"| {val} | {cnt:,} | {pct_val}% |")
    banner(f"{col}: {vc.to_dict()}")

summary["categorical_distributions"] = {
    col: df[col].value_counts(dropna=False).to_dict()
    for col in cat_cols if col != APPLICANT_ID_COL
}

# ── 8. Suspiciously perfect relationships ─────────
md_lines.append(section("8. Suspicious Perfect Relationships"))
if found_target:
    perfect = []
    for col in df.columns:
        if col in [found_target, APPLICANT_ID_COL, DRS_COL]:
            continue
        try:
            corr_val = abs(df[col].corr(df[found_target]))
            if corr_val > 0.95:
                perfect.append((col, corr_val))
        except Exception:
            pass
    if perfect:
        md_lines.append("**⚠️ Columns with |correlation| > 0.95 with target:**\n")
        for col, c in perfect:
            md_lines.append(f"- `{col}`: r = {c:.4f}")
            issues.append(f"HIGH_CORRELATION_WITH_TARGET: {col} r={c:.4f}")
    else:
        md_lines.append("- ✅ No column has |correlation| > 0.95 with target (excluding `default_risk_score`).")

    banner(f"Perfect-relationship check done. Found: {perfect}")
    summary["perfect_relationships"] = [(col, round(c, 4)) for col, c in perfect]

# ── 9. Feature plan summary ──────────────────────
md_lines.append(section("9. Proposed Feature Exclusions for Baseline"))
excluded = []
if APPLICANT_ID_COL in df.columns:
    excluded.append((APPLICANT_ID_COL, "Identifier — must never be a feature"))
if DRS_COL in df.columns:
    excluded.append((DRS_COL, "Unverified provenance / likely post-loan leakage risk"))

md_lines.append("| Column | Reason for Exclusion |")
md_lines.append("|--------|----------------------|")
for col, reason in excluded:
    md_lines.append(f"| `{col}` | {reason} |")

remaining = [c for c in df.columns if c not in [found_target] + [x[0] for x in excluded]]
md_lines.append(f"\n**Proposed baseline features ({len(remaining)}):**\n")
md_lines.append(", ".join(f"`{c}`" for c in remaining))

summary["exclusions"] = [{"column": c, "reason": r} for c, r in excluded]
summary["proposed_features"] = remaining

# ── 10. Overall verdict ──────────────────────────
md_lines.append(section("10. Overall Data-Quality Verdict"))
critical = [i for i in issues if "IMPOSSIBLE_VALUE" in i or "HIGH_CORR" in i or "STRONG_LEAKAGE" in i]
warnings_list = [i for i in issues if i not in critical]

if not issues and dup_rows == 0:
    verdict_overall = "PASS — No critical issues detected. Dataset is conditionally suitable for experimental baseline."
    md_lines.append(f"\n✅ **{verdict_overall}**")
elif critical:
    verdict_overall = "CONDITIONAL — Critical issues require investigation before training."
    md_lines.append(f"\n⚠️ **{verdict_overall}**")
else:
    verdict_overall = "PASS_WITH_WARNINGS — Minor issues noted; review before production use."
    md_lines.append(f"\n⚠️ **{verdict_overall}**")

md_lines.append(f"\n**Issues log:**")
if issues:
    for iss in issues:
        md_lines.append(f"- `{iss}`")
else:
    md_lines.append("- *(none)*")

md_lines.append(f"""
**Unresolved concerns requiring human verification:**
- `default_risk_score` formula/source is not documented in the CSV or its README.
  - Cannot confirm whether it uses post-loan outcome data.
  - **Action required:** Obtain dataset documentation or compute the score manually from applicant-time features before including it.
- Dataset provenance is synthetic or anonymised; cannot confirm it represents real-world Indian lending population distributions.
""")

summary["verdict"] = verdict_overall
summary["issues"] = issues

# ── Write report ────────────────────────────────
header = f"""# Indian Loan Dataset — Data Quality Audit
**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}
**Source:** `data/raw/indian_loan/Cleaned_Dataset.csv`
**Script:** `src/audit_india_loan.py`

---
"""

full_report = header + "\n".join(md_lines)
REPORT_MD.write_text(full_report, encoding="utf-8")

with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, default=str)

print(f"\n[audit_india_loan] ✅ Report written to: {REPORT_MD}")
print(f"[audit_india_loan] ✅ JSON summary written to: {SUMMARY_JSON}")
print(f"\n{'='*60}")
print(f"  AUDIT COMPLETE — Verdict: {verdict_overall}")
print(f"{'='*60}\n")
