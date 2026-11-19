# Indian Loan Dataset — Data Quality Audit
**Generated:** 2026-10-12 15:49 UTC
**Source:** `data/raw/indian_loan/Cleaned_Dataset.csv`
**Script:** `src/audit_india_loan.py`

---

## 1. Shape & Columns

- **Rows:** 24,493
- **Columns:** 22

**Column list:**

| # | Column | dtype |
|---|--------|-------|
| 1 | `applicant_id` | str |
| 2 | `age` | int64 |
| 3 | `gender` | str |
| 4 | `marital_status` | str |
| 5 | `city_tier` | str |
| 6 | `state` | str |
| 7 | `employment_type` | str |
| 8 | `bank_account_vintage_years` | float64 |
| 9 | `monthly_income_inr` | int64 |
| 10 | `existing_loans_count` | int64 |
| 11 | `existing_emi_monthly_inr` | float64 |
| 12 | `credit_utilization_ratio_pct` | float64 |
| 13 | `num_credit_inquiries_last_6m` | float64 |
| 14 | `late_payments_last_12m` | int64 |
| 15 | `loan_amount_requested_inr` | int64 |
| 16 | `loan_tenure_months` | int64 |
| 17 | `loan_purpose` | str |
| 18 | `collateral_provided` | str |
| 19 | `cibil_score` | int64 |
| 20 | `interest_rate_offered_pct` | float64 |
| 21 | `default_risk_score` | float64 |
| 22 | `default_flag` | int64 |

## 2. Target Column Detection

- Expected target `default_flag`: ✅ **found**

**Target distribution (`default_flag`):**

| Value | Count | % |
|-------|-------|---|
| 0 | 22,993 | 93.88% |
| 1 | 1,500 | 6.12% |

- Null target values: **0**
- Unique values: `[0, 1]`
- ✅ Binary 0/1 — looks like a proper default flag

## 3. Missing Values

- ✅ No missing values detected in any column.

## 4. Duplicate Rows & Applicant IDs

- **Duplicate rows:** 0
- **Duplicate `applicant_id` values:** 0
- **Null `applicant_id` values:** 0

> `applicant_id` is an identifier column and **must be excluded from model features**.

## 5. `default_risk_score` — Leakage Investigation

Column `default_risk_score` is present. Checking correlation with `default_flag`:

- Pearson r = **0.4854**
- Spearman ρ = **0.3174**

**`default_risk_score` stats grouped by `default_flag`:**

| default_flag | count | mean | std | min | 25% | 50% | 75% | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 22993.000 | 4.782 | 8.695 | 0.000 | 0.490 | 1.580 | 4.810 | 97.100 |
| 1 | 1500.000 | 28.911 | 24.793 | 0.090 | 8.438 | 21.810 | 44.652 | 99.080 |


> ⚠️ **LEAKAGE RISK (MODERATE):** Correlation |r| = 0.4854. Exclude until provenance is confirmed.
- Class score ranges overlap: **True**

**Verdict:** `MODERATE_LEAKAGE_SUSPECTED` — Excluded from baseline model features.


## 6. Data Types & Value Range Checks

- **Numeric columns (14):** ['age', 'bank_account_vintage_years', 'monthly_income_inr', 'existing_loans_count', 'existing_emi_monthly_inr', 'credit_utilization_ratio_pct', 'num_credit_inquiries_last_6m', 'late_payments_last_12m', 'loan_amount_requested_inr', 'loan_tenure_months', 'cibil_score', 'interest_rate_offered_pct', 'default_risk_score', 'default_flag']
- **Categorical columns (8):** ['applicant_id', 'gender', 'marital_status', 'city_tier', 'state', 'employment_type', 'loan_purpose', 'collateral_provided']

**Numeric column summary (min / max / mean / std):**

| Column | min | max | mean | std | neg_count |
|--------|-----|-----|------|-----|-----------|
| `age` | 21.00 | 65.00 | 35.31 | 8.56 | 0 |
| `bank_account_vintage_years` | 0.50 | 25.00 | 3.98 | 3.85 | 0 |
| `monthly_income_inr` | 12000.00 | 727000.00 | 54784.48 | 38215.06 | 0 |
| `existing_loans_count` | 0.00 | 6.00 | 1.10 | 1.05 | 0 |
| `existing_emi_monthly_inr` | 0.00 | 202300.00 | 8437.61 | 10628.38 | 0 |
| `credit_utilization_ratio_pct` | 0.00 | 100.00 | 41.55 | 19.78 | 0 |
| `num_credit_inquiries_last_6m` | 0.00 | 9.00 | 1.30 | 1.13 | 0 |
| `late_payments_last_12m` | 0.00 | 8.00 | 1.45 | 1.27 | 0 |
| `loan_amount_requested_inr` | 25000.00 | 1519000.00 | 158493.41 | 104409.26 | 0 |
| `loan_tenure_months` | 12.00 | 84.00 | 42.41 | 19.21 | 0 |
| `cibil_score` | 360.00 | 866.00 | 615.35 | 63.00 | 0 |
| `interest_rate_offered_pct` | 10.57 | 26.00 | 20.72 | 3.41 | 0 |
| `default_risk_score` | 0.00 | 99.08 | 6.26 | 11.92 | 0 |
| `default_flag` | 0.00 | 1.00 | 0.06 | 0.24 | 0 |

- ✅ No impossible values detected in numeric columns.

## 7. Categorical Column Distributions


**`gender`** — 3 unique values:

| Value | Count | % |
|-------|-------|---|
| Male | 14,616 | 59.67% |
| Female | 9,404 | 38.39% |
| Other | 473 | 1.93% |

**`marital_status`** — 4 unique values:

| Value | Count | % |
|-------|-------|---|
| Married | 14,251 | 58.18% |
| Single | 7,757 | 31.67% |
| Divorced | 1,493 | 6.1% |
| Widowed | 992 | 4.05% |

**`city_tier`** — 3 unique values:

| Value | Count | % |
|-------|-------|---|
| Tier 1 | 9,760 | 39.85% |
| Tier 2 | 9,332 | 38.1% |
| Tier 3 | 5,401 | 22.05% |

**`state`** — 15 unique values:

| Value | Count | % |
|-------|-------|---|
| Gujarat | 1,719 | 7.02% |
| Tamil Nadu | 1,701 | 6.94% |
| Andhra Pradesh | 1,696 | 6.92% |
| Rajasthan | 1,692 | 6.91% |
| Kerala | 1,666 | 6.8% |
| Telangana | 1,646 | 6.72% |
| Uttar Pradesh | 1,634 | 6.67% |
| Karnataka | 1,631 | 6.66% |
| Madhya Pradesh | 1,614 | 6.59% |
| West Bengal | 1,606 | 6.56% |
| Maharashtra | 1,596 | 6.52% |
| Bihar | 1,593 | 6.5% |
| Delhi NCR | 1,582 | 6.46% |
| Punjab | 1,565 | 6.39% |
| Haryana | 1,552 | 6.34% |

**`employment_type`** — 5 unique values:

| Value | Count | % |
|-------|-------|---|
| Salaried - Private | 11,023 | 45.0% |
| Salaried - Government | 3,733 | 15.24% |
| Self-Employed Professional | 3,671 | 14.99% |
| Business Owner | 3,627 | 14.81% |
| Salaried - PSU | 2,439 | 9.96% |

**`loan_purpose`** — 8 unique values:

| Value | Count | % |
|-------|-------|---|
| Personal Expense | 5,487 | 22.4% |
| Wedding | 3,393 | 13.85% |
| Home Renovation | 3,231 | 13.19% |
| Medical Emergency | 3,174 | 12.96% |
| Vehicle Purchase | 2,423 | 9.89% |
| Debt Consolidation | 2,418 | 9.87% |
| Education | 2,396 | 9.78% |
| Business Expansion | 1,971 | 8.05% |

**`collateral_provided`** — 2 unique values:

| Value | Count | % |
|-------|-------|---|
| No | 19,142 | 78.15% |
| Yes | 5,351 | 21.85% |

## 8. Suspicious Perfect Relationships

- ✅ No column has |correlation| > 0.95 with target (excluding `default_risk_score`).

## 9. Proposed Feature Exclusions for Baseline

| Column | Reason for Exclusion |
|--------|----------------------|
| `applicant_id` | Identifier — must never be a feature |
| `default_risk_score` | Unverified provenance / likely post-loan leakage risk |

**Proposed baseline features (19):**

`age`, `gender`, `marital_status`, `city_tier`, `state`, `employment_type`, `bank_account_vintage_years`, `monthly_income_inr`, `existing_loans_count`, `existing_emi_monthly_inr`, `credit_utilization_ratio_pct`, `num_credit_inquiries_last_6m`, `late_payments_last_12m`, `loan_amount_requested_inr`, `loan_tenure_months`, `loan_purpose`, `collateral_provided`, `cibil_score`, `interest_rate_offered_pct`

## 10. Overall Data-Quality Verdict


✅ **PASS — No critical issues detected. Dataset is conditionally suitable for experimental baseline.**

**Issues log:**
- *(none)*

**Unresolved concerns requiring human verification:**
- `default_risk_score` formula/source is not documented in the CSV or its README.
  - Cannot confirm whether it uses post-loan outcome data.
  - **Action required:** Obtain dataset documentation or compute the score manually from applicant-time features before including it.
- Dataset provenance is synthetic or anonymised; cannot confirm it represents real-world Indian lending population distributions.
