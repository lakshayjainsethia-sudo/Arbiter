"""
test_indian_loan_endpoint.py
=============================
ARBITER Stage 3D — Integration tests for the Indian loan research endpoint.

Tests verify:
  1.  US /predict/loan endpoint still works (non-regression)
  2.  Indian model loads and responds to valid input
  3.  Invalid categorical values are rejected (422)
  4.  Missing required fields are rejected (422)
  5.  Out-of-range numeric values are rejected (422)
  6.  applicant_id in the request body does NOT influence the prediction
  7.  default_risk_score in the request body does NOT influence the prediction
  8.  Indian inputs never reach the US model endpoint
  9.  Model metadata is correct and contains governance fields
  10. Experimental label is present in every prediction response
  11. No real-world validation claim appears in responses

Usage (from project root):
    python -m pytest src/test_indian_loan_endpoint.py -v
"""

import pytest
from fastapi.testclient import TestClient

# Add backend/ to path so we can import main
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
from main import app  # noqa: E402  — must come after sys.path insert

client = TestClient(app)

# ─────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────
VALID_INDIAN_PAYLOAD = {
    "age": 35,
    "gender": "Male",
    "marital_status": "Married",
    "city_tier": "Tier 1",
    "state": "Maharashtra",
    "employment_type": "Salaried",
    "bank_account_vintage_years": 5,
    "monthly_income_inr": 75000,
    "existing_loans_count": 1,
    "existing_emi_monthly_inr": 8000,
    "credit_utilization_ratio_pct": 35.0,
    "num_credit_inquiries_last_6m": 2,
    "late_payments_last_12m": 0,
    "loan_amount_requested_inr": 500000,
    "loan_tenure_months": 36,
    "loan_purpose": "Home Improvement",
    "collateral_provided": "No",
    "cibil_score": 740,
    "interest_rate_offered_pct": 11.5,
}

VALID_US_PAYLOAD = {
    "revenue": 65000.0,
    "dti_n": 16.5,
    "loan_amnt": 12000.0,
    "fico_n": 710.0,
    "experience_c": 8.0,
    "emp_length": "5 years",
    "purpose": "debt_consolidation",
    "home_ownership_n": "MORTGAGE",
    "addr_state": "CA",
}


# ─────────────────────────────────────────────────────────────
# 1. US endpoint non-regression
# ─────────────────────────────────────────────────────────────
class TestUSEndpointNonRegression:
    def test_us_predict_returns_200(self):
        r = client.post("/predict/loan", json=VALID_US_PAYLOAD)
        assert r.status_code == 200, r.text

    def test_us_predict_contains_project_key(self):
        r = client.post("/predict/loan", json=VALID_US_PAYLOAD)
        assert r.json()["project"] == "ARBITER"

    def test_us_predict_contains_default_probability(self):
        r = client.post("/predict/loan", json=VALID_US_PAYLOAD)
        body = r.json()
        assert "default_probability" in body
        assert 0.0 <= body["default_probability"] <= 1.0

    def test_us_predict_contains_review_band(self):
        r = client.post("/predict/loan", json=VALID_US_PAYLOAD)
        body = r.json()
        assert "review_band" in body

    def test_us_health_still_healthy(self):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "healthy"


# ─────────────────────────────────────────────────────────────
# 2. Indian endpoint — valid requests
# ─────────────────────────────────────────────────────────────
class TestIndianLoanValidRequests:
    def test_valid_payload_returns_200(self):
        r = client.post("/predict/indian-loan", json=VALID_INDIAN_PAYLOAD)
        assert r.status_code == 200, r.text

    def test_response_has_project_field(self):
        r = client.post("/predict/indian-loan", json=VALID_INDIAN_PAYLOAD)
        assert r.json()["project"] == "ARBITER"

    def test_prediction_probability_in_range(self):
        r = client.post("/predict/indian-loan", json=VALID_INDIAN_PAYLOAD)
        prob = r.json()["prediction"]["default_probability_raw"]
        assert 0.0 <= prob <= 1.0

    def test_predicted_class_is_binary(self):
        r = client.post("/predict/indian-loan", json=VALID_INDIAN_PAYLOAD)
        cls = r.json()["prediction"]["predicted_class"]
        assert cls in (0, 1)

    def test_cv_threshold_reported_correctly(self):
        r = client.post("/predict/indian-loan", json=VALID_INDIAN_PAYLOAD)
        assert r.json()["prediction"]["cv_threshold"] == pytest.approx(0.7623, abs=1e-4)

    def test_features_used_count(self):
        r = client.post("/predict/indian-loan", json=VALID_INDIAN_PAYLOAD)
        assert r.json()["features_used"] == 19

    def test_all_categories_produce_200(self):
        """Smoke test all valid categorical combinations don't cause server errors."""
        for gender in ["Male", "Female", "Other"]:
            payload = {**VALID_INDIAN_PAYLOAD, "gender": gender}
            r = client.post("/predict/indian-loan", json=payload)
            assert r.status_code == 200, f"Failed for gender={gender}: {r.text}"

        for tier in ["Tier 1", "Tier 2", "Tier 3"]:
            payload = {**VALID_INDIAN_PAYLOAD, "city_tier": tier}
            r = client.post("/predict/indian-loan", json=payload)
            assert r.status_code == 200, f"Failed for city_tier={tier}"

        for collateral in ["Yes", "No"]:
            payload = {**VALID_INDIAN_PAYLOAD, "collateral_provided": collateral}
            r = client.post("/predict/indian-loan", json=payload)
            assert r.status_code == 200, f"Failed for collateral_provided={collateral}"


# ─────────────────────────────────────────────────────────────
# 3. Invalid categorical values are rejected
# ─────────────────────────────────────────────────────────────
class TestIndianLoanCategoryValidation:
    def test_invalid_gender_rejected(self):
        payload = {**VALID_INDIAN_PAYLOAD, "gender": "INVALID_GENDER"}
        r = client.post("/predict/indian-loan", json=payload)
        assert r.status_code == 422, r.text

    def test_invalid_state_rejected(self):
        payload = {**VALID_INDIAN_PAYLOAD, "state": "California"}  # US state
        r = client.post("/predict/indian-loan", json=payload)
        assert r.status_code == 422, r.text

    def test_invalid_city_tier_rejected(self):
        payload = {**VALID_INDIAN_PAYLOAD, "city_tier": "Tier 4"}
        r = client.post("/predict/indian-loan", json=payload)
        assert r.status_code == 422, r.text

    def test_invalid_employment_type_rejected(self):
        payload = {**VALID_INDIAN_PAYLOAD, "employment_type": "Retired"}
        r = client.post("/predict/indian-loan", json=payload)
        assert r.status_code == 422, r.text

    def test_invalid_loan_purpose_rejected(self):
        payload = {**VALID_INDIAN_PAYLOAD, "loan_purpose": "Gambling"}
        r = client.post("/predict/indian-loan", json=payload)
        assert r.status_code == 422, r.text

    def test_invalid_collateral_rejected(self):
        payload = {**VALID_INDIAN_PAYLOAD, "collateral_provided": "Maybe"}
        r = client.post("/predict/indian-loan", json=payload)
        assert r.status_code == 422, r.text

    def test_invalid_marital_status_rejected(self):
        payload = {**VALID_INDIAN_PAYLOAD, "marital_status": "Complicated"}
        r = client.post("/predict/indian-loan", json=payload)
        assert r.status_code == 422, r.text


# ─────────────────────────────────────────────────────────────
# 4. Missing required fields
# ─────────────────────────────────────────────────────────────
class TestIndianLoanMissingFields:
    def test_missing_cibil_score_rejected(self):
        payload = {k: v for k, v in VALID_INDIAN_PAYLOAD.items() if k != "cibil_score"}
        r = client.post("/predict/indian-loan", json=payload)
        assert r.status_code == 422, r.text

    def test_missing_monthly_income_rejected(self):
        payload = {k: v for k, v in VALID_INDIAN_PAYLOAD.items() if k != "monthly_income_inr"}
        r = client.post("/predict/indian-loan", json=payload)
        assert r.status_code == 422, r.text

    def test_empty_body_rejected(self):
        r = client.post("/predict/indian-loan", json={})
        assert r.status_code == 422, r.text


# ─────────────────────────────────────────────────────────────
# 5. Out-of-range numeric values
# ─────────────────────────────────────────────────────────────
class TestIndianLoanRangeValidation:
    def test_cibil_below_300_rejected(self):
        payload = {**VALID_INDIAN_PAYLOAD, "cibil_score": 200}
        r = client.post("/predict/indian-loan", json=payload)
        assert r.status_code == 422, r.text

    def test_cibil_above_900_rejected(self):
        payload = {**VALID_INDIAN_PAYLOAD, "cibil_score": 950}
        r = client.post("/predict/indian-loan", json=payload)
        assert r.status_code == 422, r.text

    def test_age_below_18_rejected(self):
        payload = {**VALID_INDIAN_PAYLOAD, "age": 16}
        r = client.post("/predict/indian-loan", json=payload)
        assert r.status_code == 422, r.text

    def test_age_above_70_rejected(self):
        payload = {**VALID_INDIAN_PAYLOAD, "age": 80}
        r = client.post("/predict/indian-loan", json=payload)
        assert r.status_code == 422, r.text

    def test_credit_utilization_above_100_rejected(self):
        payload = {**VALID_INDIAN_PAYLOAD, "credit_utilization_ratio_pct": 150.0}
        r = client.post("/predict/indian-loan", json=payload)
        assert r.status_code == 422, r.text

    def test_zero_loan_amount_rejected(self):
        payload = {**VALID_INDIAN_PAYLOAD, "loan_amount_requested_inr": 0}
        r = client.post("/predict/indian-loan", json=payload)
        assert r.status_code == 422, r.text

    def test_zero_income_rejected(self):
        payload = {**VALID_INDIAN_PAYLOAD, "monthly_income_inr": 0}
        r = client.post("/predict/indian-loan", json=payload)
        assert r.status_code == 422, r.text


# ─────────────────────────────────────────────────────────────
# 6 & 7. Excluded features cannot influence predictions
# ─────────────────────────────────────────────────────────────
class TestExcludedFeaturesIgnored:
    """applicant_id and default_risk_score must not influence predictions.

    Strategy: submit two requests with identical feature values but different
    extra fields. The endpoint should not accept these fields at all (they are
    not defined in IndianLoanApplication), so Pydantic will simply ignore
    any extra fields sent (default behavior). The probability must be identical
    between the two calls because the model sees identical input.
    """

    def test_applicant_id_does_not_change_prediction(self):
        # Base prediction
        r1 = client.post("/predict/indian-loan", json=VALID_INDIAN_PAYLOAD)
        assert r1.status_code == 200
        prob1 = r1.json()["prediction"]["default_probability_raw"]

        # Add applicant_id — should be silently ignored (extra fields stripped by Pydantic)
        payload_with_id = {**VALID_INDIAN_PAYLOAD, "applicant_id": "HACK_12345"}
        r2 = client.post("/predict/indian-loan", json=payload_with_id)
        assert r2.status_code == 200
        prob2 = r2.json()["prediction"]["default_probability_raw"]

        assert prob1 == pytest.approx(prob2, abs=1e-6), (
            f"applicant_id changed the prediction: {prob1} vs {prob2}"
        )

    def test_default_risk_score_does_not_change_prediction(self):
        r1 = client.post("/predict/indian-loan", json=VALID_INDIAN_PAYLOAD)
        prob1 = r1.json()["prediction"]["default_probability_raw"]

        payload_with_score = {**VALID_INDIAN_PAYLOAD, "default_risk_score": 99.9}
        r2 = client.post("/predict/indian-loan", json=payload_with_score)
        assert r2.status_code == 200
        prob2 = r2.json()["prediction"]["default_probability_raw"]

        assert prob1 == pytest.approx(prob2, abs=1e-6), (
            f"default_risk_score changed the prediction: {prob1} vs {prob2}"
        )

    def test_excluded_features_listed_in_response(self):
        r = client.post("/predict/indian-loan", json=VALID_INDIAN_PAYLOAD)
        excluded = r.json()["excluded_features"]
        assert "applicant_id" in excluded
        assert "default_risk_score" in excluded


# ─────────────────────────────────────────────────────────────
# 8. Indian inputs never reach the US model
# ─────────────────────────────────────────────────────────────
class TestModelIsolation:
    def test_indian_payload_fails_on_us_endpoint(self):
        """The Indian payload (no 'revenue', 'fico_n' etc.) must be rejected by /predict/loan."""
        r = client.post("/predict/loan", json=VALID_INDIAN_PAYLOAD)
        # Should fail validation because Indian fields != US fields
        assert r.status_code == 422, (
            f"Expected 422 but got {r.status_code}: US endpoint accepted Indian payload"
        )

    def test_us_payload_fails_on_indian_endpoint(self):
        """US payload (no 'cibil_score', 'state' as Indian state, etc.) must be rejected."""
        r = client.post("/predict/indian-loan", json=VALID_US_PAYLOAD)
        assert r.status_code == 422, (
            f"Expected 422 but got {r.status_code}: Indian endpoint accepted US payload"
        )

    def test_us_endpoint_unchanged_after_adding_indian_model(self):
        """Double-check US endpoint is unaffected by the Indian model loading."""
        r = client.post("/predict/loan", json=VALID_US_PAYLOAD)
        assert r.status_code == 200
        body = r.json()
        # Ensure response has no Indian-model keys
        assert "cv_threshold" not in body
        assert "cibil_score" not in body
        assert "governance" not in body


# ─────────────────────────────────────────────────────────────
# 9. Metadata endpoint correctness
# ─────────────────────────────────────────────────────────────
class TestIndianMetadataEndpoint:
    def test_metadata_returns_200(self):
        r = client.get("/predict/indian-loan/metadata")
        assert r.status_code == 200

    def test_metadata_has_model_block(self):
        r = client.get("/predict/indian-loan/metadata")
        body = r.json()
        assert "model" in body
        assert body["model"]["features_used"] == 19
        assert body["model"]["cv_threshold"] == pytest.approx(0.7623, abs=1e-4)

    def test_metadata_has_valid_categories(self):
        r = client.get("/predict/indian-loan/metadata")
        cats = r.json()["valid_categories"]
        assert "gender" in cats
        assert "state" in cats
        assert len(cats["state"]) == 15

    def test_metadata_excluded_features_correct(self):
        r = client.get("/predict/indian-loan/metadata")
        excluded = r.json()["model"]["excluded"]
        assert "applicant_id" in excluded
        assert "default_risk_score" in excluded

    def test_metadata_has_governance(self):
        r = client.get("/predict/indian-loan/metadata")
        gov = r.json()["governance"]
        assert gov["gender_in_model"] is True
        assert gov["fairness_review_required"] is True
        assert gov["real_world_validation"] is False

    def test_metadata_has_shap_features(self):
        r = client.get("/predict/indian-loan/metadata")
        shap_feats = r.json()["shap_top_features"]
        assert len(shap_feats) >= 5
        # First feature should be cibil_score
        assert shap_feats[0]["feature"] == "cibil_score"


# ─────────────────────────────────────────────────────────────
# 10. Experimental label present
# ─────────────────────────────────────────────────────────────
class TestExperimentalLabelling:
    def test_experiment_label_in_prediction_response(self):
        r = client.post("/predict/indian-loan", json=VALID_INDIAN_PAYLOAD)
        body = r.json()
        assert body["model"]["experiment_label"] == "EXPERIMENTAL"

    def test_dataset_caveat_non_empty(self):
        r = client.post("/predict/indian-loan", json=VALID_INDIAN_PAYLOAD)
        caveat = r.json()["model"]["dataset_caveat"]
        assert isinstance(caveat, str) and len(caveat) > 20

    def test_governance_warnings_non_empty(self):
        r = client.post("/predict/indian-loan", json=VALID_INDIAN_PAYLOAD)
        warnings = r.json()["governance"]["warnings"]
        assert len(warnings) >= 3


# ─────────────────────────────────────────────────────────────
# 11. No real-world validation claim
# ─────────────────────────────────────────────────────────────
class TestNoRealWorldClaim:
    def test_real_world_validation_false(self):
        r = client.post("/predict/indian-loan", json=VALID_INDIAN_PAYLOAD)
        assert r.json()["governance"]["real_world_validation"] is False

    def test_no_approved_or_rejected_in_response(self):
        """The words 'approved' and 'rejected' must not appear in the response."""
        r = client.post("/predict/indian-loan", json=VALID_INDIAN_PAYLOAD)
        text = str(r.json()).lower()
        assert "approved" not in text, "Response contains the word 'approved'"
        assert "rejected" not in text, "Response contains the word 'rejected'"

    def test_probability_label_warns_uncalibrated(self):
        r = client.post("/predict/indian-loan", json=VALID_INDIAN_PAYLOAD)
        label = r.json()["prediction"]["probability_label"]
        assert "UNCALIBRATED" in label or "uncalibrated" in label.lower()
