"""
tests/test_metrics.py
=====================
66-test pytest suite for Credit Agent's deterministic credit metric functions.

Tests cover:
  - EBITDA computation (standard, with rent, missing inputs)
  - Free Cash Flow computation
  - Fixed Charge Coverage (FCC) computation
  - Leverage ratio computation
  - EBITDA margin computation
  - Sign convention handling (capex as positive or negative)
  - None propagation (missing inputs return None, not errors)
  - Zero denominator handling
  - Covenant breach detection
  - A full realistic scenario modelled on Church & Dwight (CHD) FY2024

Run with:
    cd /Users/liamcleary/Optimus2-backend-for-credit-AI
    pytest tests/test_metrics.py -v
"""

import pytest
from metrics import (
    safe_divide,
    abs_outflow,
    compute_ebitda,
    compute_free_cash_flow,
    compute_fcc,
    compute_leverage,
    compute_ebitda_margin,
)


# ── safe_divide ───────────────────────────────────────────────────────────────

class TestSafeDivide:
    def test_basic_division(self):
        assert safe_divide(10.0, 2.0) == 5.0

    def test_returns_none_when_numerator_is_none(self):
        assert safe_divide(None, 2.0) is None

    def test_returns_none_when_denominator_is_none(self):
        assert safe_divide(10.0, None) is None

    def test_returns_none_when_both_none(self):
        assert safe_divide(None, None) is None

    def test_returns_none_on_zero_denominator(self):
        assert safe_divide(10.0, 0.0) is None

    def test_handles_negative_numerator(self):
        assert safe_divide(-10.0, 2.0) == -5.0

    def test_handles_negative_denominator(self):
        assert safe_divide(10.0, -2.0) == -5.0

    def test_decimal_result(self):
        result = safe_divide(1.0, 3.0)
        assert abs(result - 0.3333) < 0.001

    def test_integer_inputs(self):
        assert safe_divide(9, 3) == 3.0

    def test_large_numbers(self):
        assert safe_divide(1_000_000, 1_000) == 1000.0


# ── abs_outflow ───────────────────────────────────────────────────────────────

class TestAbsOutflow:
    def test_positive_value_unchanged(self):
        assert abs_outflow(250.0) == 250.0

    def test_negative_value_converted_to_positive(self):
        assert abs_outflow(-250.0) == 250.0

    def test_returns_none_for_none_input(self):
        assert abs_outflow(None) is None

    def test_zero_input(self):
        assert abs_outflow(0.0) == 0.0

    def test_large_negative(self):
        assert abs_outflow(-1_500.0) == 1_500.0


# ── compute_ebitda ────────────────────────────────────────────────────────────

class TestComputeEbitda:
    def test_standard_ebitda_no_rent(self):
        # EBITDA = 500 + 100 = 600
        result = compute_ebitda(
            operating_income=500.0,
            depreciation_amortization=100.0,
            rent_expense=None,
            include_rent=False,
        )
        assert result == 600.0

    def test_ebitda_with_rent_included(self):
        # EBITDA = 500 + 100 + 50 = 650
        result = compute_ebitda(
            operating_income=500.0,
            depreciation_amortization=100.0,
            rent_expense=50.0,
            include_rent=True,
        )
        assert result == 650.0

    def test_ebitda_rent_excluded_when_flag_false(self):
        # Rent is present but include_rent=False so it's ignored
        result = compute_ebitda(
            operating_income=500.0,
            depreciation_amortization=100.0,
            rent_expense=50.0,
            include_rent=False,
        )
        assert result == 600.0

    def test_returns_none_when_operating_income_missing(self):
        result = compute_ebitda(
            operating_income=None,
            depreciation_amortization=100.0,
        )
        assert result is None

    def test_returns_none_when_da_missing(self):
        result = compute_ebitda(
            operating_income=500.0,
            depreciation_amortization=None,
        )
        assert result is None

    def test_returns_none_when_both_missing(self):
        result = compute_ebitda(
            operating_income=None,
            depreciation_amortization=None,
        )
        assert result is None

    def test_negative_operating_income(self):
        # Company is operating at a loss
        result = compute_ebitda(
            operating_income=-200.0,
            depreciation_amortization=100.0,
        )
        assert result == -100.0

    def test_rent_none_with_include_rent_true(self):
        # include_rent=True but no rent available — should still work
        result = compute_ebitda(
            operating_income=500.0,
            depreciation_amortization=100.0,
            rent_expense=None,
            include_rent=True,
        )
        assert result == 600.0

    def test_zero_da(self):
        result = compute_ebitda(
            operating_income=500.0,
            depreciation_amortization=0.0,
        )
        assert result == 500.0

    def test_zero_operating_income(self):
        result = compute_ebitda(
            operating_income=0.0,
            depreciation_amortization=100.0,
        )
        assert result == 100.0


# ── compute_free_cash_flow ────────────────────────────────────────────────────

class TestComputeFreeCashFlow:
    def test_basic_fcf(self):
        # FCF = 800 - 200 = 600
        assert compute_free_cash_flow(cfo=800.0, capex=200.0) == 600.0

    def test_capex_as_negative(self):
        # Capex reported as negative — abs_outflow normalises it
        assert compute_free_cash_flow(cfo=800.0, capex=-200.0) == 600.0

    def test_returns_none_when_cfo_missing(self):
        assert compute_free_cash_flow(cfo=None, capex=200.0) is None

    def test_returns_none_when_capex_missing(self):
        assert compute_free_cash_flow(cfo=800.0, capex=None) is None

    def test_negative_fcf(self):
        # Company spending more on capex than it generates from operations
        assert compute_free_cash_flow(cfo=100.0, capex=300.0) == -200.0

    def test_zero_capex(self):
        assert compute_free_cash_flow(cfo=800.0, capex=0.0) == 800.0


# ── compute_fcc ───────────────────────────────────────────────────────────────

class TestComputeFcc:
    def test_healthy_fcc(self):
        # FCC = (600 - 100 - 80) / (50 + 120) = 420 / 170 = 2.47x
        result = compute_fcc(
            ebitda=600.0,
            capex=100.0,
            cash_taxes=80.0,
            cpltd=50.0,
            cash_interest=120.0,
        )
        assert abs(result - 2.47) < 0.01

    def test_fcc_below_one_covenant_breach(self):
        # FCC < 1.0 means the company cannot cover its fixed charges
        result = compute_fcc(
            ebitda=200.0,
            capex=150.0,
            cash_taxes=80.0,
            cpltd=50.0,
            cash_interest=120.0,
        )
        assert result < 1.0

    def test_capex_as_negative_sign_convention(self):
        # Capex reported as negative — should produce same result
        result_pos = compute_fcc(600.0, 100.0, 80.0, 50.0, 120.0)
        result_neg = compute_fcc(600.0, -100.0, 80.0, 50.0, 120.0)
        assert abs(result_pos - result_neg) < 0.001

    def test_cash_taxes_as_negative_sign_convention(self):
        result_pos = compute_fcc(600.0, 100.0, 80.0, 50.0, 120.0)
        result_neg = compute_fcc(600.0, 100.0, -80.0, 50.0, 120.0)
        assert abs(result_pos - result_neg) < 0.001

    def test_returns_none_when_ebitda_missing(self):
        assert compute_fcc(None, 100.0, 80.0, 50.0, 120.0) is None

    def test_returns_none_when_capex_missing(self):
        assert compute_fcc(600.0, None, 80.0, 50.0, 120.0) is None

    def test_returns_none_when_cash_taxes_missing(self):
        assert compute_fcc(600.0, 100.0, None, 50.0, 120.0) is None

    def test_returns_none_when_cpltd_missing(self):
        assert compute_fcc(600.0, 100.0, 80.0, None, 120.0) is None

    def test_returns_none_when_cash_interest_missing(self):
        assert compute_fcc(600.0, 100.0, 80.0, 50.0, None) is None

    def test_zero_cpltd_and_interest(self):
        # Denominator is zero — should return None (no fixed charges)
        assert compute_fcc(600.0, 100.0, 80.0, 0.0, 0.0) is None

    def test_fcc_exactly_one(self):
        # FCC = 1.0 means exactly breaking even on fixed charges
        result = compute_fcc(
            ebitda=300.0,
            capex=100.0,
            cash_taxes=50.0,
            cpltd=50.0,
            cash_interest=100.0,
        )
        assert abs(result - 1.0) < 0.001


# ── compute_leverage ──────────────────────────────────────────────────────────

class TestComputeLeverage:
    def test_standard_leverage(self):
        # 3.0x leverage
        assert compute_leverage(total_debt=1500.0, ebitda=500.0) == 3.0

    def test_returns_none_when_debt_missing(self):
        assert compute_leverage(total_debt=None, ebitda=500.0) is None

    def test_returns_none_when_ebitda_missing(self):
        assert compute_leverage(total_debt=1500.0, ebitda=None) is None

    def test_returns_none_when_ebitda_zero(self):
        # Leverage is undefined when EBITDA = 0
        assert compute_leverage(total_debt=1500.0, ebitda=0.0) is None

    def test_returns_none_when_ebitda_negative(self):
        # Leverage is not meaningful when EBITDA is negative
        assert compute_leverage(total_debt=1500.0, ebitda=-100.0) is None

    def test_zero_debt(self):
        # Company with no debt has 0x leverage
        assert compute_leverage(total_debt=0.0, ebitda=500.0) == 0.0

    def test_high_leverage(self):
        # 7x leverage — highly leveraged
        result = compute_leverage(total_debt=3500.0, ebitda=500.0)
        assert result == 7.0

    def test_low_leverage(self):
        # 1.5x leverage — conservative
        result = compute_leverage(total_debt=750.0, ebitda=500.0)
        assert result == 1.5


# ── compute_ebitda_margin ─────────────────────────────────────────────────────

class TestComputeEbitdaMargin:
    def test_standard_margin(self):
        # 20% EBITDA margin
        result = compute_ebitda_margin(ebitda=200.0, revenue=1000.0)
        assert result == 0.20

    def test_returns_none_when_ebitda_missing(self):
        assert compute_ebitda_margin(ebitda=None, revenue=1000.0) is None

    def test_returns_none_when_revenue_missing(self):
        assert compute_ebitda_margin(ebitda=200.0, revenue=None) is None

    def test_returns_none_when_revenue_zero(self):
        assert compute_ebitda_margin(ebitda=200.0, revenue=0.0) is None

    def test_negative_margin(self):
        # Company generating negative EBITDA (operating losses)
        result = compute_ebitda_margin(ebitda=-100.0, revenue=1000.0)
        assert result == -0.10

    def test_high_margin(self):
        # 45% margin (e.g. software company)
        result = compute_ebitda_margin(ebitda=450.0, revenue=1000.0)
        assert result == 0.45


# ── Covenant breach detection ─────────────────────────────────────────────────

class TestCovenantBreachDetection:
    """
    Tests that simulate checking extracted metrics against covenant thresholds.
    In the platform, these checks happen in validate_and_compute() and the
    results are added to validation_flags.
    """

    def test_leverage_within_covenant(self):
        leverage = compute_leverage(total_debt=1500.0, ebitda=500.0)
        max_leverage = 4.5
        assert leverage <= max_leverage, f"Leverage {leverage}x breaches {max_leverage}x covenant"

    def test_leverage_breaches_covenant(self):
        leverage = compute_leverage(total_debt=3000.0, ebitda=500.0)
        max_leverage = 4.5
        assert leverage > max_leverage

    def test_fcc_within_covenant(self):
        fcc = compute_fcc(
            ebitda=600.0, capex=100.0, cash_taxes=80.0,
            cpltd=50.0, cash_interest=120.0,
        )
        min_fcc = 1.15
        assert fcc >= min_fcc, f"FCC {fcc:.2f}x breaches {min_fcc}x covenant"

    def test_fcc_breaches_covenant(self):
        fcc = compute_fcc(
            ebitda=250.0, capex=100.0, cash_taxes=80.0,
            cpltd=50.0, cash_interest=120.0,
        )
        min_fcc = 1.15
        assert fcc < min_fcc

    def test_covenant_not_checkable_when_metric_none(self):
        # If leverage can't be computed, we can't determine a breach
        leverage = compute_leverage(total_debt=None, ebitda=500.0)
        assert leverage is None


# ── Realistic CHD scenario ────────────────────────────────────────────────────

class TestRealisticCHDScenario:
    """
    Full end-to-end scenario modelled on Church & Dwight (CHD) FY2024 financials.
    Values are approximate and used for testing purposes only.

    CHD FY2024 approximate figures (in $MM):
      Revenue:          $5,900
      Operating Income: $1,000
      D&A:              $220
      Rent Expense:     $40
      Total Debt:       $2,800
      CFO:              $1,050
      Capex:            $210
      Cash Interest:    $95
      Cash Taxes:       $220
      CPLTD:            $500
    """

    @pytest.fixture
    def chd_fy2024(self):
        return {
            "revenue":            5_900.0,
            "operating_income":   1_000.0,
            "da":                 220.0,
            "rent_expense":       40.0,
            "total_debt":         2_800.0,
            "cfo":                1_050.0,
            "capex":              210.0,
            "cash_interest":      95.0,
            "cash_taxes":         220.0,
            "cpltd":              500.0,
        }

    def test_ebitda_computed_correctly(self, chd_fy2024):
        ebitda = compute_ebitda(
            chd_fy2024["operating_income"],
            chd_fy2024["da"],
            chd_fy2024["rent_expense"],
            include_rent=True,
        )
        assert ebitda == 1_260.0  # 1000 + 220 + 40

    def test_leverage_in_reasonable_range(self, chd_fy2024):
        ebitda = compute_ebitda(
            chd_fy2024["operating_income"],
            chd_fy2024["da"],
        )
        leverage = compute_leverage(chd_fy2024["total_debt"], ebitda)
        assert leverage is not None
        assert 1.5 <= leverage <= 4.0, f"CHD leverage {leverage:.2f}x outside expected range"

    def test_fcf_is_positive(self, chd_fy2024):
        fcf = compute_free_cash_flow(chd_fy2024["cfo"], chd_fy2024["capex"])
        assert fcf > 0, "CHD should have positive FCF"
        assert fcf == 840.0  # 1050 - 210

    def test_fcc_above_covenant_threshold(self, chd_fy2024):
        ebitda = compute_ebitda(
            chd_fy2024["operating_income"],
            chd_fy2024["da"],
        )
        fcc = compute_fcc(
            ebitda=ebitda,
            capex=chd_fy2024["capex"],
            cash_taxes=chd_fy2024["cash_taxes"],
            cpltd=chd_fy2024["cpltd"],
            cash_interest=chd_fy2024["cash_interest"],
        )
        assert fcc is not None
        assert fcc >= 1.15, f"CHD FCC {fcc:.2f}x below 1.15x covenant"

    def test_ebitda_margin_in_reasonable_range(self, chd_fy2024):
        ebitda = compute_ebitda(
            chd_fy2024["operating_income"],
            chd_fy2024["da"],
        )
        margin = compute_ebitda_margin(ebitda, chd_fy2024["revenue"])
        assert margin is not None
        assert 0.15 <= margin <= 0.30, f"CHD EBITDA margin {margin:.1%} outside expected range"

    def test_capex_sign_convention_does_not_affect_fcf(self, chd_fy2024):
        fcf_positive_capex = compute_free_cash_flow(chd_fy2024["cfo"], 210.0)
        fcf_negative_capex = compute_free_cash_flow(chd_fy2024["cfo"], -210.0)
        assert fcf_positive_capex == fcf_negative_capex

    def test_all_metrics_computable(self, chd_fy2024):
        """Verify all five core metrics are computable from CHD data."""
        ebitda = compute_ebitda(chd_fy2024["operating_income"], chd_fy2024["da"])
        fcf      = compute_free_cash_flow(chd_fy2024["cfo"], chd_fy2024["capex"])
        leverage = compute_leverage(chd_fy2024["total_debt"], ebitda)
        fcc      = compute_fcc(ebitda, chd_fy2024["capex"], chd_fy2024["cash_taxes"],
                               chd_fy2024["cpltd"], chd_fy2024["cash_interest"])
        margin   = compute_ebitda_margin(ebitda, chd_fy2024["revenue"])

        assert ebitda   is not None, "EBITDA not computable"
        assert fcf      is not None, "FCF not computable"
        assert leverage is not None, "Leverage not computable"
        assert fcc      is not None, "FCC not computable"
        assert margin   is not None, "EBITDA margin not computable"
