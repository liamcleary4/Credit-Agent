"""
metrics.py
==========
All credit metric calculations used in the underwriting memo.

Every function here is DETERMINISTIC — meaning it uses only basic arithmetic
and returns the same answer every time for the same inputs. No AI is involved.
This is intentional: credit metrics need to be auditable and reproducible.

The formulas used match what commercial credit analysts calculate manually
when "spreading" a company's financials. The definitions (especially FCC) are
specific to this platform and reflect how the analyst team defines them.

WHY THESE SPECIFIC FORMULAS:
  EBITDA  = Operating Income + D&A (+ Rent if including lease expense)
             This is the standard proxy for "cash earnings before financing costs"
  FCC     = (EBITDA - Capex - Cash Taxes) / (Current Portion of LTD + Cash Interest)
             Measures whether the company generates enough cash to service its debt
             after spending what it needs to maintain/grow the business
  Leverage = Total Debt / EBITDA
             The most common debt capacity measure in commercial lending
  FCF     = CFO - |Capex|
             How much cash is left after the company pays to maintain its assets
  Altman Z and PD Score are in credit_scoring.py

IMPORTANT: All input values are expected to be in the same unit (millions, thousands,
or actual dollars). The statement_basis field in schemas.py tells you which unit
the extracted data uses. These functions do NOT do unit conversion.

None-handling: If any required input is None (meaning extraction failed or the
value wasn't found in the filing), the function returns None rather than
throwing an error. The calling code in validate_and_compute() (agents.py)
treats None as "not computable" and flags it in the validation notes.
"""

from __future__ import annotations

from typing import Optional


def safe_divide(n: Optional[float], d: Optional[float]) -> Optional[float]:
    """
    Divides two numbers, returning None instead of crashing if either is None
    or if the denominator is zero.

    Used everywhere we need to avoid division-by-zero errors in ratio calculations.
    For example, if EBITDA is zero, leverage (debt/EBITDA) would be infinite —
    we return None instead, which the UI displays as "N/A".
    """
    if n is None or d is None:
        return None
    if d == 0:
        return None
    return n / d


def abs_outflow(x: Optional[float]) -> Optional[float]:
    """
    Normalises a cash outflow to a positive number regardless of how it was
    recorded in the filing.

    WHY THIS EXISTS:
    Cash outflows (capex, taxes paid, interest paid) can appear as either
    positive or negative numbers depending on how the company formats its
    cash flow statement. Some companies show capex as -250, others as 250.
    This function converts both to 250 so our calculations are consistent.
    """
    if x is None:
        return None
    return abs(float(x))


def compute_ebitda(
    operating_income: Optional[float],
    depreciation_amortization: Optional[float],
    rent_expense: Optional[float] = None,
    include_rent: bool = True,
) -> Optional[float]:
    """
    Computes EBITDA (Earnings Before Interest, Taxes, Depreciation & Amortization).

    FORMULA: EBITDA = Operating Income + D&A [+ Rent Expense]

    WHY WE ADD D&A:
    D&A is a non-cash expense — it reduces reported earnings but doesn't
    actually consume cash. Adding it back gives a cleaner picture of how
    much cash the business generates from operations.

    WHY WE OPTIONALLY ADD RENT:
    For companies with significant operating leases (e.g. retailers), rent
    is often added back to create "EBITDAR" (the R stands for Rent). This
    is common in lease-heavy industries like retail and restaurants where
    rent is effectively a financing cost rather than an operating cost.
    The include_rent flag controls this. Set include_rent=False for the
    completeness score calculation, where we want a conservative EBITDA.

    Returns None if operating_income or D&A are unavailable — these are
    the minimum required to compute any form of EBITDA.
    """
    if operating_income is None or depreciation_amortization is None:
        return None

    # Start with Operating Income + D&A (the standard EBITDA formula)
    ebitda = float(operating_income) + float(depreciation_amortization)

    # Optionally add rent expense back in (creating EBITDAR)
    if include_rent and rent_expense is not None:
        ebitda += float(rent_expense)

    return ebitda


def compute_free_cash_flow(cfo: Optional[float], capex: Optional[float]) -> Optional[float]:
    """
    Computes Free Cash Flow (FCF).

    FORMULA: FCF = Operating Cash Flow - |Capex|

    WHY THIS MATTERS IN CREDIT:
    FCF represents the cash a company has left over after paying for the
    capital expenditures needed to maintain or grow its business. A company
    with positive and growing FCF can service debt and reinvest without
    needing to borrow more. Negative FCF means the company is burning cash.

    We use |capex| (absolute value) because capex can be reported as either
    a positive or negative number in cash flow statements. See abs_outflow().
    """
    capex_mag = abs_outflow(capex)
    if cfo is None or capex_mag is None:
        return None
    return float(cfo) - capex_mag


def compute_fcc(
    ebitda: Optional[float],
    capex: Optional[float],
    cash_taxes: Optional[float],
    cpltd: Optional[float],
    cash_interest: Optional[float],
) -> Optional[float]:
    """
    Computes the Fixed Charge Coverage (FCC) ratio.

    FORMULA:
      Numerator:   EBITDA - |Capex| - |Cash Taxes Paid|
      Denominator: Current Portion of Long-Term Debt + |Cash Interest Paid|
      FCC = Numerator / Denominator

    HOW TO INTERPRET FCC:
      FCC > 1.2x  → The company comfortably covers its fixed charges (good)
      FCC = 1.0x  → The company barely breaks even — no cushion (concerning)
      FCC < 1.0x  → The company cannot cover its fixed charges from operations (bad)

    WHAT FCC MEASURES:
    FCC answers the question: "After the company spends what it needs on capex
    and taxes, does it generate enough cash to pay its debt obligations (principal
    and interest)?"

    WHY THIS DEFINITION SPECIFICALLY:
    This definition uses CASH taxes paid (from the cash flow statement, not the
    income statement) and CASH interest paid (same) because these represent
    actual cash leaving the business, not accounting accruals. The denominator
    uses current portion of LTD (CPLTD) — the scheduled debt repayments due
    within the next 12 months — rather than total debt, because only the
    near-term maturities create an immediate cash obligation.

    Returns None if any required input is unavailable — FCC cannot be
    estimated if any component is missing.
    """
    # Normalise all outflows to positive magnitudes
    capex_mag = abs_outflow(capex)
    taxes_mag = abs_outflow(cash_taxes)
    int_mag   = abs_outflow(cash_interest)

    # All five components are required — return None if any is missing
    if ebitda is None or capex_mag is None or taxes_mag is None or cpltd is None or int_mag is None:
        return None

    # Numerator: EBITDA minus what the company must spend to operate and pay taxes
    numerator = float(ebitda) - capex_mag - taxes_mag

    # Denominator: scheduled debt principal repayments + interest on all debt
    denominator = float(cpltd) + int_mag

    return safe_divide(numerator, denominator)


def compute_leverage(total_debt: Optional[float], ebitda: Optional[float]) -> Optional[float]:
    """
    Computes the leverage ratio (Total Debt / EBITDA).

    HOW TO INTERPRET LEVERAGE:
      < 2.0x  → Low leverage, conservative capital structure
      2-4x    → Moderate leverage, typical for investment-grade companies
      4-6x    → High leverage, common in leveraged buyouts
      > 6x    → Very high leverage, heightened default risk

    WHY EBITDA IN THE DENOMINATOR:
    Lenders use EBITDA as the denominator because it approximates the annual
    cash generation capacity of the business before financing costs. Dividing
    total debt by this gives an intuitive "how many years of earnings would it
    take to pay off all the debt" measure.

    Returns None if EBITDA is zero or negative — leverage is undefined when
    a company has no positive earnings (or is losing money).
    """
    if total_debt is None or ebitda is None:
        return None
    # We can't compute leverage if EBITDA is zero or negative
    # (it would mean infinite or negative leverage, which is not meaningful)
    if ebitda <= 0:
        return None
    return float(total_debt) / float(ebitda)


def compute_ebitda_margin(ebitda: Optional[float], revenue: Optional[float]) -> Optional[float]:
    """
    Computes EBITDA margin as a percentage of revenue.

    FORMULA: EBITDA Margin = EBITDA / Revenue

    HOW TO INTERPRET:
    EBITDA margin tells you how much of every dollar of revenue becomes
    EBITDA. A 20% margin means the company keeps $0.20 as EBITDA for
    every $1.00 of revenue. Higher margins indicate more efficient operations
    and more cushion to absorb cost pressures.

    Used in the memo to compare profitability across years and against industry peers.
    """
    if ebitda is None or revenue is None:
        return None
    if revenue == 0:
        return None
    return float(ebitda) / float(revenue)
