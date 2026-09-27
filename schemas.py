"""
schemas.py
==========
Data shapes (schemas) used throughout the Credit AI platform.

This file defines what every piece of data looks like as it flows through
the system. Think of each class here as a "contract" — every function that
produces or consumes financial data agrees to use these shapes.

WHY PYDANTIC:
These classes use Pydantic (a Python data validation library). Pydantic
automatically checks that values are the right type, fills in defaults,
and converts JSON from the API into Python objects. If GPT-4o returns
data that doesn't match these schemas, Pydantic raises an error before
it can cause incorrect calculations downstream.

DATA FLOW THROUGH THE SYSTEM:
  Filing (PDF/HTML)
       ↓
  GPT-4o extraction → ExtractionResult (list of ExtractedPeriod objects)
       ↓
  validate_and_compute() → fills in derived_metrics on each period
       ↓
  generate_underwriting_memo() → uses BorrowerProfile + CovenantSet + ExtractionResult
       ↓
  API response → StructuredMetrics (organised for the frontend chart)
"""

from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Literal, Union

# A metric value can be a number, boolean, string, or missing (None)
MetricValue = Union[float, int, bool, str, None]


class BorrowerProfile(BaseModel):
    """
    Information about the company being analysed, provided by the analyst.

    These fields are entered in the frontend form and used to personalise
    the underwriting memo. Without them, the system still runs but generates
    a generic financial summary rather than a full credit memo.
    """
    name:            str = Field(..., description="Borrower legal name")
    industry:        str = Field(..., description="Primary industry")
    facility_type:   str = Field(..., description="Term loan, revolver, ABL, etc.")
    use_of_proceeds: str = Field(..., description="Working capital, refinance, acquisition, etc.")


class CovenantSet(BaseModel):
    """
    Financial covenant thresholds for the proposed credit facility.

    Covenants are contractual limits that the borrower must stay within.
    If the borrower exceeds these thresholds, they are in technical default
    and the lender can accelerate the loan.

    The system compares the extracted metrics against these thresholds
    and flags any covenant breaches in the underwriting memo.

    All fields are optional — many analyses are run without covenant data,
    especially for preliminary assessments.
    """
    max_total_leverage: Optional[float] = None  # e.g. 4.5 means debt cannot exceed 4.5x EBITDA
    min_fcc:            Optional[float] = None  # e.g. 1.15 means FCC must stay above 1.15x
    min_dscr:           Optional[float] = None  # Debt Service Coverage Ratio minimum (if used)


class ExtractedPeriod(BaseModel):
    """
    All financial data for a single fiscal year.

    One ExtractedPeriod = one year of data. A typical analysis covers
    three fiscal years (e.g. FY2022, FY2023, FY2024), so ExtractionResult
    usually contains a list of three ExtractedPeriod objects.

    Fields:
        period_name:      Label for this fiscal year, e.g. "FY2024" or "LTM"
        income_statement: Revenue, EBITDA, operating income, interest expense, etc.
        balance_sheet:    Cash, total assets, total debt, revolver data, etc.
        cash_flow:        CFO, capex, cash taxes paid, cash interest paid, etc.
        derived_metrics:  Calculated ratios like leverage, FCC, Altman Z-Score.
                          These are computed by validate_and_compute() in agents.py
                          AFTER GPT-4o returns the raw financials.
        notes:            Internal debugging notes including where each value
                          was found in the document (used by Review & Correct).
    """
    period_name:      str
    income_statement: Dict[str, Optional[float]] = Field(default_factory=dict)
    balance_sheet:    Dict[str, Optional[float]] = Field(default_factory=dict)
    cash_flow:        Dict[str, Optional[float]] = Field(default_factory=dict)
    derived_metrics:  Dict[str, MetricValue]     = Field(default_factory=dict)
    notes:            List[str]                  = Field(default_factory=list)


class ExtractionResult(BaseModel):
    """
    The complete output of GPT-4o's financial extraction for one filing.

    This is the central data object in the system. It is:
      - Produced by extract_financials_from_text() in agents.py
      - Enriched by validate_and_compute() which adds derived_metrics
      - Stored as JSON in the SQLite database for caching
      - Returned to the frontend as part of the API response
      - Used by the export functions (Excel, Word) to build documents

    Fields:
        statement_basis: Whether numbers are in "actual" dollars, "thousands",
                         or "millions". GPT-4o detects this from the filing.
                         All downstream calculations assume numbers are in
                         this unit consistently.
        periods:         List of fiscal years, most recent first.
        validation_flags: Warning messages about data quality, missing fields,
                         covenant breaches, or profile corrections applied.
                         Displayed as yellow warning banners in the frontend.
        schema_version:  Version of this schema (for backward compatibility
                         when loading cached results after schema changes).
        extractor_version: Date-stamped version of the extraction prompt.
    """
    statement_basis:   Literal["actual", "thousands", "millions"] = "actual"
    periods:           List[ExtractedPeriod] = Field(default_factory=list)
    validation_flags:  List[str]             = Field(default_factory=list)
    schema_version:    str                   = "1.0"
    extractor_version: str                   = "2026-01-25"


class MemoRequest(BaseModel):
    """
    Input to the underwriting memo generation endpoint.
    Combines the borrower context, covenant thresholds, and extracted financials.
    """
    borrower:  BorrowerProfile
    covenants: CovenantSet
    extracted: ExtractionResult


class MemoResponse(BaseModel):
    """The underwriting memo returned as a Markdown-formatted string."""
    memo_markdown: str


# =============================================================================
# Structured Metrics — organised for the frontend charts and display
# =============================================================================
# The ExtractionResult above stores data in a format optimised for the
# extraction pipeline (nested dicts by section). The classes below reorganise
# that data into a format optimised for the frontend — one object per period
# with all metrics at the top level, plus pre-computed year-over-year comparisons.

class MetricComparison(BaseModel):
    """
    A year-over-year comparison for a single metric.

    For example: "Revenue grew from $5.2B in FY2023 to $5.8B in FY2024,
    an increase of $0.6B (+11.5%)."

    The frontend uses these to render the YoY change indicators
    (green arrows for improvements, red for deterioration).
    """
    metric_name:   str
    current_period: str
    current_value:  Optional[float] = None
    prior_period:   Optional[str]   = None
    prior_value:    Optional[float] = None
    change_amount:  Optional[float] = None  # current_value - prior_value
    change_pct:     Optional[float] = None  # (change_amount / prior_value)
    narrative:      Optional[str]   = None  # Plain-English description of the change
    unit:           Optional[str]   = None  # "MM", "K", "x", "%" for display formatting


class PeriodMetrics(BaseModel):
    """
    All key metrics for a single fiscal year, flattened into one object.

    Unlike ExtractedPeriod which organises by section (income statement,
    balance sheet, cash flow), this class has all metrics at the top level
    for easy access in the frontend without nested dict lookups.
    """
    period_name:        str

    # Income statement metrics
    revenue:            Optional[float] = None
    cost_of_sales:      Optional[float] = None
    sga_expense:        Optional[float] = None
    operating_income:   Optional[float] = None
    ebitda:             Optional[float] = None
    ebitda_margin:      Optional[float] = None  # EBITDA / Revenue (as a decimal, e.g. 0.22 = 22%)
    net_income:         Optional[float] = None

    # Balance sheet and leverage metrics
    total_debt:         Optional[float] = None
    leverage:           Optional[float] = None  # Total Debt / EBITDA (in "x" turns)
    cash:               Optional[float] = None

    # Cash flow metrics
    cfo:                Optional[float] = None
    capex:              Optional[float] = None
    free_cash_flow:     Optional[float] = None
    fcc:                Optional[float] = None  # Fixed Charge Coverage (in "x" turns)

    # Revolver / credit facility data
    revolver_facility_size: Optional[float] = None  # Total committed facility size
    revolver_borrowings:    Optional[float] = None  # Amount currently drawn
    revolver_availability:  Optional[float] = None  # Undrawn capacity remaining

    # Credit risk scores
    altman_z_score: Optional[float] = None  # > 2.99 safe, 1.81-2.99 grey zone, < 1.81 distress
    pd_score:       Optional[int]   = None  # 1-12 scale: 1 = lowest risk, 12 = highest risk
    credit_rating:  Optional[str]   = None  # e.g. "Investment Grade", "Speculative"


class StructuredMetrics(BaseModel):
    """
    All metrics organised for the frontend, covering all fiscal years.

    Returned in the API response alongside the memo and raw extracted data.
    The frontend uses this to render the metrics panel, trend charts,
    and year-over-year comparison table without needing to parse the
    raw ExtractedPeriod objects.
    """
    statement_basis:  str                        # Unit: "actual", "thousands", or "millions"
    periods:          List[PeriodMetrics]        = Field(default_factory=list)
    yoy_comparisons:  List[MetricComparison]     = Field(default_factory=list)


# =============================================================================
# API Response Schemas
# =============================================================================

class UnderwriteCreateResponse(BaseModel):
    """
    The response returned after a successful /v1/analyze call.

    Contains everything the frontend needs to display the results:
      - run_id: unique ID for this analysis (used for export calls)
      - memo_markdown: the full underwriting memo as formatted text
      - structured_metrics: pre-organised metrics for charts and tables
      - extracted: the raw extracted data (used by Review & Correct)
      - validation_flags: warnings about data quality or profile conflicts
      - completeness: score from 0-1 indicating how complete the extraction was
    """
    run_id:             str
    build:              Optional[str]              = None
    status:             Optional[Literal["queued", "running", "completed", "failed"]] = None
    completeness:       Optional[float]            = None
    validation_flags:   Optional[List[str]]        = None
    memo_markdown:      Optional[str]              = None
    structured_metrics: Optional[StructuredMetrics] = None
    extracted:          Optional[ExtractionResult] = None
    mda_summary:        Optional[str]              = None
    cache_bypass:       Optional[bool]             = None
    excerpt_preview:    Optional[str]              = None
    model_raw_preview:  Optional[str]              = None


class UnderwriteStatusResponse(BaseModel):
    """
    Status response for polling a long-running analysis.
    Contains the same fields as UnderwriteCreateResponse once completed.
    """
    run_id:             str
    status:             Literal["queued", "running", "completed", "failed"]
    error:              Optional[str]              = None
    extracted:          Optional[ExtractionResult] = None
    memo_markdown:      Optional[str]              = None
    structured_metrics: Optional[StructuredMetrics] = None
    completeness:       Optional[float]            = None
    validation_flags:   Optional[List[str]]        = None
