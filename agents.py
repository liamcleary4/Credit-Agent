from __future__ import annotations

import os
import re
import json
import html
import time
import logging
from typing import Tuple, Optional

from openai import OpenAI
import anthropic

from schemas import ExtractionResult, BorrowerProfile, CovenantSet

from metrics import (
    compute_ebitda,
    compute_free_cash_flow,
    compute_fcc,
    compute_leverage,
    compute_ebitda_margin,
    abs_outflow,
)

from credit_scoring import (
    compute_altman_z_score,
    compute_pd_score,
)

logger = logging.getLogger("credit_ai")

# Company extraction profiles (lazy import to avoid circular deps)
try:
    import profiles as _profile_store
    _PROFILES_AVAILABLE = True
except ImportError:
    _PROFILES_AVAILABLE = False
    logger.warning("profiles.py not found — profile system disabled")

_client = None
_anthropic_client = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is missing.")
        _client = OpenAI()
    return _client


def get_anthropic_client() -> anthropic.Anthropic:
    global _anthropic_client
    if _anthropic_client is None:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY is missing.")
        _anthropic_client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    return _anthropic_client


# ============================================================
# RATE LIMIT HELPER
# ============================================================

def _call_claude_with_retry(client, **kwargs) -> str:
    """Call Claude with exponential backoff on rate limit (3 attempts: 0s, 60s, 120s)."""
    delays = [0, 60, 120]
    last_error = None
    for attempt, delay in enumerate(delays):
        if delay > 0:
            logger.warning(f"Rate limit hit — waiting {delay}s before retry {attempt+1}/3")
            time.sleep(delay)
        try:
            resp = client.messages.create(**kwargs)
            return resp.content[0].text if resp.content else ""
        except anthropic.RateLimitError as e:
            last_error = e
            if attempt == len(delays) - 1:
                raise
    raise last_error


# ============================================================
# HTML STRIPPING
# ============================================================

# ── HTML stripping ────────────────────────────────────────────────────────────
# 10-K filings from EDGAR arrive as HTML with thousands of tags, inline XBRL
# annotations, and JavaScript. This function strips all markup and converts
# tables to pipe-delimited rows that GPT-4o can read as financial data.
#
# Table structure is preserved: </tr> becomes a newline, </td> becomes a tab.
# This means "Net sales | 5,200 | 4,800" reads as a three-column row with
# revenue figures for two fiscal years — critical for multi-period extraction.
#
# HTML entities (&nbsp;, &#160;, &#8212; etc.) are decoded to Unicode.
# Output is typically 80-90% smaller than the input HTML.
def _strip_html_to_text(raw: str) -> str:def _strip_html_to_text(raw: str) -> str:
    if not raw or not raw.strip():
        return raw
    if "<html" not in raw.lower() and "<span" not in raw.lower():
        return raw
    logger.info(f"HTML stripping: input {len(raw):,} characters")
    text = raw
    text = re.sub(r'<head[\s\S]*?</head>', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'<ix:header[\s\S]*?</ix:header>', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'<script[\s\S]*?</script>', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'<style[\s\S]*?</style>', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'</tr\s*>', '\n', text, flags=re.IGNORECASE)
    text = re.sub(r'</td\s*>|</th\s*>', '\t', text, flags=re.IGNORECASE)
    block_tags = r'</?(p|div|section|article|header|footer|h[1-6]|li|ul|ol|blockquote|pre)'
    text = re.sub(block_tags + r'[^>]*>', '\n', text, flags=re.IGNORECASE)
    text = re.sub(r'<br\s*/?>', '\n', text, flags=re.IGNORECASE)
    text = re.sub(r'<[^>]+>', ' ', text)
    entity_map = {
        '&#8217;': "'", '&#8216;': "'", '&#8220;': '"', '&#8221;': '"',
        '&#8211;': '\u2013', '&#8212;': '\u2014', '&#8203;': '', '&#160;': ' ',
        '&nbsp;': ' ', '&amp;': '&', '&lt;': '<', '&gt;': '>',
        '&quot;': '"', '&#x2019;': "'", '&#x2018;': "'",
        '&#x201C;': '"', '&#x201D;': '"', '&#x2013;': '\u2013', '&#x2014;': '\u2014',
    }
    for entity, replacement in entity_map.items():
        text = text.replace(entity, replacement)
    text = html.unescape(text)
    text = text.replace('\t', ' ')
    text = re.sub(r'[ ]{2,}', ' ', text)
    lines = [line.rstrip() for line in text.splitlines()]
    clean_lines: list[str] = []
    blank_count = 0
    for line in lines:
        if line.strip() == '':
            blank_count += 1
            if blank_count <= 2:
                clean_lines.append('')
        else:
            blank_count = 0
            clean_lines.append(line)
    text = '\n'.join(clean_lines).strip()
    logger.info(f"HTML stripping: output {len(text):,} characters")
    return text


# ============================================================
# SCHEMA CONSTANTS
# ============================================================

EXPECTED_INCOME_KEYS = [
    "revenue", "cost_of_sales", "sga_expense", "operating_income", "ebitda",
    "net_income", "interest_expense", "income_tax_expense",
    "depreciation_amortization", "rent_expense",
]
EXPECTED_BS_KEYS = [
    "cash", "total_assets", "total_liabilities", "total_equity", "total_debt",
    "long_term_debt", "current_portion_long_term_debt",
    "revolver_facility_size", "revolver_borrowings", "revolver_availability",
]
EXPECTED_CF_KEYS = [
    "cfo", "cfi", "cff", "capex", "cash_paid_for_interest",
    "cash_paid_for_income_taxes", "dividends_distributions_paid",
]

# ── GPT-4o extraction prompt ─────────────────────────────────────────────────
# The system prompt contains hard rules refined from systematic extraction failures:
#
# MULTI-PERIOD RULE: GPT-4o was returning only one period when filings showed
# three years of columns. The prompt now explicitly explains pipe-separated tables
# and requires one period object per fiscal year column shown.
#
# CASH TAXES RULE: Companies report taxes twice — on the income statement (accrual,
# not cash) and in supplemental cash flow disclosures (actual cash paid). GPT-4o
# kept picking up the income statement figure. The prompt now explicitly requires
# the supplemental cash flow total only.
#
# ZERO DEFAULT RULE: current_portion_long_term_debt and revolver_borrowings should
# be 0 (not null) when absent — a missing line means the value is genuinely zero,
# not that we couldn't find it. All other fields default to null when not found.
#
# The schema prompt defines the exact JSON structure GPT-4o must return.
# _coerce_json and _normalize_payload fix malformed responses and enforce types.
EXTRACTOR_SYSTEM_PROMPT = """EXTRACTOR_SYSTEM_PROMPT = """
You are a financial statement extraction assistant for a commercial credit analyst.
Return ONLY valid JSON. No extra commentary.

Hard rules:
- MULTI-PERIOD: When a filing shows columns for multiple years (2024, 2023, 2022),
  create a SEPARATE period object for EACH year. Never merge years.
- Look for pipe-separated rows like: "Net sales | 5,200 | 4,800 | 4,500"
  This means: FY2024=5200, FY2023=4800, FY2022=4500 — three separate period objects.
- Each numeric field in each period object MUST be a single scalar number or null.
- NEVER put multiple years or commentary into a single numeric field.
- Do not compute derived metrics; derived_metrics must be {}.
- Do not compute EBITDA; leave ebitda null unless explicitly labeled as EBITDA.
- Prioritize: operating_income, depreciation_amortization, total_debt,
  current_portion_long_term_debt, cash_paid_for_interest,
  cash_paid_for_income_taxes, capex, cfo.


CASH TAXES DISAMBIGUATION RULE (critical):
- cash_paid_for_income_taxes MUST come from the Cash Flow Statement supplemental
  disclosures — a single total line such as:
  "Cash paid for income taxes", "Income taxes paid", "Cash payments for income taxes",
  "Cash income tax payments", or "Taxes paid".
- Do NOT use values from tax footnotes that break taxes down by jurisdiction
  (Federal, State, International, China, India, Korea etc.).
  Those are components — not the cash paid total from the cash flow statement.
- If you see both a jurisdictional breakdown AND a supplemental cash flow line,
  ALWAYS use the supplemental cash flow line.

CASH INTEREST DISAMBIGUATION RULE:
- cash_paid_for_interest MUST come from the supplemental cash flow disclosures —
  a single total line such as "Cash paid for interest" or "Interest paid".
- Do NOT use interest expense from the income statement.

ZERO DEFAULT RULE (only for specific fields):
- current_portion_long_term_debt: If this line item does not appear anywhere in the
  balance sheet or debt schedule, set it to 0 (not null). Companies with no
  near-term debt maturities simply omit this line — the correct value is 0.
- revolver_borrowings: If a revolving credit facility exists but shows no borrowings
  outstanding, set it to 0 (not null). An omitted line means nothing drawn.
- All other fields: leave as null if not found. Never default revenue, operating
  income, total debt, capex, or other fields to 0.
"""

EXTRACTOR_SCHEMA_PROMPT = """
Return JSON with this exact structure:
{
  "statement_basis": "actual" | "thousands" | "millions",
  "periods": [
    {
      "period_name": "FY2024",
      "income_statement": {
        "revenue": number|null, "operating_income": number|null, "ebitda": number|null,
        "cost_of_sales": number|null, "sga_expense": number|null, "net_income": number|null,
        "interest_expense": number|null, "income_tax_expense": number|null,
        "depreciation_amortization": number|null, "rent_expense": number|null
      },
      "balance_sheet": {
        "cash": number|null, "total_assets": number|null, "total_liabilities": number|null,
        "total_equity": number|null, "total_debt": number|null, "long_term_debt": number|null,
        "current_portion_long_term_debt": number|null, "revolver_facility_size": number|null,
        "revolver_borrowings": number|null, "revolver_availability": number|null
      },
      "cash_flow": {
        "cfo": number|null, "cfi": number|null, "cff": number|null, "capex": number|null,
        "cash_paid_for_interest": number|null, "cash_paid_for_income_taxes": number|null,
        "dividends_distributions_paid": number|null
      },
      "derived_metrics": {},
      "notes": []
    }
  ],
  "validation_flags": []
}
"""


# ============================================================
# PARSING HELPERS
# ============================================================

def _coerce_json(text: str) -> dict:
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.lstrip("`").strip()
        if text.lower().startswith("json"):
            text = text[4:].lstrip("\n").strip()
        if text.endswith("```"):
            text = text[:-3].strip()
    return json.loads(text)


def _to_float(value):
    if value is None: return None
    if isinstance(value, (int, float)): return float(value)
    if not isinstance(value, str): return None
    s = value.strip()
    if "|" in s or "\n" in s: return None
    if len(set(re.findall(r"\b20\d{2}\b", s))) >= 2: return None
    s = s.replace("$", "").replace("€", "").replace("£", "")
    s = re.sub(r"\[\d+\]$", "", s).strip()
    neg = False
    if s.startswith("(") and s.endswith(")"):
        neg = True
        s = s[1:-1].strip()
    s = s.replace(",", "")
    if not re.fullmatch(r"-?\d+(\.\d+)?", s): return None
    return -float(s) if neg else float(s)


def _clean_statement_dict(d: dict, expected_keys: list) -> dict:
    return {k: (_to_float(d.get(k)) if isinstance(d, dict) else None) for k in expected_keys}


def _normalize_payload(data: dict) -> dict:
    if not isinstance(data, dict):
        return {"statement_basis": "actual", "periods": [], "validation_flags": []}
    sb = data.get("statement_basis")
    if sb not in ("actual", "thousands", "millions"): sb = "actual"
    cleaned_periods = []
    for i, p in enumerate((data.get("periods") or [])[:5]):
        if not isinstance(p, dict): continue
        cleaned_periods.append({
            "period_name": str(p.get("period_name") or f"PERIOD_{i+1}"),
            "income_statement": _clean_statement_dict(p.get("income_statement") or {}, EXPECTED_INCOME_KEYS),
            "balance_sheet": _clean_statement_dict(p.get("balance_sheet") or {}, EXPECTED_BS_KEYS),
            "cash_flow": _clean_statement_dict(p.get("cash_flow") or {}, EXPECTED_CF_KEYS),
            "derived_metrics": {},
            "notes": p.get("notes") if isinstance(p.get("notes"), list) else [],
        })

    # Zero-default post-processing:
    # current_portion_long_term_debt and revolver_borrowings should be 0 not null
    # when absent — an omitted line in these cases means the value is genuinely zero.
    for p in cleaned_periods:
        bs = p.get("balance_sheet", {})
        if bs.get("current_portion_long_term_debt") is None:
            bs["current_portion_long_term_debt"] = 0.0
        if bs.get("revolver_borrowings") is None:
            bs["revolver_borrowings"] = 0.0

    return {
        "statement_basis": sb,
        "periods": cleaned_periods,
        "validation_flags": data.get("validation_flags") if isinstance(data.get("validation_flags"), list) else [],
        "schema_version": "1.0",
        "extractor_version": "2026-01-27",
    }


# ============================================================
# MD&A SECTION EXTRACTION (TOC-aware)
# ============================================================

# ── MD&A section extraction ───────────────────────────────────────────────────
# The MD&A (Management's Discussion & Analysis) is Item 7 of a 10-K.
# It explains WHY the numbers changed — which segments grew, what drove margins,
# what headwinds were offset by tailwinds. This is what makes the output
# different from any data terminal.
#
# Finding Item 7 is harder than it sounds:
#   - The Table of Contents also contains "ITEM 7. MANAGEMENT'S DISCUSSION..."
#     and must be skipped to find the actual section
#   - Different companies format the header differently
#
# Search strategy (three passes):
#   Pass 1: Standard "ITEM 7." patterns with TOC detection
#   Pass 2: Broader patterns like "RESULTS OF OPERATIONS"
#   Pass 3: Last 40% of document as a last-resort fallback
#
# TOC detection: if we find "ITEM 7." but the surrounding 2000 chars contain
# 5+ other "ITEM X." references, it's the Table of Contents — skip it.
#
# After finding the start, the preamble (boilerplate before actual results)
# is trimmed by advancing to the first line containing financial signal words
# like "NET SALES" or "REVENUE DECREASED".
def _extract_mda_section(raw_text: str, max_chars: int = 350000) -> str:def _extract_mda_section(raw_text: str, max_chars: int = 350000) -> str:
    if not raw_text: return ""
    upper = raw_text.upper()
    doc_len = len(raw_text)

    MDA_SIGNALS = [
        "NET SALES", "NET REVENUES", "TOTAL REVENUES", "REVENUE", "GROSS MARGIN",
        "OPERATING INCOME", "RESULTS OF OPERATIONS", "COMPARED TO", "YEAR ENDED",
        "INCREASED", "DECREASED", "COST OF GOODS", "COST OF SALES", "SELLING GENERAL",
        "SEGMENT", "FAVORABLE", "UNFAVORABLE", "PARTIALLY OFFSET", "MILLION", "BILLION",
    ]
    TOC_SIGNALS = [
        "ITEM 7A.", "ITEM 8.", "ITEM 9.", "ITEM 10.",
        "QUANTITATIVE AND QUALITATIVE", "FINANCIAL STATEMENTS AND SUPPLEMENTARY",
        "CHANGES IN AND DISAGREEMENTS", "CONTROLS AND PROCEDURES",
    ]
    pass1_markers = [
        "ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION AND RESULTS OF OPERATIONS",
        "ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS",
        "ITEM 7 - MANAGEMENT'S DISCUSSION AND ANALYSIS",
        "ITEM 7\u2014MANAGEMENT'S DISCUSSION AND ANALYSIS",
        "ITEM 7. MANAGEMENT DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION AND RESULTS OF OPERATIONS",
        "ITEM 7. MANAGEMENT DISCUSSION AND ANALYSIS",
        "MANAGEMENT'S DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION AND RESULTS OF OPERATIONS",
        "MANAGEMENT'S DISCUSSION AND ANALYSIS",
        "MANAGEMENT DISCUSSION AND ANALYSIS",
        "MD&A",
    ]
    pass2_markers = [
        "RESULTS OF OPERATIONS", "CONSOLIDATED RESULTS OF OPERATIONS",
        "OPERATING RESULTS", "DISCUSSION OF OPERATIONS",
        "FINANCIAL REVIEW", "FINANCIAL AND OPERATING REVIEW",
        "OPERATING AND FINANCIAL REVIEW", "FINANCIAL OVERVIEW",
        "EXECUTIVE OVERVIEW", "REVIEW OF FINANCIAL RESULTS",
        "REVIEW OF OPERATIONS", "FINANCIAL DISCUSSION",
    ]

    def _is_toc(idx: int) -> bool:
        w = upper[idx: idx + 2000]
        return len(re.findall(r'\bITEM\s+\d+[A-Z]?[\.\s]', w)) >= 5 or sum(1 for s in TOC_SIGNALS if s in w) >= 3

    def _quality_search(markers, window=15000):
        for marker in markers:
            pos = 0
            while True:
                idx = upper.find(marker, pos)
                if idx == -1: break
                if _is_toc(idx):
                    logger.info(f"Skipping TOC hit: '{marker[:50]}' at {idx:,}")
                    pos = idx + 1
                    continue
                w = upper[idx: idx + window]
                if sum(1 for s in MDA_SIGNALS if s in w) >= 3 and len(w.replace(" ","").replace("\n","")) > 500:
                    logger.info(f"MD&A hit: '{marker[:50]}' at {idx:,}")
                    return idx, marker
                pos = idx + 1
        return -1, None

    start_idx, marker_used = _quality_search(pass1_markers)
    if start_idx == -1:
        logger.warning("Pass 1 empty — trying pass 2 markers")
        start_idx, marker_used = _quality_search(pass2_markers)
    if start_idx == -1:
        for m in pass1_markers + pass2_markers:
            idx = upper.rfind(m)
            if idx != -1 and not _is_toc(idx) and len(raw_text[idx:idx+500].strip()) > 100:
                start_idx, marker_used = idx, f"{m} (rfind)"
                break
    if start_idx == -1:
        start_idx = max(0, int(doc_len * 0.60))
        marker_used = "last-40%-fallback"
        logger.warning(f"No MD&A marker — fallback at {start_idx:,}")

    logger.info(f"MD&A start: '{marker_used[:60]}' at {start_idx:,}")

    end_idx = doc_len
    for m in ["ITEM 7A.", "ITEM 7A ", "ITEM 7A\u2014", "ITEM 7A-",
              "ITEM 8.", "ITEM 8 ", "ITEM 8\u2014", "ITEM 8-"]:
        idx = upper.find(m, start_idx + 5000)
        if idx != -1 and idx < end_idx:
            end_idx = idx
            logger.info(f"MD&A end: '{m}' at {idx:,}")
            break

    mda_text = raw_text[start_idx:end_idx]
    logger.info(f"MD&A extracted: {len(mda_text):,} chars")

    # Trim leading preamble — advance to first financial results content
    RESULTS_SIGNALS = [
        "NET SALES", "NET REVENUES", "TOTAL REVENUES", "REVENUES WERE",
        "REVENUES DECREASED", "REVENUES INCREASED", "THE FOLLOWING TABLE",
        "YEAR ENDED DECEMBER", "COMPARED TO THE PRIOR YEAR",
    ]
    tu = mda_text.upper()
    earliest = min((tu.find(s) for s in RESULTS_SIGNALS if tu.find(s) != -1), default=len(mda_text))
    if earliest > 2000:
        logger.info(f"MD&A trimmed: skipping first {earliest:,} chars of preamble")
        mda_text = mda_text[earliest:]

    if len(mda_text) > max_chars:
        logger.warning(f"MD&A truncated from {len(mda_text):,} to {max_chars:,} chars")

    if any(kw in mda_text.upper() for kw in ["SEGMENT", "BUSINESS UNIT", "DIVISION"]):
        logger.info("Segment content confirmed in MD&A.")

    logger.info(f"MD&A first 300 chars: {mda_text[:300].replace(chr(10),' ')}")
    return mda_text[:max_chars]


# ============================================================
# MD&A SUMMARISATION — Claude with retry
# ============================================================

# ── MD&A analysis with Claude Sonnet ─────────────────────────────────────────
# After extracting the MD&A text, this function sends it to Claude Sonnet
# for a structured segment-by-segment performance analysis.
#
# WHY CLAUDE INSTEAD OF GPT-4O:
# Claude produces more coherent analyst-quality prose for narrative analysis
# and handles the long MD&A text (up to 80,000 chars) better than GPT-4o.
#
# The prompt asks Claude to:
#   1. Identify every business segment mentioned
#   2. For each segment: extract revenue, COGS, gross margin YoY
#   3. Identify specific drivers (volume, price, mix, new products, geography)
#   4. Identify offsetting factors
#   5. Write cross-segment themes
#
# Output is formatted into structured markdown by _format_mda_summary
# and displayed in the MD&A Analysis tab in the frontend.
#
# Rate limiting: MD&A text is capped at 80,000 chars (MDA_CHAR_LIMIT env var).
# Retry logic handles transient rate limit errors with exponential backoff
# (0s, 60s, 120s between attempts — max 3 attempts total).
def _summarize_mda_for_drivers(mda_text: str, extracted: ExtractionResult) -> str:def _summarize_mda_for_drivers(mda_text: str, extracted: ExtractionResult) -> str:
    if not mda_text or not mda_text.strip():
        return "No MD&A section found."
    if not extracted.periods:
        return "No periods extracted."

    p0 = extracted.periods[0]
    p1 = extracted.periods[1] if len(extracted.periods) >= 2 else None
    current_label = p0.period_name
    prior_label = p1.period_name if p1 else "the prior year"

    # Reduce char limit to stay under 30k token/min rate limit
    MDA_CHAR_LIMIT = int(os.environ.get("MDA_CHAR_LIMIT", "80000"))
    mda_for_prompt = mda_text[:MDA_CHAR_LIMIT]
    if len(mda_text) > MDA_CHAR_LIMIT:
        logger.warning(f"MD&A truncated to {MDA_CHAR_LIMIT:,} chars for rate limit compliance")

    logger.info(f"MD&A summarisation: current={current_label} prior={prior_label} sending={len(mda_for_prompt):,} chars")

    system_prompt = (
        "You are a senior financial analyst. Read SEC 10-K MD&A sections and produce "
        "structured commentary on why each business segment performed the way it did. "
        "Always cite exact dollar amounts from the filing. Never use placeholder text. "
        "If a figure is not disclosed, write 'Not disclosed in filing'."
    )

    user_prompt = f"""Read the MD&A section below and produce a structured performance analysis
comparing {prior_label} to {current_label}.

MD&A TEXT ({len(mda_for_prompt):,} characters):
{mda_for_prompt}

INSTRUCTIONS:

Step 1 — List every business segment discussed:
SEGMENTS IDENTIFIED: [name 1], [name 2], ...

Step 2 — CONSOLIDATED SUMMARY:

REVENUE:
Total: [actual] vs [prior] (% YoY)
Key themes: [2-3 sentences with actual figures]

COST OF GOODS SOLD:
Total: [actual] vs [prior] (% YoY)
Key themes: [2-3 sentences with actual figures]

GROSS MARGIN:
Total: [actual] vs [prior]
Margin %: [actual]% vs [prior]%
Key themes: [2-3 sentences with actual figures]

Step 3 — For each segment:

SEGMENT: [Exact name]

REVENUE:
Amount: [actual] vs [prior] (change of [delta])
Drivers:
- [Driver]: [positive/negative] impact of approximately [dollar amount]
Offsetting factors:
- [Factor]: approximately [dollar amount]

COST OF GOODS SOLD:
Amount: [actual] vs [prior] (or: Not separately disclosed.)
Drivers:
- [Factor]: approximately [amount]
Offsetting factors:
- [Factor]: approximately [amount]

GROSS MARGIN:
Amount: [actual] vs [prior] (or: Not separately disclosed.)
Margin %: [actual]% vs [prior]% (or: Not separately disclosed.)
Drivers:
- [Factor]: approximately [amount]
Offsetting factors:
- [Factor]: approximately [amount]

Step 4 — KEY CROSS-SEGMENT THEMES
3-5 sentences using actual figures.

Rules: use exact dollar amounts, no placeholder text."""

    try:
        client = get_anthropic_client()
        raw = _call_claude_with_retry(
            client,
            model="claude-sonnet-4-6",
            max_tokens=8000,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        refusal = ["i'm sorry", "i cannot", "i can't assist", "unable to help"]
        if any(s in raw.lower()[:200] for s in refusal):
            logger.error(f"Claude refusal: {raw[:200]}")
            return f"MD&A extraction failed: {raw[:200]}"
        logger.info(f"MD&A summary: {len(raw):,} chars")
        return _format_mda_summary(raw)
    except Exception as e:
        logger.error(f"Error in _summarize_mda_for_drivers: {e}")
        return f"Error extracting MD&A insights: {str(e)}"


# ============================================================
# MD&A FORMATTER
# ============================================================

def _format_mda_summary(raw: str) -> str:
    if not raw or not raw.strip(): return raw
    lines = raw.split('\n')
    out = []

    def _blank_before():
        if out and out[-1] != '': out.append('')

    def _blank_after():
        out.append('')

    for line in lines:
        stripped = line.strip()
        clean = stripped.strip('*').strip()
        if not stripped:
            if out and out[-1] != '': out.append('')
            continue
        if clean.upper().startswith('SEGMENTS IDENTIFIED:'):
            _blank_before(); out.append(f'## {clean}'); _blank_after()
        elif re.match(r'^Step\s+\d+\s*[—\-]', clean, re.IGNORECASE):
            pass  # skip step wrappers
        elif clean.upper() in ('CONSOLIDATED SUMMARY:', 'CONSOLIDATED SUMMARY',
                                'CONSOLIDATED PERFORMANCE SUMMARY:', 'CONSOLIDATED PERFORMANCE SUMMARY') \
             or re.match(r'^STEP\s+2', clean.upper()):
            _blank_before(); out.append('## Consolidated Performance Summary'); _blank_after()
        elif 'KEY CROSS-SEGMENT THEMES' in clean.upper():
            _blank_before(); out.append('## Key Cross-Segment Themes'); _blank_after()
        elif re.match(r'^(STEP\s+3|INDIVIDUAL SEGMENT ANALYSIS)', clean.upper()):
            pass
        elif re.match(r'^SEGMENT\s*:', clean, re.IGNORECASE):
            name = re.sub(r'^SEGMENT\s*:\s*', '', clean, flags=re.IGNORECASE).strip().strip('*')
            _blank_before(); out.append(f'### Segment: {name}'); _blank_after()
        elif re.match(r'^(REVENUE|COST OF GOODS SOLD|GROSS MARGIN)\s*:?\s*$', clean, re.IGNORECASE):
            _blank_before(); out.append(f'#### {clean.rstrip(":").strip().title()}'); _blank_after()
        elif re.match(r'^(Amount|Total)\s*:', stripped, re.IGNORECASE):
            out.append(f'  - **{stripped}**')
        elif re.match(r'^Margin\s*%?\s*:', stripped, re.IGNORECASE):
            out.append(f'  - **{stripped}**')
        elif stripped.lower().startswith('key themes:'):
            out.append(f'  - {stripped}')
        elif re.match(r'^(Drivers|Offsetting factors.*|Volume and price.*)\s*:?\s*$', stripped, re.IGNORECASE):
            out.append(f'  - **{stripped.rstrip(":")}:**')
        elif stripped.startswith('- '):
            out.append(f'    - {stripped[2:]}')
        elif re.search(r'not (separately )?disclosed', stripped, re.IGNORECASE):
            out.append(f'  - *{stripped}*')
        else:
            out.append(f'  {stripped}')

    # Deduplicate blank lines
    final = []
    prev_blank = False
    for line in out:
        is_blank = (line == '')
        if is_blank and prev_blank: continue
        final.append(line)
        prev_blank = is_blank

    # Guarantee blank line after every header
    result = []
    for i, line in enumerate(final):
        result.append(line)
        if re.match(r'^#{2,4}\s', line):
            if i + 1 < len(final) and final[i + 1] != '':
                result.append('')

    return '\n'.join(result).strip()


# ============================================================
# MEMO FORMATTER
# ============================================================

def _format_memo_markdown(raw: str) -> str:
    if not raw or not raw.strip(): return raw
    lines = raw.split('\n')
    out = []
    FINANCIAL_HEADERS = {
        'revenue', 'cost of goods sold', 'gross margin',
        'ebitda and margins', 'sg&a', 'cash flow',
        'leverage and debt service', 'liquidity',
    }
    in_segment_block = False
    in_financial_section = False

    for line in lines:
        stripped = line.strip()
        if not stripped:
            if out and out[-1] != '': out.append('')
            in_segment_block = False
            continue
        if stripped.startswith('## '):
            if out and out[-1] != '': out.append('')
            out.append(stripped); out.append('')
            in_segment_block = False; in_financial_section = False
            continue
        if stripped.startswith('### '):
            sname = stripped[4:].lower().strip()
            in_financial_section = any(s in sname for s in FINANCIAL_HEADERS)
            if out and out[-1] != '': out.append('')
            out.append(stripped); out.append('')
            in_segment_block = False
            continue
        if stripped.startswith('#### '):
            sname = stripped[5:].lower().strip()
            in_segment_block = in_financial_section and not any(s in sname for s in FINANCIAL_HEADERS)
            if out and out[-1] != '': out.append('')
            out.append(stripped); out.append('')
            continue
        if stripped == '---':
            if out and out[-1] != '': out.append('')
            out.append('---'); out.append('')
            continue
        if stripped.startswith('- '):
            out.append(f'  - {stripped[2:]}' if in_segment_block else stripped)
            continue
        if re.match(r'^[0-9]+[.)][[:space:]]', stripped) or (len(stripped) > 1 and stripped[0].isdigit() and stripped[1] in '.)' and len(stripped) > 2 and stripped[2] == ' '):
            _parts = stripped.split(' ', 1)
            _clean = _parts[1] if len(_parts) > 1 else stripped
            out.append(f'- {_clean}')
            continue
        out.append(f'  {stripped}' if in_segment_block else stripped)

    # Deduplicate blanks
    final = []
    prev_blank = False
    for line in out:
        is_blank = (line == '')
        if is_blank and prev_blank: continue
        final.append(line)
        prev_blank = is_blank

    # Blank after every header
    result = []
    for i, line in enumerate(final):
        result.append(line)
        if re.match(r'^#{2,4}\s', line):
            if i + 1 < len(final) and final[i + 1] != '':
                result.append('')

    return '\n'.join(result).strip()


# ============================================================
# LEASE DATA EXTRACTION
# ============================================================

# ── Lease expense extraction ──────────────────────────────────────────────────
# Operating lease cost does NOT appear in the main income statement for most
# companies post-ASC 842 (2019 lease accounting standard). Instead, it lives
# in a "LEASES" footnote in the Notes to Financial Statements.
#
# This function:
#   1. Finds the Notes section (after the financial statements)
#   2. Finds the LEASES footnote
#   3. Calls GPT-4o-mini to extract the lease cost table
#
# WHY GPT-4O-MINI: The lease footnote is a simple structured table.
# GPT-4o-mini handles it well at ~$0.05 vs ~$0.50 for GPT-4o.
#
# The extracted operating_lease_cost is added back into EBITDA as rent expense,
# creating a lease-adjusted EBITDAR which is common in lease-heavy industries.
def extract_lease_data(raw_text: str, period_names: list[str]) -> dict:def extract_lease_data(raw_text: str, period_names: list[str]) -> dict:
    if not raw_text or not period_names: return {}
    upper = raw_text.upper()
    TOC_SIGNALS = ["ITEM 7A.", "ITEM 8.", "ITEM 9.", "ITEM 10.",
                   "QUANTITATIVE AND QUALITATIVE", "FINANCIAL STATEMENTS AND SUPPLEMENTARY"]
    notes_idx = -1
    for anchor in ["NOTES TO CONSOLIDATED FINANCIAL STATEMENTS", "NOTE 1.", "NOTE 1 "]:
        idx = upper.find(anchor)
        if idx != -1:
            w = upper[idx: idx + 500]
            if len(re.findall(r'\bITEM\s+\d+[A-Z]?[\.\s]', w)) < 3 and sum(1 for s in TOC_SIGNALS if s in w) < 2:
                notes_idx = idx
                logger.info(f"Notes section at {idx:,} via '{anchor}'")
                break
    if notes_idx == -1:
        notes_idx = upper.rfind("NOTE")
        if notes_idx == -1: return {}

    lease_section_start = -1
    for kw in ["LEASES", "LEASE EXPENSE", "OPERATING LEASE COST", "FINANCE LEASE COST",
                "RIGHT-OF-USE ASSETS", "RENTAL EXPENSE", "LEASE LIABILITIES"]:
        idx = upper.find(kw, notes_idx)
        if idx != -1 and idx < notes_idx + 500000:
            lease_section_start = idx
            logger.info(f"Lease section at {idx} via '{kw}'")
            break
    if lease_section_start == -1: return {}

    lease_section = raw_text[lease_section_start:lease_section_start + 50000]
    prompt = f"""Extract lease expense data for periods: {', '.join(period_names)}.
TEXT: {lease_section[:30000]}
Look for: Rental expense, Operating lease cost, Finance lease cost (Amortization of ROU assets,
Interest on lease liabilities), Short-term lease cost, Variable lease cost, Total lease cost.
Return ONLY valid JSON: {{"FY2024": {{"rental_expense": 269.4, "operating_lease_cost": 87.2,
"amortization_of_rou_assets": 45.5, "interest_on_lease_liabilities": 6.1,
"short_term_lease_cost": 0.2, "variable_lease_cost": 19.5, "total_lease_cost": 158.5}}}}
Use null if not found."""
    try:
        resp = get_client().chat.completions.create(
            model="gpt-4o-mini", temperature=0.0, max_tokens=2000,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "Extract lease expense data and return valid JSON."},
                {"role": "user", "content": prompt},
            ],
        )
        lease_data = json.loads(resp.choices[0].message.content or "{}")
        logger.info(f"Lease data: {json.dumps(lease_data, indent=2)}")
        return lease_data
    except Exception as e:
        logger.error(f"Error extracting lease data: {e}")
        return {}


def calculate_adjusted_rent_expense(
    rental_expense, operating_lease_cost, amortization_of_rou,
    interest_on_lease_liab, short_term_lease_cost, total_lease_cost,
) -> Optional[float]:
    if rental_expense is not None and (amortization_of_rou is not None
                                        or interest_on_lease_liab is not None
                                        or short_term_lease_cost is not None):
        adj = rental_expense
        if amortization_of_rou is not None: adj -= amortization_of_rou
        if interest_on_lease_liab is not None: adj -= interest_on_lease_liab
        if short_term_lease_cost is not None: adj -= short_term_lease_cost
        return adj
    return operating_lease_cost or rental_expense or total_lease_cost


# ============================================================
# REVOLVER EXTRACTION
# ============================================================

# ── Revolver extraction with Claude Sonnet ────────────────────────────────────
# Revolving credit facility data (facility size, borrowings, availability)
# is critical for liquidity analysis but appears in inconsistent formats:
#   - Structured table in the debt footnote
#   - Plain English prose paragraphs
#   - Combined with term loans and other facilities
#
# SEARCH STRATEGY (4 passes, each a fallback if the previous fails):
#   Pass 1: NOTE header regex matching "NOTE X. LONG-TERM DEBT"
#   Pass 2: Extended NOTE header patterns for less-standard formats
#   Pass 3: Keyword search for "REVOLVING CREDIT FACILITY" past the TOC
#   Pass 4: Prose search for "$X billion revolving credit facility" in plain text
#           — handles filings where the revolver is described narratively
#
# WHY CLAUDE: The revolver appears in prose, tables, or a mix of both.
# Claude handles unstructured prose better and the prompt includes concrete
# examples of both table and prose formats with expected JSON output.
def extract_revolver_data(raw_text: str, period_names: list[str]) -> dict:
    """
    Extract revolving credit facility data from 10-K debt notes.

    Search strategy (original logic preserved, fallbacks added):
    Pass 1: NOTE header regex — matches "NOTE 9. LONG-TERM DEBT" style headers
    Pass 2: Extended NOTE header patterns for more filing formats
    Pass 3: Keyword search past TOC zone (original fallback)
    Pass 4: NEW — prose search: find any paragraph mentioning revolving credit
            facility size in plain English (e.g. "has a $4.25 billion revolving
            credit facility"). Handles filings where the revolver is described
            in prose rather than a structured table.

    The Claude prompt explicitly handles both table AND prose formats so it can
    extract facility_size, borrowings, and availability from either.
    """
    if not raw_text or not period_names: return {}
    upper = raw_text.upper()
    doc_len = len(raw_text)
    TOC_SIGNALS = ["ITEM 7A.", "ITEM 8.", "ITEM 9.", "ITEM 10.",
                   "QUANTITATIVE AND QUALITATIVE", "FINANCIAL STATEMENTS AND SUPPLEMENTARY",
                   "CHANGES IN AND DISAGREEMENTS", "CONTROLS AND PROCEDURES"]

    def _is_toc(idx):
        w = upper[idx: idx + 2000]
        return len(re.findall(r'\bITEM\s+\d+[A-Z]?[\.\s]', w)) >= 5 or sum(1 for s in TOC_SIGNALS if s in w) >= 3

    debt_section_start = -1
    debt_section_end = -1

    # Pass 1 + 2: NOTE header patterns (original + extended)
    note_patterns = [
        r'NOTE\s+\d+[\.\:\-\s]+(DEBT|LONG[\-\s]?TERM DEBT|BORROWINGS)',
        r'NOTE\s+\d+[\.\:\-\s]+(CREDIT FACILITIES|CREDIT AGREEMENTS?)',
        r'NOTE\s+\d+[\.\:\-\s]+(FINANCING|FINANCIAL OBLIGATIONS)',
        r'NOTE\s+\d+[\.\:\-\s]+(SHORT[\-\s]?TERM BORROWINGS|NOTES PAYABLE)',
        # Extended patterns for more filing formats
        r'NOTE\s+\d+[\.\:\-\s]+(LONG[\-\s]?TERM DEBT AND SHORT[\-\s]?TERM BORROWINGS)',
        r'NOTE\s+\d+[\.\:\-\s]+(DEBT AND CREDIT FACILIT)',
        r'NOTE\s+\d+[\.\:\-\s]+(LINES OF CREDIT|LINE OF CREDIT)',
        r'NOTE\s+\d+[\.\:\-\s]+(CAPITAL RESOURCES|LIQUIDITY)',
        r'\d+\.\s+(LONG[\-\s]?TERM DEBT|DEBT|BORROWINGS)\s*\n',
    ]
    for pattern in note_patterns:
        for m in re.finditer(pattern, upper):
            if not _is_toc(m.start()):
                debt_section_start = m.start()
                logger.info(f"Debt note: '{m.group(0).strip()}' at {debt_section_start:,}")
                break
        if debt_section_start != -1: break

    if debt_section_start != -1:
        nxt = re.search(r'NOTE\s+\d+[\.\:\-]', upper[debt_section_start + 100:])
        debt_section_end = (debt_section_start + 100 + nxt.start()) if nxt else min(doc_len, debt_section_start + 250000)
    else:
        # Pass 3: Keyword search past TOC zone (original fallback)
        logger.warning("No debt NOTE header — keyword fallback")
        toc_end = int(doc_len * 0.10)
        for kw in ["REVOLVING CREDIT FACILITY", "REVOLVING CREDIT AGREEMENT", "CREDIT FACILIT",
                   "REVOLVING CREDIT", "BORROWINGS OUTSTANDING", "LONG-TERM DEBT"]:
            idx = upper.find(kw, toc_end)
            if idx != -1:
                debt_section_start = max(0, idx - 5000)
                debt_section_end = min(doc_len, idx + 250000)
                logger.info(f"Debt section via keyword '{kw}' at {idx:,}")
                break

    # Pass 4: NEW — prose fallback
    # Finds any paragraph mentioning a revolving credit facility with a dollar amount.
    # Handles filings where the revolver is described in plain English prose
    # rather than a structured table (e.g. "has a $4.25 billion revolving credit facility").
    if debt_section_start == -1:
        logger.warning("All note/keyword searches failed — trying prose revolving credit search")
        prose_patterns = [
            r'HAS A \$[\d\.]+\s*(BILLION|MILLION)[^.]*?REVOLVING CREDIT',
            r'REVOLVING CREDIT FACILITY[^.]*?\$[\d\.]+\s*(BILLION|MILLION)',
            r'REVOLVING CREDIT AGREEMENT[^.]*?\$[\d\.]+\s*(BILLION|MILLION)',
            r'[\$][\d\.]+\s*(BILLION|MILLION)[^.]*?REVOLVING CREDIT FACILIT',
        ]
        for pattern in prose_patterns:
            m = re.search(pattern, upper)
            if m:
                # Grab a generous window around the prose mention
                debt_section_start = max(0, m.start() - 500)
                debt_section_end   = min(doc_len, m.start() + 50000)
                logger.info(f"Prose revolving credit found at {m.start():,}: '{upper[m.start():m.start()+80]}'")
                break

    if debt_section_start == -1:
        logger.warning("Could not find debt/revolver section in document.")
        return {}

    debt_section = raw_text[debt_section_start:debt_section_end]
    extra = f',\n  "{period_names[1]}": {{"facilities": [...]}}' if len(period_names) > 1 else ""

    # Enhanced prompt: explicitly handles both table and prose formats
    prompt = f"""Extract ALL revolving credit facilities for periods: {', '.join(period_names)}.

TEXT ({len(debt_section[:120000]):,} chars):
{debt_section[:120000]}

INSTRUCTIONS:
1. Find every distinct revolving credit facility. Ignore term loans and LC sub-limits.
2. The facility may be described in a table OR in plain prose paragraphs. Read both.
   Examples of prose descriptions to extract from:
   - "has a $4.25 billion five-year revolving credit facility... was undrawn"
     -> facility_size=4250.0, borrowings=0.0, availability=4250.0
   - "a $2.0 billion revolving credit facility with $500 million outstanding"
     -> facility_size=2000.0, borrowings=500.0, availability=1500.0
3. If the text says "undrawn", "no borrowings", or "no amounts outstanding" -> borrowings=0.0
4. Letters of credit reduce availability: availability = facility_size - borrowings - LC_outstanding
5. Convert billions to millions (e.g. $4.25 billion = 4250.0)
6. For each period extract: name, facility_size, borrowings, availability
7. If a period is not mentioned, use the most recent data available.

Return ONLY valid JSON — no commentary, no markdown:
{{
  "{period_names[0]}": {{
    "facilities": [
      {{
        "name": "exact facility name",
        "facility_size": 4250.0,
        "borrowings": 0.0,
        "availability": 3650.0
      }}
    ]
  }}{extra}
}}
Use null only if a value genuinely cannot be determined."""

    result_text = "{}"
    try:
        client = get_anthropic_client()
        result_text = _call_claude_with_retry(
            client,
            model="claude-sonnet-4-6", max_tokens=4000,
            system="Extract revolving credit facility data from 10-K debt notes. Return valid JSON only.",
            messages=[{"role": "user", "content": prompt}],
        )
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0].strip()
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0].strip()

        revolver_data = json.loads(result_text)
        for period in list(revolver_data.keys()):
            if not isinstance(revolver_data[period], dict):
                revolver_data[period] = {"facilities": []}
        for period, pd in revolver_data.items():
            for f in (pd.get("facilities") or []):
                if isinstance(f, dict):
                    size, borr = f.get("facility_size"), f.get("borrowings")
                    if f.get("availability") is None and size is not None and borr is not None:
                        f["availability"] = size - borr

        logger.info("=" * 50)
        logger.info("REVOLVER SUMMARY")
        for period, pd in revolver_data.items():
            facs = (pd.get("facilities") or []) if isinstance(pd, dict) else []
            logger.info(f"  {period}: {len(facs)} facility/ies")
            for f in facs:
                if isinstance(f, dict):
                    logger.info(f"    {f.get('name')}: size=${f.get('facility_size')} borr=${f.get('borrowings')} avail=${f.get('availability')}")
        logger.info("=" * 50)
        return revolver_data
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error in revolver extraction: {e} — raw: {result_text[:300]}")
        return {}
    except Exception as e:
        logger.error(f"Error extracting revolver data: {e}")
        return {}


# ============================================================
# FINANCIAL STATEMENTS TEXT SELECTION
# ============================================================

# ── Financial statement text selection ────────────────────────────────────────
# A full 10-K runs 200-400 pages. Sending everything to GPT-4o would cost $5-10
# and actually reduce accuracy — the model attends better to focused inputs.
#
# This function finds the start of the financial statements section
# (typically "CONSOLIDATED STATEMENTS OF INCOME") and extracts a
# 260,000-character window from that point forward, covering the statements
# and all footnotes.
#
# SEARCH STRATEGY:
#   Tier 1: Full statement headers like "CONSOLIDATED STATEMENTS OF INCOME"
#   Tier 2: Broader markers like "FINANCIAL STATEMENTS" as fallback
#
# Each candidate is verified to contain financial signal words (NET SALES,
# TOTAL ASSETS, etc.) to avoid false positives in the Table of Contents.
def _select_relevant_statement_text(raw_text: str, max_chars: int = 260000) -> str:def _select_relevant_statement_text(raw_text: str, max_chars: int = 260000) -> str:
    if not raw_text: return ""
    upper = raw_text.upper()
    FS_SIGNALS = [
        "NET SALES", "NET REVENUES", "TOTAL REVENUES", "COST OF GOODS SOLD",
        "COST OF SALES", "GROSS PROFIT", "OPERATING INCOME", "NET INCOME",
        "TOTAL ASSETS", "TOTAL LIABILITIES", "STOCKHOLDERS", "CASH AND CASH EQUIVALENTS",
    ]
    tier1 = [
        "CONSOLIDATED STATEMENTS OF OPERATIONS", "CONSOLIDATED STATEMENTS OF INCOME",
        "CONSOLIDATED STATEMENTS OF EARNINGS", "CONSOLIDATED BALANCE SHEETS",
        "CONSOLIDATED STATEMENTS OF CASH FLOWS", "NOTES TO CONSOLIDATED FINANCIAL STATEMENTS",
    ]
    tier2 = ["FINANCIAL STATEMENTS", "ITEM 8. FINANCIAL STATEMENTS", "STATEMENTS OF CASH FLOWS"]
    start = -1
    for anchors in [tier1, tier2]:
        for anchor in anchors:
            pos = 0
            while True:
                idx = upper.find(anchor, pos)
                if idx == -1: break
                if any(s in raw_text[idx:idx+3000].upper() for s in FS_SIGNALS):
                    start = max(0, idx - 5000)
                    logger.info(f"FS anchor: '{anchor[:40]}' at {idx:,}")
                    break
                pos = idx + 1
            if start != -1: break
        if start != -1: break
    if start == -1:
        for anchor in tier1:
            idx = upper.rfind(anchor)
            if idx != -1: start = max(0, idx - 5000); break
    if start == -1:
        start = max(0, int(len(raw_text) * 0.65))

    slice_text = raw_text[start: min(len(raw_text), start + max_chars)]
    tables_idx = upper.rfind("TABLES:")
    if tables_idx == -1: tables_idx = upper.find("TABLES:")
    if tables_idx != -1 and "TABLES:" not in slice_text:
        slice_text += "\n\n" + raw_text[tables_idx: min(len(raw_text), tables_idx + 220000)]
    logger.info(f"FS text: start={start} len={len(slice_text)}")
    return slice_text


def _extract_tables_block(text: str, max_chars: int = 160000) -> str:
    if not text: return ""
    idx = text.upper().find("TABLES:")
    return "" if idx == -1 else text[idx: idx + max_chars]


# ── Numeric excerpt ───────────────────────────────────────────────────────────
# Further filters the financial statements section to only lines containing
# financial amounts, plus a small context window around each match.
#
# WHY KEEP CONTEXT LINES:
# A line containing "5,200" by itself means nothing. The line before it —
# "Net sales" — defines what the number is. context_lines=2 keeps the label.
#
# WHY KEEP YEAR HEADER ROWS:
# Lines like "2024 | 2023 | 2022" are column headers that tell GPT-4o which
# number belongs to which fiscal year. Without them, multi-period extraction
# would fail — the model wouldn't know that the first number is FY2024.
# These rows are kept even if they contain no dollar amounts.
#
# Reduces input from ~260,000 to ~130,000 chars while preserving all
# financially meaningful content. Saves ~$0.50 per run in token costs.
def _numeric_excerpt(text: str, max_chars: int = 130000, context_lines: int = 2) -> str:
    """
    Extract lines containing financial amounts plus surrounding context.
    Also keeps year-header rows (e.g. "2024 | 2023 | 2022") so the model
    knows which column belongs to which period — critical for multi-period extraction.
    """
    if not text: return ""
    lines = text.splitlines()
    keep = [False] * len(lines)
    pat = re.compile(r"(\(\s*\d{1,3}(?:,\d{3})+(?:\.\d+)?\s*\))|(\d{1,3}(?:,\d{3})+(?:\.\d+)?)|(\d+\.\d+)")
    # Keep rows that look like year column headers: contain 2+ different years
    year_header_re = re.compile(r'\b20\d{2}\b')
    for i, line in enumerate(lines):
        years_found = set(year_header_re.findall(line))
        is_year_header = len(years_found) >= 2
        if pat.search(line) or is_year_header:
            for j in range(max(0, i - context_lines), min(len(lines), i + context_lines + 1)):
                keep[j] = True
    filtered = [lines[i].strip() for i in range(len(lines)) if keep[i] and lines[i].strip()]
    return ("\n".join(filtered) if filtered else text)[:max_chars]


# ============================================================
# MAIN EXTRACTION ENTRY POINT
# ============================================================

class _extract_profile_context:
    """Simple namespace — set company_name before calling extract_financials_from_text."""
    company_name = None


# ── Main extraction entry point ───────────────────────────────────────────────
# Every analysis — file upload, EDGAR search, URL paste — flows through here.
#
# STEP-BY-STEP:
#
# 1. PREP: Strip HTML, select the financial statements section, build the
#    numeric excerpt and tables block, combine into the GPT-4o input.
#
# 2. RULES INJECTION: If the company has saved extraction rules in the
#    profile system (from Review & Correct), inject them into the prompt.
#    These override the general aliases for this specific company.
#    Example: "For CHD, rent_expense: look for 'Operating lease cost'
#    in the LEASES section. Ignore 'Total lease cost'."
#
# 3. GPT-4O CALL: Returns a JSON object with one period per fiscal year column.
#    _normalize_payload validates and coerces all values to the right types.
#
# 4. CONTEXT RECORDING: For each extracted numeric value, find where in the
#    original document it came from. Stored as a structured note on the period.
#    Powers the "Show where it was found" display in Review & Correct.
#
# 5. LEASE EXTRACTION: Separate GPT-4o-mini call to the lease footnote.
#    Merged into income_statement["rent_expense"].
#
# 6. REVOLVER EXTRACTION: Separate Claude call to the debt footnotes.
#    Merged into balance_sheet (facility_size, borrowings, availability).
#
# 7. CASH TAXES FALLBACK: If any period still has null cash_paid_for_income_taxes
#    after the main extraction, try the 3M-style footnote table format.
#
# 8. MD&A: Extract Item 7, send to Claude Sonnet for segment analysis.
#
# 9. PROFILE: Apply saved corrections, flag remaining conflicts.
#
# Returns (ExtractionResult, excerpt_text, raw_gpt_response, mda_summary)
def extract_financials_from_text(raw_text: str) -> Tuple[ExtractionResult, str, str, str]:def extract_financials_from_text(raw_text: str) -> Tuple[ExtractionResult, str, str, str]:
    raw_text = _strip_html_to_text(raw_text)
    selected = _select_relevant_statement_text(raw_text, max_chars=260000)
    tables_block = _extract_tables_block(selected, max_chars=200000)
    numeric_block = _numeric_excerpt(selected, max_chars=130000, context_lines=1)
    combined_excerpt = (tables_block + "\n\n" if tables_block else "") + "NUMERIC LINES + CONTEXT:\n" + numeric_block

    # Inject company-specific extraction rules if available
    rules_block = ""
    company_name_for_rules = getattr(_extract_profile_context, 'company_name', None)
    if company_name_for_rules and _PROFILES_AVAILABLE:
        rules_block = _profile_store.build_rules_prompt_block(company_name_for_rules)
        if rules_block:
            logger.info(f"Injecting extraction rules for {company_name_for_rules}")

    user_prompt = f"""Extract financial statement line items for ALL periods shown.

{rules_block}CRITICAL MULTI-PERIOD RULES:
- Financial statements show multiple year columns separated by | pipes
  e.g. "Net sales | 11,123 | 13,696 | 14,812" means FY2024=11123, FY2023=13696, FY2022=14812
- You MUST return ONE separate period object for EACH year column shown
- Look for header rows like "Year Ended December 31, | 2024 | 2023 | 2022"
- If you see 3 years of data return 3 period objects; if 2 years return 2 objects
- Return up to 5 periods most recent first named FY2024, FY2023, FY2022 etc.
- NEVER combine multiple years into one period object
- NEVER put multiple numbers into a single numeric field

GENERAL: prefer pipe-delimited TABLES rows; each field is a single scalar or null;
derived_metrics must be {{}}; leave ebitda null unless explicitly labeled EBITDA.

ZERO DEFAULT (two fields only):
- current_portion_long_term_debt: use 0 if the line does not appear (not null)
- revolver_borrowings: use 0 if a revolver exists but no borrowings line appears (not null)
- All other fields: null if not found — never fabricate a zero for missing data.

ALIASES:
revenue: Net revenue/sales/Total revenues
cost_of_sales: Cost of sales/products/goods sold
sga_expense: Selling general and administrative
operating_income: Operating income/profit/EBIT
net_income: Net income/earnings
interest_expense: Interest expense
income_tax_expense: Provision for income taxes
depreciation_amortization: D&A
rent_expense: Operating lease cost/Rent expense/Rental expense
cash: Cash and cash equivalents
total_debt: Total debt/borrowings
long_term_debt: Long-term debt
current_portion_long_term_debt: Current portion of long-term debt
cfo: Net cash from operating activities
cfi: Net cash from investing activities
cff: Net cash from financing activities
capex: Capital expenditures/Additions to PP&E
cash_paid_for_interest: Cash paid for interest
cash_paid_for_income_taxes: Cash paid for income taxes,
  Cash payments for income taxes, Cash income tax payments, Taxes paid (net),
  Income taxes paid (net). SOURCE: supplemental cash flow disclosures only — single total line.
dividends_distributions_paid: Dividends/Distributions paid,
  Cash dividends paid, Distributions paid.

cash_paid_for_interest: Cash paid for interest, Interest paid, Cash interest paid,
  Cash payments for interest, Interest paid (net).
  SOURCE RULE: extract ONLY from supplemental cash flow disclosures.
  Do NOT use interest expense from the income statement.

STATEMENT TEXT:
{combined_excerpt}

{EXTRACTOR_SCHEMA_PROMPT}"""

    resp = get_client().chat.completions.create(
        model="gpt-4o",
        temperature=0.0,
        messages=[
            {"role": "system", "content": EXTRACTOR_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    raw_model_text = resp.choices[0].message.content or ""
    extracted = ExtractionResult.model_validate(_normalize_payload(_coerce_json(raw_model_text)))
    period_names = [p.period_name for p in extracted.periods] if extracted.periods else []

    # Record extraction context for each field (what the extractor found and where)
    # This powers the "Show me what it found" display in the Review & Correct UI
    # Runs always (not gated on company_name) so context is available even without borrower name
    if _PROFILES_AVAILABLE:
        for p in (extracted.periods or []):
            for section_name, section_keys in [
                ("income_statement", ["revenue","cost_of_sales","sga_expense",
                  "operating_income","net_income","interest_expense",
                  "income_tax_expense","depreciation_amortization","rent_expense"]),
                ("balance_sheet",    ["cash","total_debt","long_term_debt",
                  "current_portion_long_term_debt","revolver_facility_size",
                  "revolver_borrowings","revolver_availability"]),
                ("cash_flow",        ["cfo","capex","cash_paid_for_interest",
                  "cash_paid_for_income_taxes","dividends_distributions_paid"]),
            ]:
                section = getattr(p, section_name, None) or {}
                for field_key in section_keys:
                    val = section.get(field_key) if isinstance(section, dict) else None
                    if val is not None:
                        ctx = _profile_store.extract_field_context(
                            raw_text=raw_text,
                            field_name=field_key,
                            extracted_value=val,
                            context_lines=3,
                        )
                        if ctx:
                            if not isinstance(p.notes, list):
                                p.notes = []
                            # Store context as a structured note
                            ctx_note = f"CONTEXT:{field_key}:{ctx}"
                            # Remove old context note for this field if exists
                            p.notes = [n for n in p.notes
                                      if not (isinstance(n, str) and n.startswith(f"CONTEXT:{field_key}:"))]
                            p.notes.append(ctx_note)

    # Lease
    lease_data = extract_lease_data(raw_text, period_names) if period_names else {}
    for p in (extracted.periods or []):
        is_ = p.income_statement or {}
        pld = lease_data.get(p.period_name) or {}
        if pld:
            if not p.notes: p.notes = []
            p.notes.append(f"Lease data: {json.dumps(pld)}")
            ar = calculate_adjusted_rent_expense(
                pld.get("rental_expense"), pld.get("operating_lease_cost"),
                pld.get("amortization_of_rou_assets"), pld.get("interest_on_lease_liabilities"),
                pld.get("short_term_lease_cost"), pld.get("total_lease_cost"),
            )
            if ar is not None: is_["rent_expense"] = ar
        if is_.get("rent_expense") is None:
            r = _extract_rent_from_mda(raw_text, p.period_name)
            if r is not None: is_["rent_expense"] = r

    # Revolver
    revolver_data = extract_revolver_data(raw_text, period_names) if period_names else {}
    for p in (extracted.periods or []):
        bs_ = p.balance_sheet or {}
        prd = revolver_data.get(p.period_name) or {}
        if prd:
            facs = prd.get("facilities") or []
            if facs:
                bs_["revolver_facility_size"] = sum(f.get("facility_size",0) or 0 for f in facs if isinstance(f,dict)) or None
                bs_["revolver_borrowings"] = sum(f.get("borrowings",0) or 0 for f in facs if isinstance(f,dict)) or 0.0
                bs_["revolver_availability"] = sum(f.get("availability",0) or 0 for f in facs if isinstance(f,dict)) or None
                # NOTE: balance_sheet is schema-typed as Dict[str, Optional[float]] —
                # a list of facility dicts can't live there (breaks response_model
                # validation). Store the per-facility breakdown as a note instead,
                # same pattern used for lease data and extraction context.
                if not isinstance(p.notes, list):
                    p.notes = []
                p.notes = [n for n in p.notes
                           if not (isinstance(n, str) and n.startswith("REVOLVER_DETAIL:"))]
                p.notes.append(f"REVOLVER_DETAIL:{json.dumps(facs)}")
            else:
                bs_["revolver_facility_size"] = prd.get("revolver_facility_size")
                bs_["revolver_borrowings"] = prd.get("revolver_borrowings")
                bs_["revolver_availability"] = prd.get("revolver_availability")

    # Cash taxes footnote fallback
    # If any period still has null cash_paid_for_income_taxes, try the footnote table
    if extracted.periods:
        period_names_check = [p.period_name for p in extracted.periods]
        taxes_missing = any(
            (p.cash_flow or {}).get("cash_paid_for_income_taxes") is None
            for p in extracted.periods
        )
        if taxes_missing:
            logger.info("cash_paid_for_income_taxes missing for one or more periods — trying footnote fallback")
            footnote_taxes = _extract_cash_taxes_from_footnote(raw_text, period_names_check)
            if footnote_taxes:
                for p in extracted.periods:
                    cf = p.cash_flow or {}
                    if cf.get("cash_paid_for_income_taxes") is None:
                        fallback_val = footnote_taxes.get(p.period_name)
                        if fallback_val is not None:
                            cf["cash_paid_for_income_taxes"] = fallback_val
                            logger.info(f"Cash taxes footnote fallback applied: {p.period_name} = {fallback_val}")
            else:
                logger.info("Cash taxes footnote fallback: no data found")

    # MD&A
    mda_text = _extract_mda_section(raw_text, max_chars=350000)
    if not mda_text:
        logger.error("_extract_mda_section returned empty string.")
    mda_summary = _summarize_mda_for_drivers(mda_text, extracted)
    # Profile system: apply overrides then flag remaining conflicts
    company_name = getattr(_extract_profile_context, 'company_name', None)
    if company_name and _PROFILES_AVAILABLE:
        # Step 1: Apply saved corrections (override wrong extracted values)
        override_flags = _profile_store.apply_profile_overrides(company_name, extracted)
        if override_flags:
            existing = list(extracted.validation_flags or [])
            extracted.validation_flags = existing + override_flags
            logger.info(f"Profile overrides applied: {len(override_flags)} for {company_name}")
        # Step 2: Flag any remaining conflicts for review
        conflict_flags = _profile_store.apply_profile_to_extracted(company_name, extracted)
        if conflict_flags:
            existing = list(extracted.validation_flags or [])
            extracted.validation_flags = existing + conflict_flags
            logger.info(f"Profile conflicts flagged: {len(conflict_flags)} for {company_name}")

    return extracted, combined_excerpt, raw_model_text, mda_summary


def _extract_rent_from_mda(raw_text: str, period_name: str) -> Optional[float]:
    upper = raw_text.upper()
    ym = re.search(r'20\d{2}', period_name)
    if not ym: return None
    year = ym.group()
    for marker in ["OPERATING LEASE COST", "LEASE EXPENSE", "OPERATING LEASE EXPENSE", "RENT EXPENSE"]:
        idx = upper.find(marker)
        if idx != -1:
            ctx = raw_text[max(0, idx-500):idx+1500]
            pat = (r'(?:' + year + r'[^\d]*?[\$]?\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?))'
                   r'|(?:[\$]?\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)[^\d]*?' + year + r')')
            for mt in re.findall(pat, ctx, re.IGNORECASE):
                s = mt[0] or mt[1]
                if s:
                    try:
                        v = float(s.replace(',',''))
                        if 1 < v < 10000: return v
                    except: continue
    return None


# ============================================================
# CASH TAXES FOOTNOTE FALLBACK
# ============================================================

def _extract_cash_taxes_from_footnote(raw_text: str, period_names: list[str]) -> dict:
    """
    Fallback for cash_paid_for_income_taxes when the supplemental cash flow
    disclosures don't have a matching line.

    Some filings (e.g. 3M) present cash income tax payments in a standalone
    footnote table with rows for Federal, State, International and a Total row,
    rather than in the standard supplemental cash flow section.

    This function:
    1. Searches for the footnote table header keywords
    2. Finds the Total row which has values for all periods
    3. Returns a dict of {period_name: total_value}

    Only called as a last resort — after the main extractor has already tried
    the supplemental cash flow disclosures section and returned null.
    """
    if not raw_text or not period_names:
        return {}

    upper = raw_text.upper()

    # Keywords that introduce this type of footnote table
    footnote_headers = [
        "CASH INCOME TAX PAYMENTS",
        "CASH INCOME TAX PAYMENTS, NET OF REFUNDS",
        "INCOME TAX PAYMENTS, NET OF REFUNDS",
        "CASH PAID FOR INCOME TAXES CONSISTED",
        "INCOME TAXES PAID CONSISTED",
    ]

    section_start = -1
    for header in footnote_headers:
        idx = upper.find(header)
        if idx != -1:
            section_start = idx
            logger.info(f"Cash taxes footnote found via '{header}' at {idx:,}")
            break

    if section_start == -1:
        return {}

    # Extract a window around the footnote (large enough to contain the full table)
    section = raw_text[section_start: section_start + 5000]
    section_upper = section.upper()

    # Find the Total row — it should have values for all periods
    # Look for a line containing "Total" followed by numbers
    total_pattern = re.compile(
        r'(?:^|\n)\s*Total[^\n]*?'
        r'[\$\s]*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)',
        re.IGNORECASE | re.MULTILINE
    )

    result = {}
    matches = list(total_pattern.finditer(section))
    if not matches:
        logger.warning("Cash taxes footnote: no Total row found")
        return {}

    # Take the last Total match (most likely the grand total, not a subtotal)
    last_match = matches[-1]
    total_line = section[last_match.start(): last_match.start() + 300]
    logger.info(f"Cash taxes footnote Total line: {total_line[:150].replace(chr(10), ' ')}")

    # Extract all numbers from the total line
    numbers = re.findall(r'(\d{1,3}(?:,\d{3})*(?:\.\d+)?)', total_line)
    numbers = [float(n.replace(',', '')) for n in numbers]

    logger.info(f"Cash taxes footnote numbers found: {numbers}")

    # Match numbers to period names in order (most recent first)
    for i, period in enumerate(period_names):
        if i < len(numbers):
            result[period] = numbers[i]
            logger.info(f"Cash taxes footnote: {period} = {numbers[i]}")

    return result


# ============================================================
# COMPUTATIONS
# ============================================================

# ── Deterministic credit metric computation ───────────────────────────────────
# After GPT-4o extracts the raw financials, this function computes all
# credit metrics in pure Python — no LLM, no randomness, fully reproducible.
#
# WHY DETERMINISTIC:
# Credit decisions must be auditable. EBITDA, FCC, and leverage must give
# the same answer every time for the same inputs. These cannot be delegated
# to an LLM that might round differently each run.
#
# COMPUTED FOR EACH PERIOD:
#   ebitda_computed  = operating_income + D&A + rent_expense
#   free_cash_flow   = CFO - |capex|
#   ebitda_margin    = EBITDA / revenue
#   fcc              = (EBITDA - |capex| - |cash_taxes|) / (CPLTD + |cash_interest|)
#   leverage         = total_debt / EBITDA
#   altman_z_score   = weighted distress formula (assets, equity, earnings, revenue)
#   pd_score (1-12)  = derived from leverage, FCC, margin, Z-score, revenue growth
#   revenue_growth   = (current - prior) / prior (requires 2+ periods)
#
# VALIDATION FLAGS:
# When a component is missing (e.g. no total_debt = can't compute leverage),
# a human-readable warning is added to validation_flags. These appear as
# yellow warning banners in the frontend and are listed in the memo.
#
# EBITDA WRITEBACK: The computed EBITDA is written back into
# income_statement["ebitda"] so it's available everywhere the income
# statement is used (exports, memo, Review & Correct display).
def validate_and_compute(result: ExtractionResult) -> ExtractionResult:def validate_and_compute(result: ExtractionResult) -> ExtractionResult:
    flags: list[str] = []
    for p in result.periods:
        p.income_statement = dict(p.income_statement or {})
        p.balance_sheet = dict(p.balance_sheet or {})
        p.cash_flow = dict(p.cash_flow or {})
        p.derived_metrics = {}
        is_ = p.income_statement; bs = p.balance_sheet; cf = p.cash_flow
        op_inc = is_.get("operating_income"); da = is_.get("depreciation_amortization"); rent = is_.get("rent_expense")
        ebitda = compute_ebitda(op_inc, da, rent_expense=rent, include_rent=True)
        if ebitda is None:
            if op_inc is None: flags.append(f"{p.period_name}: operating_income missing; EBITDA not computed.")
            if da is None: flags.append(f"{p.period_name}: depreciation_amortization missing; EBITDA not computed.")
        else:
            p.derived_metrics["ebitda_includes_rent"] = rent is not None
            is_["ebitda"] = float(ebitda); p.derived_metrics["ebitda_computed"] = float(ebitda)
            if rent is None: flags.append(f"{p.period_name}: rent_expense not found; EBITDA excludes rent.")
        fcf = compute_free_cash_flow(cf.get("cfo"), cf.get("capex"))
        if fcf is not None: p.derived_metrics["free_cash_flow"] = float(fcf)
        em = compute_ebitda_margin(is_.get("ebitda"), is_.get("revenue"))
        if em is not None: p.derived_metrics["ebitda_margin"] = float(em)
        capex = cf.get("capex"); cash_taxes = cf.get("cash_paid_for_income_taxes")
        cpltd = bs.get("current_portion_long_term_debt"); cash_int = cf.get("cash_paid_for_interest")
        fcc = compute_fcc(ebitda=is_.get("ebitda"), capex=capex, cash_taxes=cash_taxes, cpltd=cpltd, cash_interest=cash_int)
        if fcc is None:
            for field, label in [(is_.get("ebitda"),"EBITDA"),(capex,"capex"),(cash_taxes,"cash_paid_for_income_taxes"),(cpltd,"CPLTD"),(cash_int,"cash_paid_for_interest")]:
                if field is None: flags.append(f"{p.period_name}: {label} missing; FCC not computed.")
        else:
            p.derived_metrics["fcc"] = float(fcc)
            cm, tm, im = abs_outflow(capex), abs_outflow(cash_taxes), abs_outflow(cash_int)
            if isinstance(is_.get("ebitda"),(int,float)) and cm is not None and tm is not None:
                p.derived_metrics["fcc_numerator"] = float(is_["ebitda"]) - float(cm) - float(tm)
            if cpltd is not None and im is not None:
                p.derived_metrics["fcc_denominator"] = float(cpltd) + float(im)
        lev = compute_leverage(bs.get("total_debt"), is_.get("ebitda"))
        if lev is None:
            if bs.get("total_debt") is None: flags.append(f"{p.period_name}: total_debt missing; leverage not computed.")
            ev = is_.get("ebitda")
            if ev is None: flags.append(f"{p.period_name}: EBITDA missing; leverage not computed.")
            elif isinstance(ev,(int,float)) and ev <= 0: flags.append(f"{p.period_name}: EBITDA<=0; leverage not computed.")
        else: p.derived_metrics["leverage_total_debt_to_ebitda"] = float(lev)
        z = compute_altman_z_score(revenue=is_.get("revenue"), operating_income=is_.get("operating_income"),
            total_assets=bs.get("total_assets"), total_liabilities=bs.get("total_liabilities"),
            total_equity=bs.get("total_equity"), retained_earnings=None)
        if z is not None: p.derived_metrics["altman_z_score"] = float(z)
        else: flags.append(f"{p.period_name}: Insufficient data for Altman Z-Score.")
        rev_growth = None
        if len(result.periods) >= 2:
            ci = result.periods.index(p)
            if ci < len(result.periods) - 1:
                pr = result.periods[ci+1]
                cr, ppr = is_.get("revenue"), (pr.income_statement or {}).get("revenue")
                if cr is not None and ppr is not None and ppr != 0:
                    rev_growth = (cr - ppr) / ppr
        pd_score, credit_rating = compute_pd_score(leverage=lev, fcc=fcc, ebitda_margin=p.derived_metrics.get("ebitda_margin"),
            altman_z=z, revenue_growth=rev_growth, free_cash_flow=p.derived_metrics.get("free_cash_flow"), ebitda=is_.get("ebitda"))
        if pd_score is not None:
            p.derived_metrics["pd_score"] = int(pd_score)
            p.derived_metrics["credit_rating"] = credit_rating
    seen: set[str] = set()
    result.validation_flags = [f for f in flags if not (f in seen or seen.add(f))]
    return result


# ============================================================
# FINANCIAL SUMMARY MEMO (no borrower context)
# ============================================================

def generate_financial_summary_memo(extracted: ExtractionResult) -> str:
    if not extracted.periods: return "## Financial Summary\n- No periods extracted.\n"
    extracted = validate_and_compute(extracted)
    def m(x): return "N/A" if x is None else f"${float(x):,.0f}"
    def xf(x): return "N/A" if x is None else f"{float(x):.2f}x"
    def pc(x): return "N/A" if x is None else f"{float(x)*100:.1f}%"
    def yoy(n,o):
        if n is None or o is None or float(o)==0: return None
        return (float(n)-float(o))/float(o)
    p0 = extracted.periods[0]; basis = extracted.statement_basis or "actual"
    is0=p0.income_statement or {}; bs0=p0.balance_sheet or {}; cf0=p0.cash_flow or {}; dm0=p0.derived_metrics or {}
    lines = [
        "## Financial Summary (Extracted + Computed)",
        f"*Statement basis:* `{basis}`  ",
        f"*Most recent period:* `{p0.period_name}`\n",
        "### Executive Takeaways",
        f"- Revenue: {m(is0.get('revenue'))}.",
        f"- EBITDA: {m(is0.get('ebitda'))} ({pc(dm0.get('ebitda_margin'))} margin).",
        f"- Leverage: {xf(dm0.get('leverage_total_debt_to_ebitda'))}  |  FCC: {xf(dm0.get('fcc'))}.",
        f"- FCF: {m(dm0.get('free_cash_flow'))}  |  Cash: {m(bs0.get('cash'))}.\n",
    ]
    if len(extracted.periods) >= 2:
        p1=extracted.periods[1]; is1=p1.income_statement or {}; dm1=p1.derived_metrics or {}
        lines += [
            f"### YoY Trend ({p1.period_name} → {p0.period_name})",
            f"- Revenue: {m(is1.get('revenue'))} → {m(is0.get('revenue'))} ({pc(yoy(is0.get('revenue'),is1.get('revenue')))})",
            f"- EBITDA: {m(is1.get('ebitda'))} → {m(is0.get('ebitda'))} ({pc(yoy(is0.get('ebitda'),is1.get('ebitda')))})",
            f"- EBITDA margin: {pc(dm1.get('ebitda_margin'))} → {pc(dm0.get('ebitda_margin'))}",
            "",
        ]
    flags = list(extracted.validation_flags or [])
    if flags: lines += ["### Data Gaps"] + [f"- {f}" for f in flags] + [""]
    return "\n".join(lines).strip() + "\n"


# ============================================================
# FORMATTING HELPERS
# ============================================================

def _fmt_money(x, basis):
    if x is None: return "N/A"
    s = "MM" if basis=="millions" else ("K" if basis=="thousands" else "")
    return f"${x:,.0f}{s}"

def _fmt_pct(x): return "N/A" if x is None else f"{x*100:.1f}%"
def _fmt_x(x): return "N/A" if x is None else f"{x:.2f}x"

def _safe_pct_change(n, o):
    if n is None or o is None or o==0: return None
    return (n-o)/o


def _format_extracted_data(extracted: ExtractionResult, basis: str) -> str:
    s = "MM" if basis=="millions" else ("K" if basis=="thousands" else "")
    lines = ["## Extracted Financial Data", "", f"**Statement Basis:** {basis}", ""]
    for period in extracted.periods:
        lines += [f"### {period.period_name}", ""]
        is_ = period.income_statement or {}
        if any(v is not None for v in is_.values()):
            lines.append("#### Income Statement")
            for label, key in [
                ("Revenue","revenue"),("Cost of Sales","cost_of_sales"),("SG&A","sga_expense"),
                ("Operating Income","operating_income"),("EBITDA","ebitda"),("Net Income","net_income"),
                ("Interest Expense","interest_expense"),("Income Tax","income_tax_expense"),
                ("D&A","depreciation_amortization"),("Rent Expense","rent_expense"),
            ]:
                if is_.get(key) is not None: lines.append(f"  - {label}: ${is_[key]:,.0f}{s}")
            lines.append("")
        bs = period.balance_sheet or {}
        if any(v is not None for v in bs.values()):
            lines.append("#### Balance Sheet")
            for label, key in [
                ("Cash","cash"),("Total Assets","total_assets"),("Total Liabilities","total_liabilities"),
                ("Total Equity","total_equity"),("Total Debt","total_debt"),("Long-term Debt","long_term_debt"),
                ("Current Portion LTD","current_portion_long_term_debt"),
                ("Revolver Facility Size","revolver_facility_size"),
                ("Revolver Borrowings","revolver_borrowings"),
                ("Revolver Availability","revolver_availability"),
            ]:
                if bs.get(key) is not None: lines.append(f"  - {label}: ${bs[key]:,.0f}{s}")
            lines.append("")
        cf = period.cash_flow or {}
        if any(v is not None for v in cf.values()):
            lines.append("#### Cash Flow Statement")
            for label, key in [
                ("Operating Cash Flow","cfo"),("Investing Cash Flow","cfi"),("Financing Cash Flow","cff"),
                ("Capex","capex"),("Cash Paid for Interest","cash_paid_for_interest"),
                ("Cash Paid for Taxes","cash_paid_for_income_taxes"),("Dividends Paid","dividends_distributions_paid"),
            ]:
                if cf.get(key) is not None: lines.append(f"  - {label}: ${cf[key]:,.0f}{s}")
            lines.append("")
        dm = period.derived_metrics or {}
        if any(v is not None for v in dm.values()):
            lines.append("#### Computed Metrics")
            for label, key, fmt in [
                ("EBITDA (Computed)","ebitda_computed", lambda v: f"${v:,.0f}{s}"),
                ("EBITDA Margin","ebitda_margin", lambda v: f"{v*100:.1f}%"),
                ("Leverage (Debt/EBITDA)","leverage_total_debt_to_ebitda", lambda v: f"{v:.2f}x"),
                ("FCC","fcc", lambda v: f"{v:.2f}x"),
                ("Free Cash Flow","free_cash_flow", lambda v: f"${v:,.0f}{s}"),
                ("FCC Numerator","fcc_numerator", lambda v: f"${v:,.0f}{s}"),
                ("FCC Denominator","fcc_denominator", lambda v: f"${v:,.0f}{s}"),
                ("Altman Z-Score","altman_z_score", lambda v: f"{v:.2f}"),
                ("PD Score","pd_score", lambda v: f"{int(v)}/12"),
                ("Credit Rating","credit_rating", lambda v: str(v)),
            ]:
                if dm.get(key) is not None: lines.append(f"  - {label}: {fmt(dm[key])}")
            lines.append("")
    return "\n".join(lines)


def _build_metric_summary(extracted: ExtractionResult, basis: str) -> str:
    if not extracted.periods: return "No periods available"
    p0 = extracted.periods[0]
    is0=p0.income_statement or {}; bs0=p0.balance_sheet or {}; cf0=p0.cash_flow or {}; dm0=p0.derived_metrics or {}
    lines = [
        f"MOST RECENT PERIOD: {p0.period_name}",
        f"Revenue: {_fmt_money(is0.get('revenue'),basis)}",
        f"Cost of Sales: {_fmt_money(is0.get('cost_of_sales'),basis)}",
        f"Gross Profit: {_fmt_money((is0.get('revenue') or 0) - (is0.get('cost_of_sales') or 0) if is0.get('revenue') and is0.get('cost_of_sales') else None, basis)}",
        f"SG&A: {_fmt_money(is0.get('sga_expense'),basis)}",
        f"Operating Income: {_fmt_money(is0.get('operating_income'),basis)}",
        f"EBITDA: {_fmt_money(is0.get('ebitda'),basis)}",
        f"EBITDA Margin: {_fmt_pct(dm0.get('ebitda_margin'))}",
        f"Net Income: {_fmt_money(is0.get('net_income'),basis)}",
        f"Total Debt: {_fmt_money(bs0.get('total_debt'),basis)}",
        f"Leverage (Debt/EBITDA): {_fmt_x(dm0.get('leverage_total_debt_to_ebitda'))}",
        f"CFO: {_fmt_money(cf0.get('cfo'),basis)}",
        f"Capex: {_fmt_money(abs_outflow(cf0.get('capex')),basis)}",
        f"FCF: {_fmt_money(dm0.get('free_cash_flow'),basis)}",
        f"FCC: {_fmt_x(dm0.get('fcc'))}",
        f"Cash: {_fmt_money(bs0.get('cash'),basis)}",
        f"Revolver Facility Size: {_fmt_money(bs0.get('revolver_facility_size'),basis)}",
        f"Revolver Borrowings: {_fmt_money(bs0.get('revolver_borrowings'),basis)}",
        f"Revolver Availability: {_fmt_money(bs0.get('revolver_availability'),basis)}",
        f"PD Score: {int(dm0['pd_score'])}/12 — {dm0.get('credit_rating','N/A')}" if dm0.get('pd_score') is not None else "PD Score: N/A",
        f"Altman Z-Score: {dm0['altman_z_score']:.2f}" if dm0.get('altman_z_score') is not None else "Altman Z-Score: N/A",
    ]
    if len(extracted.periods) >= 2:
        p1=extracted.periods[1]
        is1=p1.income_statement or {}; cf1=p1.cash_flow or {}; dm1=p1.derived_metrics or {}; bs1=p1.balance_sheet or {}
        lines += [
            f"\nPRIOR PERIOD: {p1.period_name}",
            f"Revenue: {_fmt_money(is1.get('revenue'),basis)}",
            f"Cost of Sales: {_fmt_money(is1.get('cost_of_sales'),basis)}",
            f"SG&A: {_fmt_money(is1.get('sga_expense'),basis)}",
            f"Operating Income: {_fmt_money(is1.get('operating_income'),basis)}",
            f"EBITDA: {_fmt_money(is1.get('ebitda'),basis)}",
            f"EBITDA Margin: {_fmt_pct(dm1.get('ebitda_margin'))}",
            f"Net Income: {_fmt_money(is1.get('net_income'),basis)}",
            f"Total Debt: {_fmt_money(bs1.get('total_debt'),basis)}",
            f"Leverage: {_fmt_x(dm1.get('leverage_total_debt_to_ebitda'))}",
            f"CFO: {_fmt_money(cf1.get('cfo'),basis)}",
            f"Capex: {_fmt_money(abs_outflow(cf1.get('capex')),basis)}",
            f"FCF: {_fmt_money(dm1.get('free_cash_flow'),basis)}",
            f"FCC: {_fmt_x(dm1.get('fcc'))}",
            f"Cash: {_fmt_money(bs1.get('cash'),basis)}",
            f"\nYoY CHANGES ({p1.period_name} → {p0.period_name}):",
        ]
        for label, nv, ov in [
            ("Revenue", is0.get('revenue'), is1.get('revenue')),
            ("Cost of Sales", is0.get('cost_of_sales'), is1.get('cost_of_sales')),
            ("EBITDA", is0.get('ebitda'), is1.get('ebitda')),
            ("CFO", cf0.get('cfo'), cf1.get('cfo')),
            ("FCF", dm0.get('free_cash_flow'), dm1.get('free_cash_flow')),
        ]:
            chg = _safe_pct_change(nv, ov)
            if chg is not None: lines.append(f"{label}: {chg*100:+.1f}%")
        if dm0.get('ebitda_margin') is not None and dm1.get('ebitda_margin') is not None:
            lines.append(f"EBITDA Margin: {(dm0['ebitda_margin']-dm1['ebitda_margin'])*100:+.1f}pp")
    return "\n".join(lines)


# ============================================================
# MEMO SYSTEM PROMPT
# ============================================================

MEMO_SYSTEM_PROMPT = """You are a senior commercial credit analyst writing a structured credit memo.

FORMATTING RULES — follow exactly:
- Use ## for top-level sections
- Use ### for financial sub-sections and segment sub-sections
- Use #### for segment names inside Segment Analysis blocks
- Put a blank line between every section, sub-section, and segment block
- Top-level bullets use - at column 0
- Sub-bullets inside a segment block use   - (2-space indent)
- Bold key dollar figures: **$1,234MM**
- Never write placeholder text like [+/-X]% — use actual figures or write: Not available.
- Never use "Not disclosed in MD&A" as a segment name or header

STRUCTURE FOR REVENUE / COGS / GROSS MARGIN SECTIONS:
Each must follow this exact order on separate lines:
1. One YoY summary sentence with actual $ and %
2. Blank line
3. **Consolidated Drivers:** (own line)
4. Driver bullets
5. Blank line
6. **Offsetting Factors:** (own line, omit if none)
7. Offsetting bullets
8. Blank line
9. **Segment Analysis:** (own line)
10. Blank line
11. #### [Segment Name] (own line)
12. Blank line
13. - [Metric]: **$actual** vs **$prior** (change of **$delta**)
14.   - Driver 1 (2-space indent)
15.   - Driver 2
16.   - Offsetting: [factor]
17. Blank line
18. #### [Next Segment] ... repeat
"""

MEMO_MODEL = os.getenv("MEMO_MODEL", "gpt-4o")
MEMO_MAX_TOKENS = int(os.getenv("MEMO_MAX_TOKENS", "10000"))
MEMO_TEMPERATURE = float(os.getenv("MEMO_TEMPERATURE", "0.1"))


# ============================================================
# UNDERWRITING MEMO GENERATOR
# ============================================================

def generate_underwriting_memo(
    borrower: BorrowerProfile,
    covenants: CovenantSet,
    extracted: ExtractionResult,
    mda_summary: str = "",
) -> str:
    extracted = validate_and_compute(extracted)
    if not extracted.periods:
        return "## Credit Memo\n\nNo financial periods available for analysis."

    basis = extracted.statement_basis or "actual"
    extracted_section = _format_extracted_data(extracted, basis)
    metric_data = _build_metric_summary(extracted, basis)

    p0 = extracted.periods[0]
    p1 = extracted.periods[1] if len(extracted.periods) >= 2 else None

    def _m(x): return "N/A" if x is None else f"${float(x):,.0f}MM"
    def _xf(x): return "N/A" if x is None else f"{float(x):.2f}x"
    def _pc(x): return "N/A" if x is None else f"{float(x)*100:.1f}%"
    def _yoy(n, o):
        if n is None or o is None or float(o) == 0: return "N/A"
        return f"{(float(n)-float(o))/float(o)*100:+.1f}%"

    is0 = p0.income_statement or {}; bs0 = p0.balance_sheet or {}
    cf0 = p0.cash_flow or {}; dm0 = p0.derived_metrics or {}
    is1 = (p1.income_statement or {}) if p1 else {}
    bs1 = (p1.balance_sheet or {}) if p1 else {}
    cf1 = (p1.cash_flow or {}) if p1 else {}
    dm1 = (p1.derived_metrics or {}) if p1 else {}

    period0 = p0.period_name
    period1 = p1.period_name if p1 else "Prior Period"

    fcf0 = dm0.get('free_cash_flow'); fcf1 = dm1.get('free_cash_flow')
    em_delta = "N/A"
    if dm0.get('ebitda_margin') is not None and dm1.get('ebitda_margin') is not None:
        em_delta = f"{(dm0['ebitda_margin']-dm1['ebitda_margin'])*100:+.1f}pp"

    rev_avail = _m(bs0.get('revolver_availability'))
    total_liq = "N/A"
    if bs0.get('cash') is not None and bs0.get('revolver_availability') is not None:
        total_liq = f"${float(bs0['cash'])+float(bs0['revolver_availability']):,.0f}MM"

    pd_line = f"{int(dm0['pd_score'])}/12 — {dm0.get('credit_rating','N/A')}" if dm0.get('pd_score') is not None else "N/A"
    z_line = f"{dm0['altman_z_score']:.2f}" if dm0.get('altman_z_score') is not None else "N/A"

    mda_is_error = not mda_summary or mda_summary.startswith("Error") or mda_summary.startswith("MD&A extraction failed")
    mda_block = "(MD&A extraction failed — segment driver data not available for this run.)" if mda_is_error else mda_summary

    user_prompt = f"""Write a credit memo for the borrower below using the exact structure and formatting specified.

BORROWER
Name:            {borrower.name}
Industry:        {borrower.industry}
Facility Type:   {borrower.facility_type}
Use of Proceeds: {borrower.use_of_proceeds}

COVENANTS
Max Total Leverage: {covenants.max_total_leverage if covenants.max_total_leverage else "Not specified"}
Min FCC:            {covenants.min_fcc if covenants.min_fcc else "Not specified"}

PRE-COMPUTED METRICS — use these exact figures, do not recalculate:
{period0} vs {period1}

Revenue:         {_m(is0.get('revenue'))} vs {_m(is1.get('revenue'))} ({_yoy(is0.get('revenue'), is1.get('revenue'))} YoY)
COGS:            {_m(is0.get('cost_of_sales'))} vs {_m(is1.get('cost_of_sales'))}
SG&A:            {_m(is0.get('sga_expense'))} vs {_m(is1.get('sga_expense'))}
Operating Inc:   {_m(is0.get('operating_income'))} vs {_m(is1.get('operating_income'))}
EBITDA:          {_m(is0.get('ebitda'))} vs {_m(is1.get('ebitda'))} ({_yoy(is0.get('ebitda'), is1.get('ebitda'))} YoY)
EBITDA Margin:   {_pc(dm0.get('ebitda_margin'))} vs {_pc(dm1.get('ebitda_margin'))} ({em_delta})
Net Income:      {_m(is0.get('net_income'))}
Total Debt:      {_m(bs0.get('total_debt'))} vs {_m(bs1.get('total_debt'))} ({_yoy(bs0.get('total_debt'), bs1.get('total_debt'))} YoY)
Leverage:        {_xf(dm0.get('leverage_total_debt_to_ebitda'))} vs {_xf(dm1.get('leverage_total_debt_to_ebitda'))}
CFO:             {_m(cf0.get('cfo'))} vs {_m(cf1.get('cfo'))} ({_yoy(cf0.get('cfo'), cf1.get('cfo'))} YoY)
Capex:           {_m(cf0.get('capex'))} vs {_m(cf1.get('capex'))} ({_yoy(cf0.get('capex'), cf1.get('capex'))} YoY)
FCF:             {_m(fcf0)} vs {_m(fcf1)} ({_yoy(fcf0, fcf1)} YoY)
FCC:             {_xf(dm0.get('fcc'))} vs {_xf(dm1.get('fcc'))}
Cash:            {_m(bs0.get('cash'))} vs {_m(bs1.get('cash'))} ({_yoy(bs0.get('cash'), bs1.get('cash'))} YoY)
Revolver Size:   {_m(bs0.get('revolver_facility_size'))}
Revolver Drawn:  {_m(bs0.get('revolver_borrowings'))}
Revolver Avail:  {rev_avail}
Total Liquidity: {total_liq}
PD Score:        {pd_line}
Altman Z-Score:  {z_line}

MD&A SEGMENT DRIVER INSIGHTS:
{mda_block}

VALIDATION FLAGS:
{json.dumps(extracted.validation_flags, indent=2) if extracted.validation_flags else "None"}

---

Write the memo now. Use this exact structure. Every header on its own line. Every bullet on its own line. Blank line between every section.

## Executive Summary

- [Bullet 1: overall credit quality with actual figures]
- [Bullet 2: key performance metrics with actual figures]
- [Bullet 3: liquidity and structure]

---

## Credit Risk Assessment

### PD Score: {pd_line}

- Scale: 1-3 Investment Grade | 4-6 Non-Investment Grade | 7-9 Speculative | 10-11 High Risk | 12 Default
- Interpretation: [1 sentence]

### Altman Z-Score: {z_line}

- Zones: Z > 2.9 Safe | 1.23–2.9 Grey | Z < 1.23 Distress
- Assessment: [1 sentence on zone and implication for this borrower]

### Key Credit Metrics

- **Leverage:** {_xf(dm0.get('leverage_total_debt_to_ebitda'))}
- **FCC:** {_xf(dm0.get('fcc'))}
- **EBITDA Margin:** {_pc(dm0.get('ebitda_margin'))}
- **Liquidity:** Cash {_m(bs0.get('cash'))} + {rev_avail} revolver availability = {total_liq} total

---

## Financial Performance Analysis

### Revenue

[One sentence: Revenue [increased/decreased] [%] YoY from [prior] to [current].]

**Consolidated Drivers:**
- [Driver 1 from MD&A — with dollar amount]
- [Driver 2 from MD&A — with dollar amount]

**Offsetting Factors:**
- [Factor — with dollar amount, or omit this entire block if none mentioned]

**Segment Analysis:**

#### [Exact Segment Name 1]

- Revenue: **$[actual]MM** vs **$[prior]MM** (change of **$[delta]MM**)
  - [Driver 1 — indented sub-bullet with dollar amount]
  - [Driver 2 — indented sub-bullet with dollar amount]
  - Offsetting: [factor — indented, or omit]

#### [Exact Segment Name 2]

- Revenue: **$[actual]MM** vs **$[prior]MM** (change of **$[delta]MM**, or: Not separately disclosed.)
  - [Driver 1]
  - Offsetting: [factor or: Not disclosed]

#### [Exact Segment Name 3]

- Revenue: [figures or: Not separately disclosed.]
  - [Driver 1]

### Cost of Goods Sold

[One sentence with actual YoY figures.]

**Consolidated Drivers:**
- [Driver 1]
- [Driver 2]

**Offsetting Factors:**
- [Factor or omit block]

**Segment Analysis:**

#### [Exact Segment Name 1]

- COGS: **$[actual]MM** vs **$[prior]MM**
  - [Driver 1]
  - [Driver 2]
  - Offsetting: [factor]

#### [Exact Segment Name 2]

- COGS: [Not separately disclosed, or figures]

#### [Exact Segment Name 3]

- COGS: [Not separately disclosed, or figures]

### Gross Margin

[One sentence: Gross margin [expanded/compressed] from [prior]% to [current]% ([delta]pp).]

**Consolidated Drivers:**
- [Driver 1]
- [Driver 2]

**Offsetting Factors:**
- [Factor or omit block]

**Segment Analysis:**

#### [Exact Segment Name 1]

- Gross Margin: **$[actual]MM** ([actual]%) vs **$[prior]MM** ([prior]%)
  - [Driver 1]
  - [Driver 2]
  - Offsetting: [factor]

#### [Exact Segment Name 2]

- Gross Margin: [Not separately disclosed, or figures]

#### [Exact Segment Name 3]

- Gross Margin: [Not separately disclosed, or figures]

### EBITDA and Margins

- **EBITDA:** {_m(is0.get('ebitda'))} vs {_m(is1.get('ebitda'))} ({_yoy(is0.get('ebitda'), is1.get('ebitda'))} YoY)
- **EBITDA Margin:** {_pc(dm0.get('ebitda_margin'))} vs {_pc(dm1.get('ebitda_margin'))} ({em_delta})
- [1 sentence on key EBITDA driver]

### SG&A

- **SG&A:** {_m(is0.get('sga_expense'))} vs {_m(is1.get('sga_expense'))} ({_yoy(is0.get('sga_expense'), is1.get('sga_expense'))} YoY)
- [1 sentence on SG&A trend if available from MD&A]

### Cash Flow

- **Operating Cash Flow:** {_m(cf0.get('cfo'))} vs {_m(cf1.get('cfo'))} ({_yoy(cf0.get('cfo'), cf1.get('cfo'))} YoY)
- **Capex:** {_m(cf0.get('capex'))} vs {_m(cf1.get('capex'))} ({_yoy(cf0.get('capex'), cf1.get('capex'))} YoY)
- **Free Cash Flow:** {_m(fcf0)} vs {_m(fcf1)} ({_yoy(fcf0, fcf1)} YoY)

### Leverage and Debt Service

- **Total Debt:** {_m(bs0.get('total_debt'))} vs {_m(bs1.get('total_debt'))} ({_yoy(bs0.get('total_debt'), bs1.get('total_debt'))} YoY)
- **Leverage:** {_xf(dm1.get('leverage_total_debt_to_ebitda'))} → {_xf(dm0.get('leverage_total_debt_to_ebitda'))}
- **FCC:** {_xf(dm1.get('fcc'))} → {_xf(dm0.get('fcc'))}

### Liquidity

- **Cash:** {_m(bs0.get('cash'))} vs {_m(bs1.get('cash'))} ({_yoy(bs0.get('cash'), bs1.get('cash'))} YoY)
- **Revolving Credit Facility:**
  - Facility Size: {_m(bs0.get('revolver_facility_size'))}
  - Borrowings: {_m(bs0.get('revolver_borrowings'))}
  - Available Capacity: {rev_avail}

---

## Credit Risks

- [Risk 1 — specific, with actual figure]
- [Risk 2 — specific, with actual figure]
- [Risk 3 — specific, with actual figure]

## Mitigants

- [Mitigant 1 — specific, with actual figure]
- [Mitigant 2 — specific, with actual figure]

## Covenant Compliance

- **Leverage:** {_xf(dm0.get('leverage_total_debt_to_ebitda'))} vs max {covenants.max_total_leverage if covenants.max_total_leverage else 'not specified'} — [compliance status and headroom]
- **FCC:** {_xf(dm0.get('fcc'))} vs min {covenants.min_fcc if covenants.min_fcc else 'not specified'} — [compliance status and headroom]

## Recommendation

[Approve / Decline / Approve with conditions — 2-3 sentences with rationale using actual figures.]

## Data Gaps and Limitations

- [Gap 1]
- [Gap 2]
"""

    resp = get_client().chat.completions.create(
        model=MEMO_MODEL,
        temperature=MEMO_TEMPERATURE,
        max_tokens=MEMO_MAX_TOKENS,
        messages=[
            {"role": "system", "content": MEMO_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    raw_memo = resp.choices[0].message.content or ""
    formatted_memo = _format_memo_markdown(raw_memo)
    return (formatted_memo.strip() + "\n\n---\n\n" + extracted_section).strip()